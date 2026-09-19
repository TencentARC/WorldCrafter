import json
import math
import os
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import apply_lora_scale, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph

from ..ucpe.bridge import UcpeSelfAttention


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def pad_for_3d_conv(x, kernel_size):
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def center_down_sample_3d(x, kernel_size):
    return torch.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)


def apply_rotary_emb_transposed(
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    x_1, x_2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x_1 * cos[..., 0::2] - x_2 * sin[..., 1::2]
    out[..., 1::2] = x_1 * sin[..., 1::2] + x_2 * cos[..., 0::2]
    return out.type_as(hidden_states)


def _get_qkv_projections(attn: "WorldCrafterAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if not attn.is_cross_attention:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


class WorldCrafterOutputNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self.norm = FP32LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, original_context_length: int):
        temb = temb[:, -original_context_length:, :]
        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2).to(hidden_states.device), scale.squeeze(2).to(hidden_states.device)
        hidden_states = hidden_states[:, -original_context_length:, :]
        hidden_states = (self.norm(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        return hidden_states


class WorldCrafterAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WorldCrafterAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WorldCrafterAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        original_context_length: int = None,
    ) -> torch.Tensor:
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = apply_rotary_emb_transposed(query, rotary_emb)
            key = apply_rotary_emb_transposed(key, rotary_emb)

        if not attn.is_cross_attention and attn.is_amplify_history:
            history_seq_len = hidden_states.shape[1] - original_context_length

            if history_seq_len > 0:
                scale_key = 1.0 + torch.sigmoid(attn.history_key_scale) * (attn.max_scale - 1.0)
                if attn.history_scale_mode == "per_head":
                    scale_key = scale_key.view(1, 1, -1, 1)
                key = torch.cat([key[:, :history_seq_len] * scale_key, key[:, history_seq_len:]], dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            # Reference: https://github.com/huggingface/diffusers/pull/12909
            parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WorldCrafterAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WorldCrafterAttnProcessor
    _available_processors = [WorldCrafterAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: int | None = None,
        cross_attention_dim_head: int | None = None,
        processor=None,
        is_cross_attention=None,
        is_amplify_history=False,
        history_scale_mode="per_head",  # [scalar, per_head]
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        if is_cross_attention is not None:
            self.is_cross_attention = is_cross_attention
        else:
            self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

        self.is_amplify_history = is_amplify_history
        if is_amplify_history:
            if history_scale_mode == "scalar":
                self.history_key_scale = nn.Parameter(torch.ones(1))
            elif history_scale_mode == "per_head":
                self.history_key_scale = nn.Parameter(torch.ones(heads))
            else:
                raise ValueError(f"Unknown history_scale_mode: {history_scale_mode}")
            self.history_scale_mode = history_scale_mode
            self.max_scale = 10.0

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if not self.is_cross_attention:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        original_context_length: int = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
            original_context_length,
            **kwargs,
        )


class WorldCrafterTimeTextEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        is_return_encoder_hidden_states: bool = True,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        if encoder_hidden_states is not None and is_return_encoder_hidden_states:
            encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states


class WorldCrafterRotaryPosEmbed(nn.Module):
    def __init__(self, rope_dim, theta):
        super().__init__()
        self.DT, self.DY, self.DX = rope_dim
        self.theta = theta
        self.register_buffer("freqs_base_t", self._get_freqs_base(self.DT), persistent=False)
        self.register_buffer("freqs_base_y", self._get_freqs_base(self.DY), persistent=False)
        self.register_buffer("freqs_base_x", self._get_freqs_base(self.DX), persistent=False)

    def _get_freqs_base(self, dim):
        return 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))

    @torch.no_grad()
    def get_frequency_batched(self, freqs_base, pos):
        freqs = torch.einsum("d,bthw->dbthw", freqs_base, pos)
        freqs = freqs.repeat_interleave(2, dim=0)
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    @lru_cache(maxsize=32)
    def _get_spatial_meshgrid(self, height, width, device_str):
        device = torch.device(device_str)
        grid_y_coords = torch.arange(height, device=device, dtype=torch.float32)
        grid_x_coords = torch.arange(width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(grid_y_coords, grid_x_coords, indexing="ij")
        return grid_y, grid_x

    @torch.no_grad()
    def forward(self, frame_indices, height, width, device):
        batch_size = frame_indices.shape[0]
        num_frames = frame_indices.shape[1]

        frame_indices = frame_indices.to(device=device, dtype=torch.float32)
        grid_y, grid_x = self._get_spatial_meshgrid(height, width, str(device))

        grid_t = frame_indices[:, :, None, None].expand(batch_size, num_frames, height, width)
        grid_y_batch = grid_y[None, None, :, :].expand(batch_size, num_frames, -1, -1)
        grid_x_batch = grid_x[None, None, :, :].expand(batch_size, num_frames, -1, -1)

        freqs_cos_t, freqs_sin_t = self.get_frequency_batched(self.freqs_base_t, grid_t)
        freqs_cos_y, freqs_sin_y = self.get_frequency_batched(self.freqs_base_y, grid_y_batch)
        freqs_cos_x, freqs_sin_x = self.get_frequency_batched(self.freqs_base_x, grid_x_batch)

        result = torch.cat([freqs_cos_t, freqs_cos_y, freqs_cos_x, freqs_sin_t, freqs_sin_y, freqs_sin_x], dim=0)

        return result.permute(1, 0, 2, 3, 4)


@maybe_allow_in_graph
class WorldCrafterTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        guidance_cross_attn: bool = False,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",  # [scalar, per_head]
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WorldCrafterAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WorldCrafterAttnProcessor(),
            is_amplify_history=is_amplify_history,
            history_scale_mode=history_scale_mode,
        )

        # 2. Cross-attention
        self.attn2 = WorldCrafterAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WorldCrafterAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # 4. Guidance cross-attention
        self.guidance_cross_attn = guidance_cross_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        original_context_length: int = None,
        camera_control_ucpe_input: dict | None = None,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        history_seq_len = hidden_states.shape[1] - original_context_length if original_context_length is not None else 0
        current_norm_hidden_states = norm_hidden_states[:, history_seq_len:, :]

        if hasattr(self, "cam_self_attn") and camera_control_ucpe_input is not None:
            if self.cam_self_attn.adaptation_method == "before":
                cam_current = self.cam_self_attn(current_norm_hidden_states, camera_control_ucpe_input)
                cam_full = torch.zeros_like(norm_hidden_states)
                cam_full[:, history_seq_len:, :] = cam_current
                norm_hidden_states = norm_hidden_states + cam_full

        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb, original_context_length)

        if hasattr(self, "cam_self_attn") and camera_control_ucpe_input is not None:
            if self.cam_self_attn.adaptation_method == "parallel":
                cam_current = self.cam_self_attn(current_norm_hidden_states, camera_control_ucpe_input)
                cam_full = torch.zeros_like(attn_output)
                cam_full[:, history_seq_len:, :] = cam_current
                attn_output = attn_output + cam_full

        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        if hasattr(self, "cam_self_attn") and camera_control_ucpe_input is not None:
            if self.cam_self_attn.adaptation_method == "after":
                cam_current = self.cam_self_attn(hidden_states[:, history_seq_len:, :], camera_control_ucpe_input)
                hidden_states = hidden_states.clone()
                hidden_states[:, history_seq_len:, :] = hidden_states[:, history_seq_len:, :] + cam_current

        # 2. Cross-attention
        if self.guidance_cross_attn:
            history_seq_len = hidden_states.shape[1] - original_context_length

            history_hidden_states, hidden_states = torch.split(
                hidden_states, [history_seq_len, original_context_length], dim=1
            )
            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states,
                None,
                None,
                original_context_length,
            )
            hidden_states = hidden_states + attn_output
            hidden_states = torch.cat([history_hidden_states, hidden_states], dim=1)
        else:
            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states,
                None,
                None,
                original_context_length,
            )
            hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class WorldCrafterTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the WorldCrafter model.

    Args:
        patch_size (`tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "patch_embedding",
        "patch_short",
        "patch_mid",
        "patch_memory",
        "condition_embedder",
        "norm",
    ]
    _no_split_modules = ["WorldCrafterTransformerBlock", "WorldCrafterOutputNorm"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
        "history_key_scale",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q", r"patch_long\..*"]
    _repeated_blocks = ["WorldCrafterTransformerBlock"]
    _cp_plan = {
        # Input split at attn level and ffn level.
        "blocks.*.attn1": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "rotary_emb": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*.attn2": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*.ffn": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        # Output gather at attn level and ffn level.
        **{f"blocks.{i}.attn1": ContextParallelOutput(gather_dim=1, expected_dims=3) for i in range(40)},
        **{f"blocks.{i}.attn2": ContextParallelOutput(gather_dim=1, expected_dims=3) for i in range(40)},
        **{f"blocks.{i}.ffn": ContextParallelOutput(gather_dim=1, expected_dims=3) for i in range(40)},
    }

    @staticmethod
    def _local_checkpoint_keys(checkpoint_dir: str) -> set[str] | None:
        index_names = (
            "diffusion_pytorch_model.safetensors.index.json",
            "model.safetensors.index.json",
            "diffusion_pytorch_model.bin.index.json",
            "pytorch_model.bin.index.json",
        )
        for index_name in index_names:
            index_path = os.path.join(checkpoint_dir, index_name)
            if os.path.exists(index_path):
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
                return set(weight_map.keys())

        for weights_name in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
            weights_path = os.path.join(checkpoint_dir, weights_name)
            if os.path.exists(weights_path):
                from safetensors import safe_open

                with safe_open(weights_path, framework="pt", device="cpu") as f:
                    return set(f.keys())

        return None

    @staticmethod
    def _resolve_local_checkpoint_dir(pretrained_model_name_or_path: str, subfolder: str | None) -> str | None:
        if not isinstance(pretrained_model_name_or_path, str):
            return None
        if not os.path.isdir(pretrained_model_name_or_path):
            return None
        checkpoint_dir = pretrained_model_name_or_path
        if subfolder is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, subfolder)
        if not os.path.isdir(checkpoint_dir):
            return None
        return checkpoint_dir

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        subfolder = kwargs.get("subfolder")
        checkpoint_dir = cls._resolve_local_checkpoint_dir(pretrained_model_name_or_path, subfolder)
        checkpoint_keys = cls._local_checkpoint_keys(checkpoint_dir) if checkpoint_dir is not None else None
        init_memory_from_short = (
            checkpoint_keys is not None
            and "patch_memory.weight" not in checkpoint_keys
            and "patch_short.weight" in checkpoint_keys
        )

        loaded = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded

        if init_memory_from_short and hasattr(model, "patch_memory") and hasattr(model, "patch_short"):
            with torch.no_grad():
                model.patch_memory.weight.copy_(model.patch_short.weight)
                if model.patch_memory.bias is not None and model.patch_short.bias is not None:
                    model.patch_memory.bias.copy_(model.patch_short.bias)
            logger.info("Initialized patch_memory weights from patch_short.")

        return loaded

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        rope_dim: tuple[int, ...] = (44, 42, 42),
        rope_theta: float = 10000.0,
        guidance_cross_attn: bool = True,
        zero_history_timestep: bool = True,
        has_multi_term_memory_patch: bool = True,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",  # [scalar, per_head]
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WorldCrafterRotaryPosEmbed(rope_dim=rope_dim, theta=rope_theta)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Initial Multi Term Memory Patch
        self.zero_history_timestep = zero_history_timestep
        self.inner_dim = inner_dim
        if has_multi_term_memory_patch:
            self.patch_short = nn.Conv3d(in_channels, self.inner_dim, kernel_size=patch_size, stride=patch_size)
            self.patch_mid = nn.Conv3d(
                in_channels,
                self.inner_dim,
                kernel_size=tuple(2 * p for p in patch_size),
                stride=tuple(2 * p for p in patch_size),
            )
            self.patch_memory = nn.Conv3d(in_channels, self.inner_dim, kernel_size=patch_size, stride=patch_size)

        # 3. Condition embeddings
        self.condition_embedder = WorldCrafterTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
        )

        # 4. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WorldCrafterTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    guidance_cross_attn=guidance_cross_attn,
                    is_amplify_history=is_amplify_history,
                    history_scale_mode=history_scale_mode,
                )
                for _ in range(num_layers)
            ]
        )

        # 5. Output norm & projection
        self.norm_out = WorldCrafterOutputNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))

        self.gradient_checkpointing = False

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        # ------------ Stage 1 ------------
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_memory=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_memory=None,
        # Backward-compatible aliases for existing stage1 callers.
        indices_latents_history_long=None,
        latents_history_long=None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if indices_latents_memory is None and indices_latents_history_long is not None:
            indices_latents_memory = indices_latents_history_long
        if latents_memory is None and latents_history_long is not None:
            latents_memory = latents_history_long

        camera_control_ucpe_input = None
        if attention_kwargs is not None:
            camera_control_ucpe_input = attention_kwargs.get("camera_control_ucpe_input", None)

        # 1. Input
        batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.config.patch_size

        # 2. Process noisy latents
        hidden_states = self.patch_embedding(hidden_states)
        _, _, post_patch_num_frames, post_patch_height, post_patch_width = hidden_states.shape

        if indices_hidden_states is None:
            indices_hidden_states = torch.arange(0, post_patch_num_frames).unsqueeze(0).expand(batch_size, -1)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        rotary_emb = self.rope(
            frame_indices=indices_hidden_states,
            height=post_patch_height,
            width=post_patch_width,
            device=hidden_states.device,
        )
        rotary_emb = rotary_emb.flatten(2).transpose(1, 2)
        original_context_length = hidden_states.shape[1]

        # 3. Process short history latents
        if latents_history_short is not None and indices_latents_history_short is not None:
            latents_history_short = latents_history_short.to(hidden_states)
            latents_history_short = self.patch_short(latents_history_short)
            _, _, _, H1, W1 = latents_history_short.shape
            latents_history_short = latents_history_short.flatten(2).transpose(1, 2)

            rotary_emb_history_short = self.rope(
                frame_indices=indices_latents_history_short,
                height=H1,
                width=W1,
                device=latents_history_short.device,
            )
            rotary_emb_history_short = rotary_emb_history_short.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_short, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_short, rotary_emb], dim=1)

        # 4. Process mid history latents
        if latents_history_mid is not None and indices_latents_history_mid is not None:
            latents_history_mid = latents_history_mid.to(hidden_states)
            latents_history_mid = pad_for_3d_conv(latents_history_mid, (2, 4, 4))
            latents_history_mid = self.patch_mid(latents_history_mid)
            latents_history_mid = latents_history_mid.flatten(2).transpose(1, 2)

            rotary_emb_history_mid = self.rope(
                frame_indices=indices_latents_history_mid,
                height=H1,
                width=W1,
                device=latents_history_mid.device,
            )
            rotary_emb_history_mid = pad_for_3d_conv(rotary_emb_history_mid, (2, 2, 2))
            rotary_emb_history_mid = center_down_sample_3d(rotary_emb_history_mid, (2, 2, 2))
            rotary_emb_history_mid = rotary_emb_history_mid.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_mid, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_mid, rotary_emb], dim=1)

        # 5. Process memory latents. These frames are not temporally/spatially downsampled.
        if latents_memory is not None and indices_latents_memory is not None:
            latents_memory = latents_memory.to(hidden_states)
            latents_memory = self.patch_memory(latents_memory)
            _, _, _, HM, WM = latents_memory.shape
            latents_memory = latents_memory.flatten(2).transpose(1, 2)

            rotary_emb_memory = self.rope(
                frame_indices=indices_latents_memory,
                height=HM,
                width=WM,
                device=latents_memory.device,
            )
            rotary_emb_memory = rotary_emb_memory.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_memory, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_memory, rotary_emb], dim=1)

        history_context_length = hidden_states.shape[1] - original_context_length

        if indices_hidden_states is not None and self.zero_history_timestep:
            timestep_t0 = torch.zeros((1), dtype=timestep.dtype, device=timestep.device)
            temb_t0, timestep_proj_t0, _ = self.condition_embedder(
                timestep_t0, encoder_hidden_states, is_return_encoder_hidden_states=False
            )
            temb_t0 = temb_t0.unsqueeze(1).expand(batch_size, history_context_length, -1)
            timestep_proj_t0 = (
                timestep_proj_t0.unflatten(-1, (6, -1))
                .view(1, 6, 1, -1)
                .expand(batch_size, -1, history_context_length, -1)
            )

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(timestep, encoder_hidden_states)
        timestep_proj = timestep_proj.unflatten(-1, (6, -1))

        if indices_hidden_states is not None and not self.zero_history_timestep:
            main_repeat_size = hidden_states.shape[1]
        else:
            main_repeat_size = original_context_length
        temb = temb.view(batch_size, 1, -1).expand(batch_size, main_repeat_size, -1)
        timestep_proj = timestep_proj.view(batch_size, 6, 1, -1).expand(batch_size, 6, main_repeat_size, -1)

        if indices_hidden_states is not None and self.zero_history_timestep:
            temb = torch.cat([temb_t0, temb], dim=1)
            timestep_proj = torch.cat([timestep_proj_t0, timestep_proj], dim=2)

        if timestep_proj.ndim == 4:
            timestep_proj = timestep_proj.permute(0, 2, 1, 3)

        # 6. Transformer blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        rotary_emb = rotary_emb.contiguous()
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    original_context_length,
                    camera_control_ucpe_input,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    original_context_length,
                    camera_control_ucpe_input,
                )

        # 7. Normalization
        hidden_states = self.norm_out(hidden_states, temb, original_context_length)
        hidden_states = self.proj_out(hidden_states)

        # 8. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
