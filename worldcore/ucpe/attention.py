import torch
import torch.nn.functional as F

from ..kernels import attn_varlen_func


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode: bool = False,
) -> torch.Tensor:
    batch, tokens, channels = q.shape
    head_dim = channels // num_heads
    q = q.view(batch, tokens, num_heads, head_dim)
    k = k.view(batch, tokens, num_heads, head_dim)
    v = v.view(batch, tokens, num_heads, head_dim)
    if compatibility_mode:
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=False,
        ).transpose(1, 2)
    else:
        out = attn_varlen_func(q, k, v)
    return out.reshape(batch, tokens, channels)
