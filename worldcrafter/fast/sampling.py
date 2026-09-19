"""Contract-driven 5+1 sampler, preserving the validated operation order."""

from typing import Any, Callable
import math
import torch
import torch.nn.functional as F
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from .contract import resolve_dmd_inference_trace
from .camera import build_ucpe_attention_kwargs_sequential_pyramid, pyramid_token_grids
from ..diffusers.pipeline import calculate_shift, optimized_scale, XLA_AVAILABLE


def sample_fast(
    self,
    latents: torch.Tensor = None,
    pyramid_num_stages: int = None,
    pyramid_num_inference_steps_list: list[int] = None,
    prompt_embeds: torch.Tensor = None,
    negative_prompt_embeds: torch.Tensor = None,
    guidance_scale: float | None = 5.0,
    indices_hidden_states: torch.Tensor = None,
    indices_latents_history_short: torch.Tensor = None,
    indices_latents_history_mid: torch.Tensor = None,
    indices_latents_history_long: torch.Tensor = None,
    latents_history_short: torch.Tensor = None,
    latents_history_mid: torch.Tensor = None,
    latents_history_long: torch.Tensor = None,
    attention_kwargs: dict | None = None,
    camera_trajectory: dict[str, Any] | None = None,
    num_latent_frames_per_chunk: int | None = None,
    chunk_index: int | None = None,
    camera_restart_each_chunk: bool = True,
    camera_translation_scale: float = 1.0,
    ucpe_pixel_center: bool = False,
    device: torch.device | None = None,
    transformer_dtype: torch.dtype = None,
    generator: torch.Generator | None = None,
    # ------------ CFG Zero ------------
    use_zero_init: bool | None = True,
    zero_steps: int | None = 1,
    # ------------ Callback ------------
    callback_on_step_end: Callable[[int, int], None]
    | PipelineCallback
    | MultiPipelineCallbacks
    | None = None,
    callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    progress_bar=None,
):
    """Denoise through the pyramid, coarsest level first.

    A distilled checkpoint walks its authenticated DMD contract.  The step
    list is used only when the pipeline is not configured as distilled.
    """
    use_dmd = bool(self.config.is_distilled)
    if use_dmd and pyramid_num_inference_steps_list is not None:
        raise ValueError(
            "DMD denoising steps come from dmd_timestep_contract.json; "
            "pyramid_num_inference_steps_list is an inference override and is not allowed."
        )
    if not use_dmd and pyramid_num_inference_steps_list is None:
        pyramid_num_inference_steps_list = [10] * int(pyramid_num_stages)

    dmd_trace = None
    if use_dmd:
        dmd_trace = resolve_dmd_inference_trace(
            self.dmd_timestep_contract,
            latent_shape=latents.shape[1:],
            history_tensors=(
                latents_history_short,
                latents_history_mid,
                latents_history_long,
            ),
            num_stages=pyramid_num_stages,
        )
    batch_size, num_channel, num_frames, height, width = latents.shape
    patch_size = self.transformer.config.patch_size

    # Each pyramid level receives its own camera input, with all levels
    # sharing the finest grid as their geometric reference.
    ucpe_attention_kwargs_per_stage = None
    if (
        camera_trajectory is not None
        and num_latent_frames_per_chunk is not None
        and chunk_index is not None
    ):
        ucpe_attention_kwargs_per_stage = (
            build_ucpe_attention_kwargs_sequential_pyramid(
                self.transformer,
                camera_trajectory,
                num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                chunk_index=chunk_index,
                token_grids=pyramid_token_grids(
                    height // patch_size[1],
                    width // patch_size[2],
                    pyramid_num_stages,
                    low_to_high=True,
                ),
                vae_scale_factor_temporal=self.vae_scale_factor_temporal,
                restart_each_chunk=camera_restart_each_chunk,
                translation_scale=camera_translation_scale,
                pixel_center=ucpe_pixel_center,
            )
        )

    latents = latents.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_frames, num_channel, height, width
    )
    for _ in range(pyramid_num_stages - 1):
        height //= 2
        width //= 2
        latents = (
            F.interpolate(
                latents,
                size=(height, width),
                mode="bilinear",
            )
            * 2
        )
    latents = latents.reshape(
        batch_size, num_frames, num_channel, height, width
    ).permute(0, 2, 1, 3, 4)

    batch_size = latents.shape[0]
    start_point_list = None
    if use_dmd:
        start_point_list = [latents]

    def shift_for(tensor):
        return calculate_shift(
            (tensor.shape[-1] * tensor.shape[-2] * tensor.shape[-3])
            // (patch_size[0] * patch_size[1] * patch_size[2]),
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )

    i = 0
    for i_s in range(pyramid_num_stages):
        controller = getattr(self, "resident_branches", None)
        stage_models = getattr(self, "stage_transformers", None)
        stage_transformer = (
            self.transformer if stage_models is None else stage_models[i_s]
        )
        if use_dmd:
            assert dmd_trace is not None
            stage_contract = self.dmd_timestep_contract.stage_tensors(
                i_s,
                empty_history=dmd_trace.empty_history,
                device=device,
            )
            self.scheduler.timesteps = stage_contract["model_timestep"]
            self.scheduler.sigmas = stage_contract["current_sigma"]
        else:
            mu = shift_for(latents)
            self.scheduler.set_timesteps(
                pyramid_num_inference_steps_list[i_s],
                i_s,
                device=device,
                mu=mu,
            )
        timesteps = self.scheduler.timesteps

        if i_s > 0:
            height *= 2
            width *= 2
            num_frames = latents.shape[2]
            latents = latents.permute(0, 2, 1, 3, 4).reshape(
                batch_size * num_frames, num_channel, height // 2, width // 2
            )
            latents = F.interpolate(latents, size=(height, width), mode="nearest")
            latents = latents.reshape(
                batch_size, num_frames, num_channel, height, width
            ).permute(0, 2, 1, 3, 4)
            # Fix the stage
            ori_sigma = (
                1 - self.scheduler.ori_start_sigmas[i_s]
            )  # the original coeff of signal
            gamma = self.scheduler.config.gamma
            alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
            beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

            batch_size, channel, num_frames, height, width = latents.shape
            noise = self.sample_block_noise(
                batch_size,
                channel,
                num_frames,
                height,
                width,
                patch_size,
                device,
                generator,
            )
            noise = noise.to(device=device, dtype=transformer_dtype)
            latents = alpha * latents + beta * noise  # To fix the block artifact

            if use_dmd:
                start_point_list.append(latents)

        stage_attention_kwargs = dict(attention_kwargs or {})
        if ucpe_attention_kwargs_per_stage is not None:
            stage_attention_kwargs.update(ucpe_attention_kwargs_per_stage[i_s])

        for idx, t in enumerate(timesteps):
            # Switch dense weights and model-owned
            # DMD LoRA/UCPE together immediately before the sixth forward.
            assert controller is not None and not controller.streaming
            assert stage_models is not None and len(timesteps) == 2
            branch = "old" if i_s == 2 and idx == 1 else "equal"
            controller.switch(branch)
            stage_transformer = stage_models[2 if branch == "old" else 0]
            if stage_models is not None:
                self.stage_forward_context = (int(chunk_index), int(i_s), int(idx))
            timestep = (
                self.dmd_timestep_contract.student_condition(
                    t, latents.shape[0], device=latents.device
                )
                if use_dmd
                else t.expand(latents.shape[0]).to(torch.int64)
            )

            with stage_transformer.cache_context("cond"):
                noise_pred = stage_transformer(
                    hidden_states=latents.to(transformer_dtype),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=stage_attention_kwargs,
                    return_dict=False,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short.to(transformer_dtype),
                    latents_history_mid=latents_history_mid.to(transformer_dtype),
                    latents_history_long=latents_history_long.to(transformer_dtype),
                )[0]

            if self.do_classifier_free_guidance:
                with stage_transformer.cache_context("uncond"):
                    noise_uncond = stage_transformer(
                        hidden_states=latents.to(transformer_dtype),
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=stage_attention_kwargs,
                        return_dict=False,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short.to(
                            transformer_dtype
                        ),
                        latents_history_mid=latents_history_mid.to(transformer_dtype),
                        latents_history_long=latents_history_long.to(transformer_dtype),
                    )[0]

                if self.config.is_cfg_zero_star:
                    noise_pred_text = noise_pred
                    positive_flat = noise_pred_text.view(batch_size, -1)
                    negative_flat = noise_uncond.view(batch_size, -1)

                    alpha = optimized_scale(positive_flat, negative_flat)
                    alpha = alpha.view(
                        batch_size, *([1] * (len(noise_pred_text.shape) - 1))
                    )
                    alpha = alpha.to(noise_pred_text.dtype)

                    if (i_s == 0 and idx <= zero_steps) and use_zero_init:
                        noise_pred = noise_pred_text * 0.0
                    else:
                        noise_pred = noise_uncond * alpha + guidance_scale * (
                            noise_pred_text - noise_uncond * alpha
                        )
                else:
                    noise_pred = noise_uncond + guidance_scale * (
                        noise_pred - noise_uncond
                    )

            if use_dmd:
                current_sigma = stage_contract["current_sigma"][idx].float()
                next_sigma = stage_contract["next_sigma"][idx].float()
                pred_image_or_video = (
                    latents.float() - current_sigma * noise_pred.float()
                )
                latents = self.dmd_timestep_contract.renoise_x0(
                    pred_image_or_video,
                    start_point_list[i_s].float(),
                    next_sigma,
                )
            else:
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    generator=generator,
                    return_dict=False,
                )[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop(
                    "negative_prompt_embeds", negative_prompt_embeds
                )

            progress_bar.update()

            if XLA_AVAILABLE:
                from ..diffusers.pipeline import xm

                xm.mark_step()

            i += 1

    return latents
