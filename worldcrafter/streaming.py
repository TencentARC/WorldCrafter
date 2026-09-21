"""Persistent I2V sessions using the same Fast sampler as offline inference."""

import torch
from diffusers.utils.torch_utils import randn_tensor
from .diffusers.pipeline import _render_repencoder_memory_latents
from .fast.sampling import sample_fast


def stream_chunks(
    pipe, *, image, prompt, generator, provider, max_chunks, cancel, on_step=None
):
    """Yield readiness, then accept cumulative local/global poses for each chunk.

    Advance and close this generator under inference_mode on its owning thread.
    The RNG calls and conditioning layout follow the offline I2V pipeline.
    """
    if image is None or not 1 <= max_chunks <= 50:
        raise ValueError("Interactive I2V requires an image and 1..50 chunks")
    device = pipe._execution_device
    pipe.fast_inference_mode = "i2v"
    pipe._guidance_scale = 1.0
    pipe._attention_kwargs = None
    pipe._interrupt = False
    pipe._current_timestep = None
    mean = torch.tensor(
        pipe.vae.config.latents_mean, device=device, dtype=pipe.vae.dtype
    ).view(1, 16, 1, 1, 1)
    std = 1.0 / torch.tensor(
        pipe.vae.config.latents_std, device=device, dtype=pipe.vae.dtype
    ).view(1, 16, 1, 1, 1)
    embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt="",
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        device=device,
        max_sequence_length=512,
    )
    embeds = embeds.to(pipe.transformer.dtype)
    pixels = pipe.video_processor.preprocess(image, height=384, width=640)
    prefix, fake = pipe.prepare_image_latents(
        pixels,
        latents_mean=mean,
        latents_std=std,
        num_latent_frames_per_chunk=9,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    sigma = torch.rand(1, device=device, generator=generator) * (0.135 - 0.111) + 0.111
    prefix = (
        sigma * randn_tensor(prefix.shape, generator=generator, device=device)
        + (1 - sigma) * prefix
    )
    sigma = torch.rand(1, device=device, generator=generator) * (0.135 - 0.111) + 0.111
    fake = (
        sigma * randn_tensor(fake.shape, generator=generator, device=device)
        + (1 - sigma) * fake
    )
    history = torch.cat([torch.zeros(1, 16, 3, 48, 80, device=device), fake], dim=2)
    generated = history[:, :, :0]
    ids = torch.arange(17).split([1, 4, 2, 1, 9])
    short_ids = torch.cat([ids[0], ids[3]]).unsqueeze(0)

    def step_callback(*args):
        cancel()
        return on_step(*args) if on_step else args[-1]

    camera = yield {"ready": True}
    for index in range(max_chunks):
        cancel()
        camera = dict(camera, c2w=camera["retrieval_pose"])
        memory = (
            generated.new_zeros(1, 16, 4, 48, 80)
            if index == 0
            else _render_repencoder_memory_latents(
                memory_provider=provider,
                generated_latents=generated,
                camera_trajectory=camera,
                chunk_index=index,
                num_latent_frames_per_chunk=9,
                vae_scale_factor_temporal=4,
                generator=generator,
            )
        )
        mid, recent = history[:, :, -3:].split([2, 1], dim=2)
        short = torch.cat([prefix, recent], dim=2)
        latents = pipe.prepare_latents(
            1,
            16,
            384,
            640,
            33,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=None,
        )
        # Upsampling noise retains the sampler's independent default generator.
        with pipe.progress_bar(total=6) as progress:
            latents = sample_fast(
                pipe,
                latents=latents,
                pyramid_num_stages=3,
                prompt_embeds=embeds,
                guidance_scale=1.0,
                indices_hidden_states=ids[4].unsqueeze(0),
                indices_latents_history_short=short_ids,
                indices_latents_history_mid=ids[2].unsqueeze(0),
                indices_latents_history_long=ids[1].unsqueeze(0),
                latents_history_short=short,
                latents_history_mid=mid,
                latents_history_long=memory,
                attention_kwargs={},
                camera_trajectory=camera,
                num_latent_frames_per_chunk=9,
                chunk_index=index,
                camera_restart_each_chunk=False,
                ucpe_pixel_center=True,
                device=device,
                transformer_dtype=pipe.transformer.dtype,
                callback_on_step_end=step_callback,
                callback_on_step_end_tensor_inputs=[],
                progress_bar=progress,
            )
        cancel()
        generated = torch.cat([generated, latents], dim=2)
        history = latents[:, :, -3:].clone()
        decoded = pipe.vae.decode(
            latents.to(pipe.vae.dtype) / std + mean, return_dict=False
        )[0]
        camera = yield dict(
            chunk_index=index,
            rgb=decoded.detach().cpu(),
            latents=latents.detach().cpu(),
            rng_state=generator.get_state().cpu(),
        )
        del decoded
    pipe._current_timestep = None
