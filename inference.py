#!/usr/bin/env python3
from __future__ import annotations

from worldcrafter.cli import parse_args, prepare_camera


def main() -> None:
    args = parse_args()
    prepare_camera(args)

    from worldcrafter.inference import WorldCrafter

    model = WorldCrafter.from_pretrained(
        args.model_path,
        model_type=args.model_type,
        device=args.device,
        height=args.height,
        width=args.width,
        seed=args.seed,
        memory_fov_h_deg=args.memory_fov_h_deg,
        memory_fov_v_deg=args.memory_fov_v_deg,
        memory_fov_samples_per_axis=args.memory_fov_samples_per_axis,
        attention_backend=args.attention_backend,
        enable_compile=args.enable_compile,
    )
    model.generate(
        mode=args.mode,
        image_path=args.image_path,
        camera_path=args.camera_path,
        local_camera_path=args.local_camera_path,
        output_path=args.output_path,
        chunk_output_dir=args.chunk_output_dir,
        state_output_dir=args.state_output_dir,
        resume_from=args.resume_from,
        stop_after_chunk=args.stop_after_chunk,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_chunks=args.num_chunks,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        fps=args.fps,
        image_noise_sigma_min=args.image_noise_sigma_min,
        image_noise_sigma_max=args.image_noise_sigma_max,
        camera_x_fov=args.camera_x_fov,
        camera_xi=args.camera_xi,
    )


if __name__ == "__main__":
    main()
