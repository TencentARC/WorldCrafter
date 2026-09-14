from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "weights" / "WorldCore_base"
DEFAULT_IMAGE = ROOT / "test" / "images" / "023_Cat_Vac.png"
DEFAULT_I2V_CAMERA = ROOT / "test" / "poses" / "camera.npy"
DEFAULT_T2V_CAMERA = ROOT / "test" / "poses" / "t2v_camera.npy"
DEFAULT_I2V_PROMPT = ROOT / "test" / "prompts" / "i2v.txt"
DEFAULT_T2V_PROMPT = ROOT / "test" / "prompts" / "t2v.txt"
DEFAULT_NEGATIVE_PROMPT = ROOT / "test" / "prompts" / "negative.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WorldCore camera-controlled image-to-video and text-to-video inference"
    )
    parser.add_argument("--mode", choices=("i2v", "t2v"), default="i2v")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--camera-path", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-path", type=Path)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--negative-prompt-path", type=Path, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--chunk-output-dir", type=Path)
    parser.add_argument("--state-output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--reference-chunk-dir", type=Path)
    parser.add_argument("--reference-chunk-count", type=int, default=0)
    parser.add_argument("--num-chunks", type=int)
    parser.add_argument("--stop-after-chunk", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--image-noise-sigma-min", type=float, default=0.111)
    parser.add_argument("--image-noise-sigma-max", type=float, default=0.135)
    parser.add_argument("--camera-x-fov", type=float, default=100.0)
    parser.add_argument("--camera-xi", type=float, default=0.0)
    parser.add_argument("--memory-fov-h-deg", type=float, default=100.0)
    parser.add_argument("--memory-fov-v-deg", type=float, default=71.13349068444832)
    parser.add_argument("--memory-fov-samples-per-axis", type=int, default=10)
    parser.add_argument(
        "--attention-backend",
        choices=("native", "auto", "flash_hub", "_flash_3_hub"),
        default="native",
    )
    parser.add_argument("--enable-compile", action="store_true")
    return parser


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"prompt file is empty: {path}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.num_chunks is not None and args.num_chunks <= 0:
        raise ValueError("--num-chunks must be positive")
    if args.stop_after_chunk is not None and args.stop_after_chunk < 0:
        raise ValueError("--stop-after-chunk must be non-negative")
    if args.resume_from is not None and args.chunk_output_dir is None:
        raise ValueError("--resume-from requires --chunk-output-dir")
    if args.reference_chunk_count < 0:
        raise ValueError("--reference-chunk-count must be non-negative")
    if args.reference_chunk_count and args.reference_chunk_dir is None:
        raise ValueError("--reference-chunk-count requires --reference-chunk-dir")

    if args.camera_path is None:
        args.camera_path = DEFAULT_I2V_CAMERA if args.mode == "i2v" else DEFAULT_T2V_CAMERA
    if args.prompt is None:
        prompt_path = args.prompt_path
        if prompt_path is None:
            prompt_path = DEFAULT_I2V_PROMPT if args.mode == "i2v" else DEFAULT_T2V_PROMPT
        args.prompt_path = prompt_path
        args.prompt = _read_text(prompt_path)
    elif args.prompt_path is not None:
        raise ValueError("use either --prompt or --prompt-path, not both")

    if args.negative_prompt is None:
        args.negative_prompt = _read_text(args.negative_prompt_path)
    if args.mode == "i2v":
        args.image_path = args.image_path or DEFAULT_IMAGE
    elif args.image_path is not None:
        raise ValueError("--image-path is only valid with --mode i2v")

    if args.output_path is None:
        args.output_path = ROOT / "outputs" / (
            "023_Cat_Vac.mp4" if args.mode == "i2v" else "prompt_02.mp4"
        )
    return args


__all__ = ["build_parser", "parse_args"]
