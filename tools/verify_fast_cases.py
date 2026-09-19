"""Validate complete fast cases against an existing byte-exact reference set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldcrafter.inference import WorldCrafter, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "inputs", "reference", "output", "model-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.cases.read_text())
    model = WorldCrafter.from_pretrained(args.model_path, model_type="fast")
    reports = []
    args.output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        rel = row["relative_dir"]
        inputs, reference, output = (
            args.inputs / rel,
            args.reference / rel,
            args.output / rel,
        )
        for filename, field in (
            ("image.png", "image_sha256"),
            ("local.npy", "local_sha256"),
            ("global.npy", "global_sha256"),
        ):
            if sha256(inputs / filename) != row[field]:
                raise ValueError(f"Input checksum mismatch: {rel}/{filename}")
        frames = int(row["num_frames"])
        result = model.generate(
            mode="i2v",
            image_path=inputs / "image.png",
            camera_path=inputs / "global.npy",
            local_camera_path=inputs / "local.npy",
            prompt=row["prompt"],
            negative_prompt=row["negative_prompt"],
            output_path=output / "video.mp4",
            chunk_output_dir=output / "chunks",
            num_chunks=frames // 33,
            num_inference_steps=6,
            guidance_scale=1.0,
            seed=42,
            fps=16,
            reference_chunk_dir=reference / "chunks",
            reference_chunk_count=frames // 33,
        )
        actual, expected = result.video_path, reference / "video.mp4"
        report = dict(
            case=rel,
            frames=frames,
            chunks=frames // 33,
            sha256=sha256(actual),
            reference_sha256=sha256(expected),
            byte_identical=actual.read_bytes() == expected.read_bytes(),
            forwards=len(result.summary["stage_model_trace"]),
            switches=result.summary["resident_branch_switches"],
            memory_calls=len(result.summary["history_selection"]),
        )
        if not report["byte_identical"] or report["forwards"] != frames // 33 * 6:
            raise RuntimeError(report)
        reports.append(report)
        (args.output / "verification.json").write_text(
            json.dumps(
                dict(
                    all_passed=len(reports) == len(rows), model_loads=1, cases=reports
                ),
                indent=2,
            )
            + "\n"
        )
        print("BYTE_IDENTICAL_CASE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
