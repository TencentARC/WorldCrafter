#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


CHUNK_FRAMES = 33
FPS = 16


def rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )


def parse_event(event: str) -> tuple[str, float]:
    match = re.fullmatch(r"(f|b|l|r|yr|yl)([0-9]+(?:\.[0-9]+)?)", event)
    if match is None:
        raise ValueError(f"unsupported camera event: {event}")
    return match.group(1), float(match.group(2))


def translation_direction(rotation: np.ndarray, kind: str) -> np.ndarray:
    axis = rotation[:, 2].copy() if kind in {"f", "b"} else rotation[:, 0].copy()
    axis[1] = 0.0
    sign = 1.0 if kind in {"f", "r"} else -1.0
    return sign * axis / np.linalg.norm(axis)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_trajectory(events: list[str]) -> tuple[np.ndarray, list[dict[str, object]]]:
    world = np.eye(4, dtype=np.float64)
    alpha = np.arange(CHUNK_FRAMES, dtype=np.float64) / CHUNK_FRAMES
    chunks: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for chunk_index, event in enumerate(events):
        kind, amount = parse_event(event)
        start = world.copy()
        poses = np.repeat(start[None], CHUNK_FRAMES, axis=0)
        end = start.copy()
        if kind in {"f", "b", "l", "r"}:
            delta = amount * translation_direction(start[:3, :3], kind)
            poses[:, :3, 3] = start[:3, 3] + alpha[:, None] * delta
            end[:3, 3] = start[:3, 3] + delta
        else:
            signed_degrees = amount if kind == "yr" else -amount
            for frame_index, fraction in enumerate(alpha):
                poses[frame_index, :3, :3] = (
                    rotation_y(signed_degrees * fraction) @ start[:3, :3]
                )
            end[:3, :3] = rotation_y(signed_degrees) @ start[:3, :3]
        chunks.append(poses)
        records.append(
            {
                "chunk_index": chunk_index,
                "event": event,
                "logical_start_c2w": start.tolist(),
                "logical_end_c2w": end.tolist(),
            }
        )
        world = end
    return np.concatenate(chunks), records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a no-zero-velocity WorldCore trajectory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--events", nargs="+", required=True)
    args = parser.parse_args()

    camera, records = build_trajectory(args.events)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera_path = args.output_dir / "camera.npy"
    np.save(camera_path, camera[:, :3, :4])
    manifest = {
        "format": "worldcore_camera_trajectory_v1",
        "fps": FPS,
        "chunk_frames": CHUNK_FRAMES,
        "num_chunks": len(args.events),
        "num_frames": int(camera.shape[0]),
        "zero_velocity_at_boundaries": False,
        "sampling": "half_open_linear_no_zero_velocity",
        "motion_sequence": args.events,
        "camera": camera_path.name,
        "camera_semantics": "global metric c2w; chunk-relative UCPE poses are derived during inference",
        "camera_dtype": "float64 to preserve exact chunk-relative pose derivation",
        "sha256": {"camera": sha256(camera_path)},
        "chunks": records,
    }
    manifest_path = args.output_dir / "trajectory.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
