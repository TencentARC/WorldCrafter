from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "output" / "trajectory"
CHUNK_FRAMES = 33
FPS = 16
EVENTS = (
    *("f1",) * 2,
    *("b1",) * 4,
    *("yr45",) * 2,
    *("yl45",) * 4,
    *("r1",) * 2,
    *("yr45",) * 4,
    *("yl45",) * 2,
)


def rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float64)


def parse(event: str) -> tuple[str, float]:
    match = re.fullmatch(r"(f|b|l|r|yr|yl)([0-9]+(?:\.[0-9]+)?)", event)
    if match is None:
        raise ValueError(event)
    return match.group(1), float(match.group(2))


def direction(rotation: np.ndarray, kind: str) -> np.ndarray:
    axis = rotation[:, 2].copy() if kind in {"f", "b"} else rotation[:, 0].copy()
    axis[1] = 0.0
    sign = 1.0 if kind in {"f", "r"} else -1.0
    return sign * axis / np.linalg.norm(axis)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    world = np.eye(4, dtype=np.float64)
    alpha = np.arange(CHUNK_FRAMES, dtype=np.float64) / CHUNK_FRAMES
    chunks = []
    records = []
    for chunk_index, event in enumerate(EVENTS):
        kind, amount = parse(event)
        start = world.copy()
        poses = np.repeat(start[None], CHUNK_FRAMES, axis=0)
        end = start.copy()
        chunk_alpha = (
            np.linspace(0.0, 1.0, CHUNK_FRAMES, dtype=np.float64)
            if chunk_index == len(EVENTS) - 1
            else alpha
        )
        if kind in {"f", "b", "l", "r"}:
            delta = amount * direction(start[:3, :3], kind)
            poses[:, :3, 3] = start[:3, 3] + chunk_alpha[:, None] * delta
            end[:3, 3] = start[:3, 3] + delta
        else:
            signed = amount if kind == "yr" else -amount
            for frame_index, fraction in enumerate(chunk_alpha):
                poses[frame_index, :3, :3] = (
                    rotation_y(signed * fraction) @ start[:3, :3]
                )
            end[:3, :3] = rotation_y(signed) @ start[:3, :3]
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

    global_pose = np.concatenate(chunks)
    if len(EVENTS) != 20 or global_pose.shape != (660, 4, 4):
        raise AssertionError((len(EVENTS), global_pose.shape))
    if not np.allclose(world, np.eye(4), atol=1.0e-12, rtol=0.0):
        raise AssertionError(f"logical endpoint is not identity:\n{world}")
    if not np.allclose(global_pose[-1], np.eye(4), atol=1.0e-12, rtol=0.0):
        raise AssertionError(
            f"final trajectory frame is not identity:\n{global_pose[-1]}"
        )
    for chunk_index in range(1, len(EVENTS)):
        boundary = chunk_index * CHUNK_FRAMES
        previous, current = global_pose[boundary - 1], global_pose[boundary]
        translation = np.linalg.norm(current[:3, 3] - previous[:3, 3])
        relative = previous[:3, :3].T @ current[:3, :3]
        rotation = math.degrees(
            math.acos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        )
        if max(translation, rotation) <= 1.0e-8:
            raise AssertionError(f"zero-velocity boundary at chunk {chunk_index}")

    ROOT.mkdir(parents=True, exist_ok=True)
    camera_path = ROOT / "camera.npy"
    np.save(camera_path, global_pose[:, :3, :4])
    manifest = {
        "format": "worldcrafter_camera_trajectory_v1",
        "fps": FPS,
        "chunk_frames": CHUNK_FRAMES,
        "num_chunks": len(EVENTS),
        "num_frames": int(global_pose.shape[0]),
        "zero_velocity_at_boundaries": False,
        "motion_sequence": list(EVENTS),
        "logical_endpoint_c2w": world.tolist(),
        "logical_endpoint_identity_max_error": float(np.max(np.abs(world - np.eye(4)))),
        "final_frame_identity_max_error": float(
            np.max(np.abs(global_pose[-1] - np.eye(4)))
        ),
        "camera": camera_path.name,
        "camera_semantics": "global metric c2w; chunk-relative UCPE poses are derived during inference",
        "sha256": {"camera": sha256(camera_path)},
        "chunks": records,
    }
    (ROOT / "trajectory.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "num_chunks",
                    "num_frames",
                    "motion_sequence",
                    "logical_endpoint_identity_max_error",
                    "sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
