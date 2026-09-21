"""Video output and resumable chunk state."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_chunk_state(
    chunk_index: int,
    state: dict[str, object],
    *,
    state_output_dir: Path,
    run_contract: dict,
    history_selection: list,
) -> None:
    state = dict(state)
    state["run_contract"] = run_contract
    state["history_selection"] = history_selection
    checkpoint_path = state_output_dir / f"chunk_{chunk_index:03d}_complete.pt"
    temporary_path = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    metadata = {
        "format": state["format"],
        "completed_chunk_index": state["completed_chunk_index"],
        "next_chunk_index": state["next_chunk_index"],
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256(checkpoint_path),
        "run_contract": run_contract,
    }
    metadata_path = state_output_dir / f"chunk_{chunk_index:03d}_complete.json"
    temporary_metadata_path = metadata_path.with_suffix(".json.tmp")
    temporary_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_metadata_path, metadata_path)
    latest_path = state_output_dir / "latest.json"
    temporary_latest_path = latest_path.with_suffix(".json.tmp")
    temporary_latest_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_latest_path, latest_path)
    print(f"[worldcrafter] saved resumable state {checkpoint_path}", flush=True)


def assemble_resumed_video(
    output_path: Path, chunk_output_dir: Path, final_chunk_index: int
) -> None:
    chunk_paths = [
        chunk_output_dir / f"chunk_{index:03d}_33f.mp4"
        for index in range(final_chunk_index + 1)
    ]
    missing_chunks = [str(path) for path in chunk_paths if not path.is_file()]
    if missing_chunks:
        raise FileNotFoundError(
            "cannot assemble resumed output; missing chunks: "
            + ", ".join(missing_chunks)
        )
    concat_path = output_path.with_suffix(".concat.txt")
    concat_path.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in chunk_paths),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            str(output_path),
        ],
        check=True,
    )
    concat_path.unlink()
