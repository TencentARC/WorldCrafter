#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldcrafter.camera import build_trajectory, parse_trajectory, save_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a WorldCrafter camera trajectory from actions")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--actions", help='Actions, e.g. "forward1x2 yaw_left30x3 backward1"')
    source.add_argument("--actions-file", type=Path, help="TXT file of camera actions")
    source.add_argument("--events", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=Path("output/trajectory"))
    args = parser.parse_args()

    if args.actions_file is not None:
        text = args.actions_file.read_text(encoding="utf-8-sig")
    else:
        text = args.actions if args.actions is not None else " ".join(args.events)
    try:
        events, options = parse_trajectory(text)
        camera, records = build_trajectory(events, **options)
    except ValueError as error:
        parser.error(str(error))
    path = save_trajectory(args.output_dir, camera, records, events=events, options=options)
    print(f"Saved {len(records)} chunks ({len(camera)} frames) to {path}")


if __name__ == "__main__":
    main()
