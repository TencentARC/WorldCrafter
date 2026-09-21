"""Launch the compiled, single-GPU I2V demo."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "weights/WorldCrafter-Fast",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output/demo",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    os.environ.update(
        WORLDCRAFTER_DEMO_MODEL=str(args.model_path.resolve()),
        WORLDCRAFTER_DEMO_DATA=str(args.output_dir.resolve()),
        WORLDCRAFTER_DEMO_MOCK=str(int(args.mock)),
    )
    import uvicorn

    uvicorn.run("demo.server:app", host=args.host, port=args.port, ws="websockets")


if __name__ == "__main__":
    main()
