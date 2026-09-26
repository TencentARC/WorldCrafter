"""Launch the compiled I2V demo on one or two GPUs."""

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
    parser.add_argument("--devices", default="0", help="One or two visible GPU indices, e.g. 0,1")
    args = parser.parse_args()
    devices = args.devices.split(",")
    if len(devices) not in (1, 2) or len(set(devices)) != len(devices) or not all(d.isdigit() for d in devices):
        parser.error("--devices must contain one or two distinct GPU indices")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        available = visible.split(",")
        if any(int(device) >= len(available) for device in devices):
            parser.error("--devices exceeds the GPUs in CUDA_VISIBLE_DEVICES")
        devices = [available[int(device)] for device in devices]
    os.environ.update(
        WORLDCRAFTER_DEMO_MODEL=str(args.model_path.resolve()),
        WORLDCRAFTER_DEMO_DATA=str(args.output_dir.resolve()),
        WORLDCRAFTER_DEMO_MOCK=str(int(args.mock)),
        CUDA_VISIBLE_DEVICES=",".join(devices),
        WORLDCRAFTER_DEMO_GPUS=str(len(devices)),
    )
    import uvicorn

    uvicorn.run("demo.server:app", host=args.host, port=args.port, ws="websockets")


if __name__ == "__main__":
    main()
