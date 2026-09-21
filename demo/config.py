"""Serving paths and the fixed I2V inference contract."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("WORLDCRAFTER_DEMO_DATA", ROOT.parent / "output/demo"))
MODEL_PATH = Path(
    os.environ.get("WORLDCRAFTER_DEMO_MODEL", ROOT.parent / "weights/WorldCrafter-Fast")
)
DEFAULT_PRESET = "teaser28__I2V__01_socrates"
ROUTE = ("A",) * 5 + ("B",)


def presets():
    return json.loads((ROOT / "configs/presets.json").read_text())
