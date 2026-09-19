#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec uv run --project "${ROOT}/uvenv" --frozen --no-sync python "${ROOT}/inference.py" --model-type fast --mode i2v "$@"
