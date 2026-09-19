#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_DIR="${ENV_DIR:-${SCRIPT_DIR}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

export UV_CACHE_DIR UV_LINK_MODE
uv venv --clear --python "${PYTHON_VERSION}" "${ENV_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
uv sync \
  --project "${SCRIPT_DIR}" \
  --active \
  --frozen \
  --index-strategy unsafe-best-match \
  --extra-index-url "${TORCH_INDEX_URL}"

python - <<'PY'
import torch
import diffusers
import transformers
from kernels import get_kernel

kernel = get_kernel("kernels-community/flash-attn3", version=1)
assert "site-packages/worldcrafter_fa3_kernel/kernel" in kernel.__file__, kernel.__file__
print("python environment ready")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
PY

echo "WorldCrafter environment: ${ENV_DIR}"
