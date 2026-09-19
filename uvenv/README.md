# WorldCrafter uv environment

Supported runtime: Linux x86_64, Python 3.11, CUDA 12.8, Hopper GPU (H100/H200).

From the project root, install the complete environment with:

```bash
uv sync --project uvenv --frozen
```

The default environment is `uvenv/.venv`. Run with its Python directly or use
`uv run --project uvenv --frozen ...`. No activation hook, token, Hugging Face
kernel cache, post-install command, or inference-time package installation is
required. The existing `sync_env.sh` remains a compatibility wrapper for callers
that set `ENV_DIR`.

A platform-provided `UV_PROJECT_ENVIRONMENT` overrides uv's default environment
location. Either keep it consistently for both `uv sync` and `uv run`, or unset
it once in your shell to use `uvenv/.venv`. The inference wrappers use
`uv run --frozen --no-sync` and never install dependencies or edit environment
variables. The Python inference code contains no environment setup.

`pyproject.toml` and `uv.lock` include the local wheel
`wheels/worldcrafter_fa3_kernel-1.0.0-py3-none-linux_x86_64.whl` (about 202 MiB).
Keep this file when copying/materializing the environment. It is a required uv
dependency, not a generated inference output; generic filters that discard
files larger than 100 MiB must make an explicit exception for this wheel.
Keep this environment spec in the repository's `uvenv` directory, alongside the
WorldCrafter source referenced by the editable dependency. It is not a
source-free project bundle. Base and fast use this same dependency graph.

The wheel contains the pinned FA3 Torch 2.10/CUDA 12.8 binary and matching
Python files. uv checks its locked SHA256 during installation. It installs into
`site-packages/worldcrafter_fa3_kernel/kernel`. An installed `.pth` registers that
local path with the existing kernels library, so `get_kernel` calls use
the installed artifact without network access. This registration does not
download or install anything. Explicit user `LOCAL_KERNELS` overrides remain
respected; unset them when validating this environment.

Pinned versions include Torch 2.10.0+cu128, kernels 0.13.0, diffusers 0.37.0,
transformers 5.3.0. The environment also includes worldcrafter-fa3-kernel 1.0.0 and flash-attn-3.

Artifact checksums:

- wheel: `3946814b56829476182e5ca57e0ee44f1dbe29bf1bcc00a8cafa546384ecb5e9`
- FA3 binary: `8aa9bb49e174872be4e2cb8c2608f2d66de29b513f704fb8d91771f9cd39df56`
- interface: `cb46c3d89606305817ea20eb7b7cc9376b013eeabfd7637a328e7967e1d008d4`
