# WorldCore uv environment

This directory contains the complete reproducible inference environment
specification. From the repository root, run:

```bash
bash uvenv/sync_env.sh
source .venv/bin/activate
```

The environment is resolved from `uv.lock`, installs the repository itself in
editable mode, and obtains the Flash-Attention 3 kernel matching the installed
PyTorch/CUDA backend through Hugging Face `kernels`.

Compiled kernel caches are intentionally not stored in this repository because
they are platform-specific and can be hundreds of megabytes.
