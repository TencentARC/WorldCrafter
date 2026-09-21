# WorldCrafter environment

See the [installation instructions](../README.md#2-environment) for uv and
conda setup. Both use Python 3.11 and the same project dependencies.

## Interactive demo

With uv:

```bash
uv sync --project uvenv --frozen --extra demo
```

With conda activated:

```bash
python -m pip install -e ".[demo]"
```

Install FFmpeg so that `ffmpeg` and `ffprobe` are available on `PATH`.

## Optional acceleration

The uv environment installs xformers and FlashAttention 3 from the package index.
The conda installation uses PyTorch attention by default. To add xformers
with CUDA 12.8:

```bash
python -m pip install -e ".[xformers]" \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

On H100/H200, optionally add FlashAttention 3 for xformers:

```bash
python -m pip install flash-attn-3==3.0.0 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

Different attention implementations can change generated pixels. Use the
locked uv environment when reproducing reference outputs.

## Dependency files

`pyproject.toml` defines the package dependencies and optional extras.
`requirements.txt` points to this package for `pip install -r requirements.txt`.
`uvenv/pyproject.toml` and `uvenv/uv.lock` pin the recommended CUDA environment.

If `UV_PROJECT_ENVIRONMENT` is set, uv uses that location instead of
`uvenv/.venv`.
