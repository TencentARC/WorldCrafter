<div align="center">

# WorldCore

### Consistent Video World Model with Implicit Memory Representation

Camera-controllable autoregressive video generation with a compact, fixed-budget, 3D-aware memory.

</div>

## Overview

WorldCore is a video world model designed for long-horizon, camera-controlled generation. It compresses selected latent observations from the generated history into a compact implicit representation and renders four target-aligned memory latents for each new video chunk. The memory budget remains fixed as the rollout grows.

The released inference stack contains:

- a camera-conditioned autoregressive video diffusion transformer;
- an online trajectory-aware history selector;
- RepEncoder, which converts nine historical latent views into four target-aligned memory views;
- a default image and a closed-loop 6-DoF camera trajectory for end-to-end verification.

The technical report is available at [`assets/main.pdf`](assets/main.pdf).

## Installation

WorldCore requires Python 3.11 and a CUDA GPU. The reference configuration was validated on NVIDIA H200 GPUs with CUDA 12.8.

```bash
git clone <WORLDCORE_REPOSITORY_URL>
cd WorldCore

bash uvenv/sync_env.sh
source .venv/bin/activate
```

The single `uvenv/` directory contains the locked inference environment. The
setup script installs WorldCore in editable mode and obtains the compatible
Flash-Attention kernel for the current PyTorch/CUDA backend. Platform-specific
compiled kernel caches are not stored in the repository.

## Model weights

Place the released weights under `weights/WorldCore_base`:

```text
weights/WorldCore_base/
├── model_index.json
├── scheduler/
├── text_encoder/
├── tokenizer/
├── vae/
├── transformer/
├── adapter/
│   ├── config.json
│   ├── pytorch_lora_weights.safetensors
│   └── camera_adapter.pth
└── repencoder/
    ├── config.json
    ├── manifest.json
    └── model.safetensors
```

`WorldCore_base` is self-contained; inference does not load weights from any path outside this directory.

## Quick start

The default command uses:

- image: `test/images/023_Cat_Vac.png`;
- prompt: `test/prompts/i2v.txt`;
- global metric camera trajectory: `test/poses/camera.npy`;
- seed `42`, 50 denoising steps, 384×640 resolution, and 16 FPS.

```bash
python inference.py
```

The generated video and its JSON run record are written to:

```text
outputs/023_Cat_Vac.mp4
outputs/023_Cat_Vac.json
```

To save each completed 33-frame chunk independently:

```bash
python inference.py --chunk-output-dir outputs/023_Cat_Vac_chunks
```

Text-to-video uses its own included prompt and closed-loop camera trajectory:

```bash
python inference.py --mode t2v
```

This writes `outputs/prompt_02.mp4`. The default T2V prompt is stored in
`test/prompts/t2v.txt`, and its 50-chunk global camera trajectory is stored in
`test/poses/t2v_camera.npy`.

Custom inputs can be provided without changing the code:

```bash
python inference.py \
  --image-path path/to/image.png \
  --camera-path path/to/global_c2w.npy \
  --prompt "Your scene description." \
  --output-path outputs/custom.mp4
```

Use `--prompt-path` to load a prompt from a text file. `--num-chunks 5` limits a
run to the first five complete chunks without modifying the camera file.

## Camera trajectory format

WorldCore accepts one global metric camera trajectory containing camera-to-world
matrices in `[T, 3, 4]` or `[1, T, 3, 4]` format. Homogeneous `[T, 4, 4]`
matrices are also accepted.

- The global trajectory is used directly by online history selection and RepEncoder geometry.
- UCPE camera conditioning derives every chunk's relative trajectory internally from the same global input.
- No second local-pose file is required or accepted by the inference CLI.
- A rollout contains complete 33-frame chunks.
- The default example has 20 chunks and 660 frames, has no zero-velocity chunk boundaries, and returns to its initial pose after the final logical motion step.

The included example trajectory is:

```text
f1×2, b1×4, yr45×2, yl45×4, right1×2, yr45×4, yl45×2
```

`f`, `b`, and `right` denote forward, backward, and rightward metric translation. `yr` and `yl` denote right and left yaw in degrees.

## Repository layout

```text
WorldCore/
├── inference.py
├── worldcore/
│   ├── diffusers/
│   ├── kernels/
│   ├── repencoder/
│   └── ucpe/
├── tools/
├── uvenv/
├── test/
│   ├── images/
│   ├── poses/
│   └── prompts/
├── weights/WorldCore_base/
├── assets/main.pdf
└── THIRD_PARTY_NOTICES.md
```

This repository contains inference and model code only. Training pipelines, dataset loaders, experiment launchers, and internal visualization utilities are not included.

## Notes

- Inference is currently optimized for batch size 1 at the video level.
- RepEncoder evaluates its four target views together in one target batch.
- Long autoregressive rollouts can amplify small numerical differences across GPU architectures or software versions. Use the documented environment and fixed seed for controlled comparisons.
- Generated content may inherit limitations or biases from the underlying video model and its pretraining data.

## License

WorldCore source code is released under the Apache License 2.0 unless a file or bundled component states otherwise. See `worldcore/THIRD_PARTY_NOTICES.md` before use or redistribution.

## Citation

```bibtex
@article{worldcore2026,
  title   = {WorldCore: Consistent Video World Model with Implicit Memory Representation},
  author  = {WorldCore Team},
  year    = {2026}
}
```
