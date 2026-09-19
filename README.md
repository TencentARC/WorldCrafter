## *WorldCrafter: Consistent Video World Model with Implicit Memory Representation*

🤗 If you find WorldCrafter useful, please consider giving this repo a ⭐. Your support helps us share and improve the project. Thank you!

## 🔆 Introduction

WorldCrafter generates camera-controlled videos with a compact, 3D-aware memory. Given an image or text prompt and a camera trajectory, it generates video autoregressively while retaining scene information from earlier views.

We provide **WorldCrafter-Base** and **WorldCrafter-Fast**, a six-step distilled variant.

## ⚙️ Setup

### 1. Environment

Validated on NVIDIA H200 with Linux, Python 3.11, and CUDA 12.8.

From the repository root, run:

```bash
uv sync --project uvenv --frozen
source uvenv/.venv/bin/activate
```

Both models use the same environment. See [uvenv/README.md](uvenv/README.md) for dependency details.

### 2. Model weights

Public weight downloads will be added when available. Place the model folders as follows:

```text
weights/
├── WorldCrafter_base/
└── WorldCrafter_fast/
```

Fast requires both folders: it shares the text encoder, tokenizer, VAE, scheduler configuration, and RepEncoder with Base.

## 💫 Inference

### 1. Image-to-video

Run the bundled image, prompt, and camera trajectory with either model:

```bash
# Base
python inference.py --output-path outputs/base.mp4

# Fast
python inference.py --model-type fast --output-path outputs/fast.mp4
```

Fast uses six denoising steps per chunk and currently supports image-to-video at 384 × 640 in eager mode, without resume.

### 2. Text-to-video

```bash
python inference.py --mode t2v --output-path outputs/t2v.mp4
```

### 3. Custom inputs

```bash
python inference.py \
  --image-path path/to/image.png \
  --camera-path path/to/camera.npy \
  --prompt "A sunlit room with neatly arranged furniture." \
  --output-path outputs/custom.mp4
```

Camera trajectories use global camera-to-world matrices in `[T, 3, 4]` or `[T, 4, 4]` NumPy arrays, with metric translations and 33 frames per chunk. Example inputs are included in [`test/`](test/). Use `--num-chunks` to limit the rollout and `--chunk-output-dir` to save individual chunks.

Run `python inference.py --help` for all options.

## 📄 License

See [LICENSE.txt](LICENSE.txt) for the terms of use and third-party attributions.

## 🤗 Related Works

[Helios](https://github.com/PKU-YuanGroup/Helios),
[LagerNVS](https://github.com/facebookresearch/lagernvs),
[DreamX-World](https://github.com/AMAP-ML/DreamX-World),
[EVOKE](https://github.com/AlayaLab/Evoke),
[HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay),
[Lyra 2.0](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2),
[Echo-WM](https://github.com/jd-opensource/JoyAI-Echo/tree/main/echo_wm),
[LingBot-World 2](https://github.com/robbyant/lingbot-world-v2),
[Matrix-Game 3.5](https://github.com/Riemann-Dynamics/Matrix-Game-3.5),
[SANA-WM](https://github.com/NVlabs/Sana/blob/main/docs/sana_wm.md).
