# WorldCrafter: Consistent Video World Model with Implicit 3D-aware Memory

<p align="center">
  <a href="https://arxiv.org/abs/2609.24984"><img src="https://img.shields.io/badge/arXiv-2609.24984-b31b1b.svg" alt="arXiv Paper"></a> &nbsp;
  <a href="https://drexubery.github.io/WorldCrafter/"><img src="https://img.shields.io/badge/Project-Page-Green" alt="Project Page"></a> &nbsp;
  <a href="https://www.youtube.com/watch?v=sg09ftQOl0E&amp;t=5s"><img src="https://img.shields.io/badge/Youtube-Video-b31b1b.svg" alt="YouTube Video"></a> &nbsp;
  <a href="https://huggingface.co/TencentARC/WorldCrafter-Fast"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-blue" alt="Hugging Face Weights"></a>
</p>

🤗 If you find WorldCrafter useful, please consider giving this repo a ⭐. Your support helps us share and improve the project. Thank you!

## 🔆 Introduction

WorldCrafter enables consistent, camera-controlled scene exploration from an image or text prompt. Its camera-queryable implicit 3D-aware memory preserves scene information across viewpoints and over long horizons.

We provide **WorldCrafter-Base** and **WorldCrafter-Fast**, a distilled model for faster inference. 

https://github.com/user-attachments/assets/e0428e46-6740-4129-b822-f4c65e7612a1

## ⚙️ Setup

### 1. Clone WorldCrafter

```bash
git clone https://github.com/TencentARC/WorldCrafter.git
cd WorldCrafter
```

### 2. Environment

Set up the environment with **uv** or **conda + pip**. Both methods use Python 3.11 on Linux and require an NVIDIA GPU with a compatible driver.

**A: uv (recommended)**

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y ffmpeg

uv sync --project uvenv --frozen --extra demo
source uvenv/.venv/bin/activate
```

For other Linux distributions, install [FFmpeg](https://ffmpeg.org/download.html)
using your system package manager.

This installs the locked PyTorch 2.10 / CUDA 12.8 environment and its
acceleration dependencies.

**B: conda + pip**

Create an environment and install PyTorch for your machine. For CUDA 12.8:

```bash
conda create -n worldcrafter -c conda-forge python=3.11 pip ffmpeg -y
conda activate worldcrafter
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[demo,xformers]" flash-attn-3==3.0.0 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

Choose the appropriate CUDA build from the
[PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v2100).

### 3. Model weights

| Models | Download Link | Notes |
| --- | --- | --- |
| WorldCrafter-Base | 🤗 [Hugging Face](https://huggingface.co/TencentARC/WorldCrafter-Base) | Base model |
| WorldCrafter-Fast | 🤗 [Hugging Face](https://huggingface.co/TencentARC/WorldCrafter-Fast) | Distilled high- and low-noise models for faster inference |

Download weights with the Hugging Face CLI:

```bash
hf download TencentARC/WorldCrafter-Fast --local-dir weights/WorldCrafter-Fast

# Optional: also download Base to run the base model
hf download TencentARC/WorldCrafter-Base --local-dir weights/WorldCrafter-Base
```

Base model uses shared components from `WorldCrafter-Fast`, so keep both folders when using base model.

## 💫 Inference

See the [inference guide](test/README.md) for camera controls, prompt writing, and examples.

### 1. Image-to-video

Run with either model:

```bash
# Base
python inference.py --model-type base --mode i2v \
  --image-path test/I2V/00_cat_vac/image.png \
  --prompt-path test/I2V/00_cat_vac/prompt.txt \
  --camera-path test/I2V/00_cat_vac/camera.npy \
  --output-path output/base.mp4

# Fast
python inference.py --model-type fast --mode i2v \
  --image-path test/I2V/00_cat_vac/image.png \
  --prompt-path test/I2V/00_cat_vac/prompt.txt \
  --camera-path test/I2V/00_cat_vac/camera.npy \
  --output-path output/fast.mp4
```


### 2. Text-to-video

```bash
# Base
python inference.py --model-type base --mode t2v \
  --prompt-path test/T2V/00_red_balloon/prompt.txt \
  --camera-path test/T2V/00_red_balloon/camera.npy \
  --output-path output/t2v.mp4

# Fast
python inference.py --model-type fast --mode t2v \
  --prompt-path test/T2V/00_red_balloon/prompt.txt \
  --camera-path test/T2V/00_red_balloon/camera.npy \
  --output-path output/fast_t2v.mp4
```

Compilation is **off by default**. Add `--enable-compile` to enable it; the first run takes longer to start.

### 3. Custom inputs

```bash
python inference.py \
  --image-path path/to/image.png \
  --camera-path path/to/camera.npy \
  --prompt "Your scene description" \
  --output-path output/custom.mp4
```



## 🎮 Interactive Demo

> The interactive demo is currently being debugged.

Explore a scene with keyboard camera controls from your activated environment:

```bash
python -m demo --model-path weights/WorldCrafter-Fast
```

Open `http://localhost:8080`. The single-GPU demo uses Fast image-to-video with
compilation enabled. See [demo/README.md](demo/README.md) for controls and deployment.

## 📝 Citation

If you find WorldCrafter useful in your research, please cite:

```bibtex
@misc{yu2026worldcrafter,
  title={WorldCrafter: Consistent Video World Model with Implicit {3D}-aware Memory},
  author={Wangbo Yu and Kunhao Liu and Wenbo Hu and Shenghai Yuan and Chaoran Feng and Haiyang Zhou and Yukun Huang and Yiran Wang and Wang Zhao and Yingmin Luo and Ying Shan},
  year={2026},
  eprint={2609.24984},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.24984}
}
```

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
