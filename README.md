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

## 🎬 Video Demos

https://github.com/user-attachments/assets/e0428e46-6740-4129-b822-f4c65e7612a1

## 🚀 Getting Started

See the [Getting Started guide](docs/GETTING_STARTED.md) for installation,
model downloads, inference, and the interactive demo.

For camera controls and prompt examples, see the
[camera and prompt guide](test/README.md).

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
