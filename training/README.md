# Fine-tune WorldCrafter-Base

This directory provides the complete workflow for **fine-tuning WorldCrafter-Base**:
download videos, prepare video latents and caption embeddings, train, and export a
model for inference. Use the **[WorldCrafter dataset](https://huggingface.co/datasets/Drexubery/WorldCrafter)**
or your own captioned videos through the same preparation script.

## WorldCrafter-Context baseline

[**WorldCrafter-Context**](https://huggingface.co/Drexubery/WorldCrafter-Context)
is an autoregressive baseline with historical-frame context memory and UCPE camera
control, without a memory encoder. It is intended for developing custom memory
mechanisms, such as point-cloud conditioning.

To construct it, combine **Base's `transformer/`**, **Context's
`camera_adapter.bin`**, and **Fast's `text_encoder/`, `tokenizer/`, `vae/` and
`scheduler/`**. Supply historical video latents as context; do not load Base/Fast's
additional LoRA, camera adapters or RepEncoder. See the
[model card](https://huggingface.co/Drexubery/WorldCrafter-Context#download) for
shared-component downloads. Context requires a custom context-memory integration;
the fine-tuning commands below target WorldCrafter-Base.

## Set up

Run all commands from the repository root:

```bash
uv sync --project uvenv/training --frozen
source uvenv/training/.venv/bin/activate
```

For a network-mounted checkout, set `UV_PROJECT_ENVIRONMENT` and `UV_CACHE_DIR`
to local SSD directories before installation, then activate that environment.

Download **WorldCrafter-Base** and **WorldCrafter-Fast** following the
[model instructions](../README.md). Keep them alongside each other:

```text
weights/
├── WorldCrafter-Base/
└── WorldCrafter-Fast/
```

Base uses the shared video encoder, text encoder, tokenizer, scheduler and memory
encoder stored with Fast. Training starts from Base's transformer and adapters.
The training environment uses native PyTorch attention and supports distributed
training with PyTorch DDP.

## Download the WorldCrafter dataset

The **[WorldCrafter dataset](https://huggingface.co/datasets/Drexubery/WorldCrafter)**
provides high-quality long videos with captions and camera annotations. Complete the
academic research agreement on the dataset page, then log in locally:

```bash
hf auth login
python training/download_dataset.py \
  --root data/WorldCrafter --download --verify --extract
```

The downloader resolves a fixed dataset revision, verifies archive and extracted
file checksums, and safely resumes completed downloads. Use `--revision COMMIT`
to select a specific version. Allow room for both downloaded archives and extracted
videos, plus the feature files generated below.

## Encode videos and captions

```bash
python training/prepare_data.py \
  --dataset-root data/WorldCrafter \
  --model-path weights/WorldCrafter-Base
```

This command reuses the supplied cameras and encodes the videos with the model's
video autoencoder and the captions with its text encoder. Captions are already
provided; the script does not generate them. It writes:

```text
data/WorldCrafter/
├── metadata/samples.jsonl       # Downloaded captions, frame ranges and camera paths
├── videos/                     # Downloaded videos
├── cameras/                    # Downloaded camera annotations
├── features/                   # Generated .pt files
└── valid_chunk_indices.json    # Generated training index for completed features
```

Each video keeps its full chunk history for memory conditioning. The downloaded
annotations specify the exact crop, frame interval and stride. The output is
640x384; each 33-frame chunk becomes nine latent frames. Existing outputs are reused
only when their input and preprocessing signatures match. Use a new `--output`
directory when changing inputs or models.

To try a small subset first, add `--max-samples 8`. Its generated index includes
only prepared samples, so the next training command can use it directly.
`--check-only` checks manifest/camera alignment without loading models.

For parallel preparation, run one worker per GPU with `--num-shards N` and distinct
`--shard-index I` values, then merge the completed indices:

```bash
python training/prepare_data.py --dataset-root data/WorldCrafter \
  --num-shards 8 --merge-index
```

Use the same manifest, output directory, seed and model in every worker. Each
worker should see its own GPU through `CUDA_VISIBLE_DEVICES`.

## Fine-tune the model

The default [train_config.yaml](train_config.yaml) points to the prepared
WorldCrafter dataset. First check that its features can be loaded and sampled:

```bash
python train.py --check-data
```

An eight-GPU example with effective batch size 32:

```bash
NPROC_PER_NODE=8 bash scripts/training/finetune_base.sh \
  --gradient-accumulation-steps 4 --max-steps 500 \
  --output-dir outputs/worldcrafter-finetuned
```

The [dataset](dataset.py) loads video latents, caption embeddings and cameras,
selects target chunks from the prepared index, and samples preceding views for
memory. The training entry point builds the DataLoader with batch size one per GPU,
a distributed sampler, and deterministic epoch shuffling. Effective batch size is
`GPU count × gradient accumulation steps`; the learning rate is not automatically
scaled. The recipe has about 1.15 billion trainable parameters, so use GPUs with
sufficient memory for the model, optimizer and activations.

| Component | Fine-tuning behavior |
| --- | --- |
| Low-rank transformer adapters | Trainable; initialized from Base |
| Camera-conditioning adapters | Trainable; initialized from Base |
| Transformer backbone | Frozen |
| Memory encoder | Frozen |
| Video and text encoders | Used during data preparation only |

The default recipe uses AdamW with learning rate `3e-5` and mixed bfloat16 compute.
It samples text-only, recent-history and full-memory conditioning with probabilities
40%, 24% and 36%. See [train_config.yaml](train_config.yaml) and
[reference_noise.json](utils/reference_noise.json) for the fixed training settings.

Resume with the same data, GPU count and recipe:

```bash
NPROC_PER_NODE=8 bash scripts/training/finetune_base.sh \
  --gradient-accumulation-steps 4 --max-steps 1000 \
  --output-dir outputs/worldcrafter-finetuned --resume latest
```

Checkpoints save trained adapters, optimizer state and each rank's random state.
Only checkpoints containing a `COMPLETE` marker can be resumed or exported.

## Prepare your own dataset

Use the **same `prepare_data.py` command** with your own videos and captions.
Create `data/my_dataset/metadata/samples.jsonl`, with one JSON object per line:

```json
{"sample_id":"scene_001","video_path":"videos/scene_001.mp4","caption":"The camera moves through a sunlit garden.","cut":[0,165],"crop":[0,1920,0,1080],"fps":24.0}
```

Paths are relative to `--dataset-root`. `cut` is `[start_frame, end_frame)` in the
decoded video; `crop` is `[left, right, top, bottom]` in pixels, before center cropping
and resizing to 640x384. Use the actual video dimensions and frame rate. Each sample
needs a unique ID, a supplied caption and at least 121 sampled frames. `stride`
defaults to 1; 2 is also supported. The script chooses a supported frame-length
bucket from the start of the cut and discards any trailing incomplete chunk.

### Videos without camera annotations

Add **`--use_da3`** to estimate cameras with **Depth Anything 3**, a multi-view model
that predicts depth and camera geometry. Install its separate environment and
download the model:

```bash
uv sync --project uvenv/data --frozen
hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --local-dir weights/DA3NESTED-GIANT-LARGE-1.1

python training/prepare_data.py \
  --dataset-root data/my_dataset \
  --model-path weights/WorldCrafter-Base \
  --use_da3
```

Keep the training environment active when running `prepare_data.py`. Camera
estimation uses `uvenv/data/.venv/bin/python` in a subprocess; if you installed it
elsewhere, pass `--camera-python /path/to/environment/bin/python`. If you set
`UV_PROJECT_ENVIRONMENT` for training, point it to a **different directory** when
installing the camera environment. Use
`--camera-model` for a different local model location and `--tmp-dir` for local
scratch storage. The video/text encoding stage starts after the camera process
exits, allowing both stages to share one GPU.

The estimator jointly processes nine anchors per 33-frame chunk, plus the video's
reference frame. It uses processing resolution 504, upper-bound resizing and the
middle reference view. World-to-camera estimates are converted to camera-to-world
poses; intrinsics are normalized by the estimator's actual processed image size.
The script saves reusable camera JSON files under `cameras/`, then runs the same
latent and caption encoding used for the released dataset. Motion and trajectory
checks select training targets; all feature chunks remain available as history.

### Videos with camera annotations

Include `camera_path` in each manifest record and omit `--use_da3`. Use the same
manifest and camera JSON format as the WorldCrafter dataset: preprocessing fields
must describe the exact frames used for estimation. The script checks sample IDs,
captions, frame anchors, camera transforms and normalized intrinsics before encoding.
Both paths produce the same training feature format:

| Feature | Shape / convention |
| --- | --- |
| Video latents | `[chunks, 16, 9, 48, 80]` |
| Caption embedding | `[512, 4096]` |
| Video-relative and chunk-relative camera-to-world poses | `[chunks, 9, 4, 4]`; translations in meters |
| Normalized intrinsics | `[chunks, 9, 4]`, ordered `fx, fy, cx, cy` |
| Camera validity and frame indices | `[chunks, 9]`; anchors at frames `0, 4, …, 32` of each chunk |

Fine-tune with the resulting features and index:

```bash
python train.py --feature-root data/my_dataset/features \
  --valid-chunks data/my_dataset/valid_chunk_indices.json --check-data

NPROC_PER_NODE=8 bash scripts/training/finetune_base.sh \
  --feature-root data/my_dataset/features \
  --valid-chunks data/my_dataset/valid_chunk_indices.json \
  --gradient-accumulation-steps 4 --max-steps 500 \
  --output-dir outputs/my-finetuned-model
```

## Export and run inference

```bash
python training/export_model.py \
  --base-model weights/WorldCrafter-Base \
  --checkpoint outputs/worldcrafter-finetuned/checkpoint-500 \
  --output weights/WorldCrafter-Base-finetuned

python inference.py --model-type base \
  --model-path weights/WorldCrafter-Base-finetuned \
  --output-path outputs/finetuned.mp4
```

Export copies the frozen transformer and saves the trained adapters in the standard
inference format. Shared components are referenced by relative path; add
`--copy-shared` to create a self-contained model directory for distribution.

## Code layout

```text
training/
├── README.md
├── download_dataset.py
├── prepare_data.py
├── train.py
├── train_config.yaml
├── dataset.py
├── model.py
├── export_model.py
├── pack_features.py        # Optional lossless feature archives
└── utils/                 # Shared data, camera, sampling, loss and checkpoint helpers
```
