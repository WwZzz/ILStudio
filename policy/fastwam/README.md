# FastWAM Policy for ILStudio

FastWAM (Fast World-Action Model) integrates a WAN 2.2 video diffusion backbone with an action expert via Mixture-of-Transformers (MoT) for joint video-action generation.

## Installation

FastWAM lives in the `FastWAM/` directory at the project root and must be installed as a dependency:

```bash
cd FastWAM
uv venv
source .venv/bin/activate
uv sync
```

Alternatively, install directly into the ILStudio environment:

```bash
pip install -e FastWAM/
```

## WAN Weight Download

WAN backbone weights are downloaded **automatically** on first use. Control the download source via the `download_source` config key or the `DIFFSYNTH_DOWNLOAD_SOURCE` environment variable:

- `huggingface` (default) -- downloads from Hugging Face Hub
- `modelscope` -- downloads from ModelScope

Override the local cache directory:

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/checkpoints
```

## Attention Modes

Three attention variants are available, selected via `attention_mode` in the YAML config:

| Mode | Config | Description |
|------|--------|-------------|
| `original` | `fastwam.yaml` | Action attends only to first-frame video tokens |
| `joint` | `fastwam_joint.yaml` | Action attends to all video tokens |
| `idm` | `fastwam_idm.yaml` | Inverse dynamics with teacher-forcing conditioning |

## Video Generation at Inference

Set `generate_video: true` in the config to produce both video frames and actions during inference. When disabled (default), only actions are generated via a faster action-only denoising path.

## Usage

### Training

```bash
python train.py --policy fastwam --task <task_config>
```

### Inference

The policy exposes `select_action(batch_obs)` which returns `[B, T, action_dim]` action tensors. It is automatically wrapped by `MetaPolicy` during evaluation.
