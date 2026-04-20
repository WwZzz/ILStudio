# FastWAM Policy for ILStudio

FastWAM (Fast World-Action Model) integrates a WAN 2.2 video diffusion backbone with an action expert via Mixture-of-Transformers (MoT) for joint video-action generation.

## Upstream dependency (git submodule)

Like `policy/openpi/openpi`, the upstream library lives **inside this policy folder**:

- **Path:** `policy/fastwam/FastWAM`
- **Registered in:** `.gitmodules` → `https://github.com/yuantianyuan01/FastWAM.git`

After cloning ILStudio, fetch the submodule and install into your ILStudio venv:

```bash
git submodule update --init --recursive
pip install -e policy/fastwam/FastWAM
```

Alternatively, use `uv` inside the submodule (see upstream `FastWAM/README.md`):

```bash
git submodule update --init --recursive
cd policy/fastwam/FastWAM
uv sync
```

If you still have a legacy copy at the **ILStudio repository root** (`./FastWAM`) and cannot reach GitHub yet, you can point the expected submodule path at it (symlink; do not commit):

```bash
# from ILStudio repo root:
ln -sfn FastWAM policy/fastwam/FastWAM
```

After `git submodule update` succeeds, remove the symlink and let Git check out the real submodule there.

## WAN Weight Download

WAN backbone weights are downloaded **automatically** on first use. Control the download source via the `download_source` config key or the `DIFFSYNTH_DOWNLOAD_SOURCE` environment variable:

- `huggingface` (default) -- downloads from Hugging Face Hub
- `modelscope` -- downloads from ModelScope

**Storage location:** weights are stored under **`$HF_HOME/fastwam_wan_models/<model_id>/`** (not `./checkpoints` in the project directory). `HF_HOME` follows Hugging Face conventions (defaults to `~/.cache/huggingface`).

Override the root directory if needed:

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/your/cache
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
