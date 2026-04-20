# DreamZero Policy

DreamZero is a world-model-based VLA (Vision-Language-Action) policy that jointly predicts future video frames and robot actions via flow-matching on a Wan2.1 video diffusion transformer.

## Architecture

- **Backbone**: Identity (no separate vision encoder)
- **Action Head**: WANPolicyHead — Wan2.1 I2V (Image-to-Video) model extended with action tokens
  - Video VAE for encoding/decoding video latents
  - T5 text encoder for language conditioning
  - CLIP image encoder for first-frame conditioning
  - DiT (Diffusion Transformer) for joint video + action denoising
- **Training Loss**: Flow-matching MSE on video noise + action noise (weighted)

## Installation

```bash
# From ILStudio root — fetch the submodule
git submodule update --init --recursive policy/dreamzero/dreamzero

# Set up a local environment
cd policy/dreamzero
uv venv
source .venv/bin/activate
uv sync
```

### Submodule

Core model code lives in the **dreamzero** git submodule at `policy/dreamzero/dreamzero/`
(repo: <https://github.com/dreamzero0/dreamzero>). The upstream package structure is:

```
dreamzero/groot/vla/model/dreamzero/   ← VLA model code
dreamzero/groot/vla/model/n1_5/        ← action encoder shared module
```

At runtime, `modeling.py` prepends the submodule root to `sys.path` so that
`groot.*` imports resolve correctly (same pattern as `policy/fastwam/FastWAM`).

### Dependencies

- `torch >= 2.1`
- `transformers >= 4.40`
- `einops`, `peft`, `safetensors`, `huggingface_hub`
- `hydra-core` (used by upstream VLA for component instantiation)
- `dm-tree`
- `decord` (video backend)

WAN2.1 model weights are auto-downloaded from `Wan-AI/Wan2.1-I2V-14B-480P` on first use.

## Usage

### Training

```bash
# ALOHA Sim (auto-downloads data)
python train.py \
  -p dreamzero \
  -t wm/aloha_sim_dreamzero \
  -o ckpt/dreamzero_aloha

# LIBERO
python train.py \
  -p dreamzero \
  -t wm/libero_wm \
  -o ckpt/dreamzero_libero \
  --task.datasets.0.args.root=/path/to/libero/data
```

## Data Pipeline

DreamZero uses the video datasets under `data_utils/datasets/wm/` which provide
efficient batch video loading for various data formats (HDF5, LeRobot, etc.).

### Data flow

```
Dataset (HDF5 / LeRobot / ...)
  → wm.XxxVideoDataset.__getitem__
    → {image: (T*K,C,H,W), reasoning: {video: {...}}, state, action, ...}
  → DreamZeroProcessor (per-sample)
    → reshape via reasoning["video"] → concat views → resize
    → {images: (T,H,W,C), text, state, action, action_mask, ...}
  → DreamZeroCollator (batching + tokenization)
    → {images, text, text_attention_mask, state, action, ...}
  → DreamZeroPolicy.forward()
    → VLA → WANPolicyHead → {loss, dynamics_loss, action_loss}
```

## References

- [DreamZero Paper](https://arxiv.org/abs/2505.04382)
- [Wan2.1 Video Model](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)
