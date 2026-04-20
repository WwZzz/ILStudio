# RynnVLA-002 — ILStudio Policy

RynnVLA-002 is an autoregressive action world model that unifies VLA (Vision-Language-Action) and world model on a Chameleon 7B backbone. It achieves 97.4% success rate on the LIBERO benchmark.

Paper: [arXiv:2511.17502](https://arxiv.org/abs/2511.17502)

## Installation

```bash
# 1. Enter policy directory
cd policy/rynnvla002

# 2. Clone the upstream repo (if not already present)
git clone https://github.com/alibaba-damo-academy/RynnVLA-002.git

# 3. Install xllmx from the repo
pip install -e RynnVLA-002/

# 4. Install remaining dependencies
pip install -r RynnVLA-002/requirements.txt
pip install flash-attn --no-build-isolation

# 5. Checkpoints (choose one)
#
# A) Automatic (default): keep ``auto_download_ckpts: true`` in configs/policy/rynnvla002.yaml.
#    On first ``train.py`` / load_model, ILStudio downloads from Hugging Face:
#    - Alibaba-DAMO-Academy/WorldVLA → chameleon/tokenizer + chameleon/starting_point
#    - Alpha-VLLM/Lumina-mGPT-7B-768 → tokenizer JSON files only
#    To force manual mode: export ILSTUDIO_RYNN_NO_AUTO_DOWNLOAD=1
#    (or HF_HUB_OFFLINE=1 / set auto_download_ckpts: false).
#
# B) Manual (same layout as upstream README Step 0):
# - Chameleon VQGAN tokenizer → RynnVLA-002/rynnvla-002/ckpts/chameleon/tokenizer/
#     (vqgan.yaml, vqgan.ckpt, text_tokenizer.json)
# - Lumina-mGPT tokenizer → RynnVLA-002/rynnvla-002/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/
# - Chameleon starting point → RynnVLA-002/rynnvla-002/ckpts/starting_point/
#   OR use fine-tuned weights from HuggingFace:
#     https://huggingface.co/Alibaba-DAMO-Academy/RynnVLA-002
```

## Configuration

See `configs/policy/rynnvla002.yaml` for defaults aligned with the checkpoint layout above. Relative paths are resolved from `policy/rynnvla002/` first (then the process working directory), so you can run `train.py` from the ILStudio repo root.

- `pretrained_path`: Chameleon **starting_point** directory, a fine-tuned checkpoint, or a Hugging Face repo id (e.g. `Alibaba-DAMO-Academy/RynnVLA-002` / Model Zoo subfolders in the upstream README).
- `tokenizer_path`: Lumina-mGPT tokenizer directory; leave empty to auto-detect under `.../ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/<revision>/`.
- `chameleon_tokenizer_dir`: Directory with `vqgan.yaml`, `vqgan.ckpt`, and `text_tokenizer.json` (default: `RynnVLA-002/rynnvla-002/ckpts/chameleon/tokenizer` relative to this policy folder).
- `auto_download_ckpts`: When true, missing tokenizer / local `starting_point` weights are fetched from Hugging Face (see installation §5A). Skipped when `pretrained_path` is a Hub repo id (e.g. `Alibaba-DAMO-Academy/RynnVLA-002`) so starting-point is not downloaded twice.

## Dataset Configuration

RynnVLA-002 expects actions and states normalized to `[-1, 1]`. Configure your task YAML with:

```yaml
meta:
  action_dim: 7
  state_dim: 8
  action_normalize: minmax
  state_normalize: minmax
```

## Architecture Notes

- **Backbone**: Chameleon 7B (multimodal autoregressive LM)
- **Image tokenization**: Meta VQGAN (discrete tokens, runs on GPU during forward pass)
- **Action representation**: 256-bin discretization for CE loss + continuous action head for L1 regression
- **Training loss**: Cross-entropy on discrete tokens + weighted L1 on continuous action predictions
- **Inference**: Single-step generation triggers the action head, which outputs `(time_horizon, action_dim)` continuous actions
