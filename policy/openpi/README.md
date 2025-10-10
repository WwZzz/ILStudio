# Installation

```shell
cd /path/to/IL-Studio
cd policy/openpi/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ../../..
source policy/openpi/openpi/.venv/bin/activate
uv pip install peft tensorflow==2.18.0 tf-keras tensorflow_datasets tianshou==0.2.0 robosuite==1.4.0
```


# Concert JAX weights to PyTorch
### Download weights
```shell
cd policy/openpi/
# Download to default cache (~/.cache/openpi)
python download.py --model pi05_droid

# Download to customized cache
python download.py --model pi0_base --output custom/path
```

### Convert weights
```shell
cd policy/openpi/openpi
# /path/to/jax/checkpoint, e.g., custom/path/pi0_base, /path/to/converted/pytorch/checkpoint=custom/path/pi0_base_pytorch
python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --output_path /path/to/converted/pytorch/checkpoint
```
### Use pretrained weights
Set `model_args: pytorch_weight_path: ...` in the .yaml config of `configs/policy/pi0_xx.yaml` like
```
# pi0_aloha.yaml
name: openpi
module_path: policy.openpi
pretrained_config:
  model_name_or_path: "facebook/opt-125m"
  is_pretrained: false
model_args:
  # Model architecture parameters
  pytorch_training_precision: bfloat16
  action_dim: 14
  max_action_dim: 32
  chunk_size: 50
  max_token_len: 48
  paligemma_variant: gemma_2b
  action_expert_variant: gemma_300m
  pi05: False
  # Task parameters (will be overridden by args if provided)
  state_dim: 14
  discrete_state_input: False
  pytorch_weight_path: /path/to/converted/pytorch/checkpoint # This line should be replaced
action_normalize: "minmax"
state_normalize: "minmax"
trainer_class: Trainer

```