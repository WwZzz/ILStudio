<div align="center">
  <img src='https://raw.githubusercontent.com/WwZzz/myfigs/refs/heads/master/fig_ilstd_logo.png'  width="200"/>
<h1> IL-Studio: A Plug-and-Play Imitation-Learning Playground for Robotics
</h1>



</div>



| ACT - Insertion                                          | Diffusion UNet - Transfer Cube                               | ACT - Square                                             |
|----------------------------------------------------------|--------------------------------------------------------------|----------------------------------------------------------|
| <img src="https://raw.githubusercontent.com/WwZzz/myfigs/refs/heads/master/act_aloha_insertion.gif" height="200">  | <img src="https://raw.githubusercontent.com/WwZzz/myfigs/refs/heads/master/diffusion_aloha_transfer.gif" height="200"> | <img src="https://raw.githubusercontent.com/WwZzz/myfigs/refs/heads/master/act_robomimic_square.gif" height="200"> |

| Koch - Pick & Place -Inference                                     |     SO101 - Fold Tower -Inference               | BC_MLP - AdroitHandDoor                                                         |
|----------------------------------------------------------|--------------------------------------------------------------|----------------------------------------------------------|
| <img src="https://raw.githubusercontent.com/WwZzz/myfigs/refs/heads/master/fig_koch.gif" height="200">             | <img src="https://raw.githubusercontent.com/WwZzz/myfigs/refs/heads/master/so101_fold.gif" height="200">          |<img src="https://raw.githubusercontent.com/WwZzz/myfigs/refs/heads/master/fig_door.gif" height="200">|

IL-Studio is an open-source repository that lets researchers and engineers jump-start imitation-learning experiments on popular robot manipulation benchmarks with minimal friction. The entire training, evaluation, and deployment pipeline has been carefully modularized so that you can swap-in your own policy, environment, or dataset without touching the rest of the stack.

# Installation
```shell
pip install -r requirements.txt
```

# TODO
- rewrite evaluation pipeline
- add octo
- add roboflamingo
- add CALVIN benchmark
- add Installation Description
- add deploy module 
- ...

# Usage

## Policy Configuration System

```shell
# Use policy_config to load from YAML file path
python train.py --policy_config configs/policy/act.yaml --task_name example_tasks --output_dir output_dir_path 

# Available policies: configs/policy/act.yaml, configs/policy/qwen2dp.yaml, etc.
python train.py --policy_config configs/policy/qwen2dp.yaml --task_name example_tasks --output_dir output_dir_path

# Evaluation
python eval.py --policy_config configs/policy/act.yaml --env_name aloha --task sim_transfer_cube_scripted

# Real-world evaluation
python eval_real.py --policy_config configs/policy/act.yaml --robot_config configs/robots/dummy.yaml --task sim_transfer_cube_scripted 
```

# Overview
We show the architecture as below:
![framework](https://raw.githubusercontent.com/WwZzz/myfigs/refs/heads/master/fig_il.png)

# Model
important APIs from each `policy.algo_name.__init__`
- `def load_model(args: transformers.HfArgumentParser) -> dict(model=transformers.PreTrainedModel, ...)` # loading models
- (OPTIONAL) `def get_data_processor(dataset: torch.utils.data.Dataset, args: transformers.HfArgumentParser, model_components: dict) -> function` # sample-level data processing
- (OPTIONAL) `def get_data_collator(args: transformers.HfArgumentParser, model_components:dict) -> function` # batch-level data processing
- (OPTIONAL) `class Trainer(transformers.trainer.Trainer)`

The model returned by `load_model` should implement:
- `def select_action(self, obs) -> action`

## Currently available algorithms:
- ACT
- Diffusion Policy
- Qwen2VL+DP
- Qwen2.5VL+DP
- DiVLA
- OpenVLA

# Dataset
Each dataset refers to a dictionary containing 
```shell
--dataset_dir
    ├─ episode_00.hdf5
    ├─ episode_01.hdf5
    ├─...
```

The architecture of each .hdf5 file should be like
```shell
# T is the length of the episode
# we denote the gripper state by the openning degree of the gripper (e.g., x m),
# and we denote the gripper action by 0=close and 1=open, which is the same as openvla 
# the minimal setting of the data configuration is labeled by * (e.g., 'ee' can be replaced by 'joint' and 'primary' can be replaced by 'wrist')

Dataset: dataset_dir (Shape: (1,), Dtype: object)           # the dataset directionary of the episode, str
Dataset: episode_id (Shape: (1,), Dtype: object)            # the id of the episode, str
Dataset: freq (Shape: (1,), Dtype: float32)                 # control frequency
Dataset: language_instruction (Shape: (1,), Dtype: object)  #*task instruction, str 
Dataset: robot (Shape: (1,), Dtype: object)                 # robot name, str
Dataset: action_ee (Shape: (T, 7), Dtype: float32)          #*action of end-effector and gripper, e.g., [ee_xyz(3), ee_yrp(3), gripper_action(1)] 
Dataset: action_joint (Shape: (T, 8), Dtype: float32)       # action of joints and gripper, e.g., [joints(7), gripper_state(1)]
Group: observations
  Group: image
    Dataset: primary (Shape: (T, H, W, 3), Dtype: uint8)    #*primary camera
    Dataset: wrist (Shape: (T, H, W, 3), Dtype: uint8)      # wrist camera
    Dataset: ...
  Group: depth                                              # the depth
    Dataset: primary (Shape: (T, H, W), Dtype: float32)     # primary camera depth
    Dataset: wrist (Shape: (T, H, W), Dtype: float32)       # wrist camera 
    Dataset: ...
  Dataset: state_ee (Shape: (T, 7), Dtype: float32)         #*state of end-effector and gripper, e.g., [ee_xyz(3), ee_yrp(3), gripper_action(1)] 
  Dataset: state_joint (Shape: (T, 8), Dtype: float32)      # state of joints and gripper, e.g., [joints(7), gripper_state(1)]
Group: reasoning                                            # reasoning information 
```

The dataset's item should be a dict like 
```python
{
    'image': torch.Tensor((K, C, H, W), dtype=torch.uint8), # K is the number of images (i.e., primary, wrist, ) and C is the number of channels 
    'action': torch.Tensor((chunk_size, action_dim), dtype=torch.float32),
    'state': torch.Tensor((chunk_size, state_dim), dtype=torch.float32),
    'raw_lang': str,
    'is_pad': torch.Tensor((chunk_size, action_dim), dtype=torch.bool),
    'reasoning': str,
}
```

To add customized datasets, please modify

## Task Configuration (YAML)

Each task should have a YAML file in `configs/task/` named `<task_name>.yaml`. Example:

```yaml
dataset_dir:
    - /path/to/sim_transfer_cube_scripted
episode_len: 400
camera_names:
    - primary
dataset_class: AlohaSimDataset
ctrl_type: abs
ctrl_space: joint
```

## Data Preparation

| Name   | Download | Remark|
|----------|-----------|-------------|
| Aloha   | [link](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O)  |   |
| VLA-OS | [link](https://huggingface.co/datasets/Linslab/VLA-OS-Dataset)  |   |
| Open-X | [link](https://github.com/google-deepmind/open_x_embodiment)  |    |


# TroubleShooting

- aloha env raises error 'mujoco.FatalError: an OpenGL platform library has not been loaded into this process, this most likely means that a valid OpenGL context has not been created before mjr_makeContext was called'. 

if the platform is headless, please use the command below to solve this issue:
```shell
export MUJOCO_GL=egl
```


# Quick Start Step by Step
## Simulation Data Preparation
There are three ways to download the generated datasets.
- 1) Download from the official repo [link](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link)
- 2) Download from the third-party huggingface `huggingface-cli download --repo-type dataset cadene/aloha_sim_transfer_cube_human_raw --local-dir ./data/sim_transfer_cube_scripted`
- 3) Generate the data by following the guidence in [link](https://github.com/tonyzhaozh/act)

## Add the dataset to configuration
```python

## Add the dataset to configuration

Create a YAML file in `configs/task/` (e.g., `sim_transfer_cube_scripted.yaml`) with the following content:

```yaml
dataset_dir:
    - /path/to/sim_transfer_cube_scripted
episode_len: 400
camera_names:
    - primary
dataset_class: AlohaSimDataset
ctrl_type: abs
ctrl_space: joint
```

```
## Train
```shell
bash scripts/example_train.sh
```

## Eval
```shell
bash scripts/example_eval.sh
```
