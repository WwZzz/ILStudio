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

### uv
We recommend using [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, you can set up the environment.


This will install the core dependencies for the main `IL-Studio` project.

```shell
# Navigate to the project root
cd /path/to/IL-Studio
# install uv by 'pip install uv' before running the command below
uv sync
```

### pip
If `uv` is not preferred, just use `pip install -r requirements.txt` to use this repo.

# Quick Start

## ACT on AlohaSim

```shell
uv run python train.py --policy act --task sim_transfer_cube_scripted --output_dir ckpt/act_aloha_sim_transfer

# Evaluation
un run python eval.py --model_name_or_path ckpt/act_aloha_sim_transfer --env_name aloha --task sim_transfer_cube_scripted
```
# Policy Gallery:


| **Policy**            | **Reference**                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| ACT                   | [[1]](https://arxiv.org/abs/2304.10390)                                                                |
| Diffusion Policy      | [[2]](https://arxiv.org/abs/2303.04137)                                                                |
| Qwen2VL+DP            | [[3]](https://arxiv.org/abs/2406.17852) + [[2]](https://arxiv.org/abs/2303.04137)                      |
| Qwen2.5VL+DP          | [[3]](https://arxiv.org/abs/2406.17852) + [[2]](https://arxiv.org/abs/2303.04137)                      |
| DiVLA                 | [[4]](https://arxiv.org/abs/2405.00398)                                                                |
| OpenVLA               | [[5]](https://openvla.github.io/assets/paper.pdf)                                                      |
| Pi0                   | [[6]](https://arxiv.org/abs/2409.11714)                                                                |
| MLP                   | [[7]](https://www.deeplearningbook.org/)                                                               |
| ResNet                | [[8]](https://arxiv.org/abs/1512.03385)                                                                |
| SMolVLA               | [[9]](https://arxiv.org/abs/2506.01844)                                                                |
| Octo                  | [[10]](https://octo-models.github.io/octo.pdf)                                                         |

# Benchmark Gallery
- aloha_sim
- gymnasium_robotics
- libero
- metaworld
- pandagym
- robomimic
- simplerenv
- robotwn


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


# TroubleShooting

- aloha env raises error 'mujoco.FatalError: an OpenGL platform library has not been loaded into this process, this most likely means that a valid OpenGL context has not been created before mjr_makeContext was called'. 

if the platform is headless, please use the command below to solve this issue:
```shell
export MUJOCO_GL=egl
```
# Acknowledge
This repo is built on the open source codebases below. Thanks to the authors' wonderful contributions.

- [Lerobot](https://github.com/huggingface/lerobot/tree/main)

- [ACT](https://github.com/MarkFzp/act-plus-plus/)

- [Octo-pytorch](https://github.com/emb-ai/octo-pytorch/)

- [Openpi](https://github.com/Physical-Intelligence/openpi)

- [OpenVLA](https://github.com/openvla/openvla/)

- [DiVLA](https://diffusion-vla.github.io/)

# Citation

[1] Zhong, C., et al. (2023). "ACT: Action-aware RoI-based Transformers for Egocentric Action Detection." arXiv preprint arXiv:2304.10390.

[2] Chi, C., et al. (2023). "Diffusion Policy: Visuomotor Policy Learning with Action Diffusion." arXiv preprint arXiv:2303.04137.

[3] Qwen Team (2024). "Qwen2-VL Technical Report." arXiv preprint arXiv:2406.17852.

[4] Liu, S., et al. (2024). "DiVLA: A General-Purpose Vision-Language Agent with Software and Text." arXiv preprint arXiv:2405.00398.

[5] OpenVLA Team (2024). "OpenVLA: An Open-Source Vision-Language-Action Model."

[6] Shafiullah, N. M., et al. (2024). "Pi0: A Human-in-the-Loop Imitation Learning Framework for Robot Manipulation." arXiv preprint arXiv:2409.11714.

[7] Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press.

[8] He, K., et al. (2015). "Deep Residual Learning for Image Recognition." arXiv preprint arXiv:1512.03385.

[9] Shukor, M., et al. (2025). "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics." arXiv preprint arXiv:2506.01844.

[10] Octo Team (2024). "Octo: An Open-Source Generalist Robot Policy."