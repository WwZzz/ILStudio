# EasyIL

This is a repo for quick start with imitation learning on several popular robotic tasks. We carefully design the whole pipeline and each module to provide supports for different policies, simulation environments. Users can easily intergrate their own policy and test it on this platform. 


# TODO
- rewrite evaluation pipeline
- add octo
- add roboflamingo
- add CALVIN benchmark
- add Installation Description
- add deploy module 
- ...

# Usage
```shell
# model_name must be in vla and task_name must be in configuration.constants.TASK_CONFIGS
python train.py --model_name act --task_name example_tasks --output_dir output_dir_path 
```

# Model
important APIs from each `vla.algo_name.__init__`
- `def load_model(args: transformers.HfArgumentParser) -> dict(model=transformers.PreTrainedModel, ...)` # loading models
- (OPTIONAL) `def wrap_data(dataset: torch.utils.data.Dataset, args: transformers.HfArgumentParser, model_components: dict) -> torch.utils.data.Dataset` # sample-level data processing
- (OPTIONAL) `def get_data_collator(args: transformers.HfArgumentParser, model_components:dict) -> function` # batch-level data processing
- (OPTIONAL) `class Trainer(transformers.trainer.Trainer)`

The model returned by `load_model` should implement several APIs:
- `def evaluate(self, obs) -> dict(output_text=str, action=numpy.ndarray((batch_size, chunk_size, action_dim))`
- `def process_env_observation(self, obs:dict) -> dict` # convert the obvervation into the same format for each algorithm

## Currently available algorithms:
- ACT
- Diffusion Policy
- Qwen2VL+DP
- OpenVLA
- Pi0

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
# for gripper state, 1=close and -1=open, the same with openvla 
# for gripper action, 0=close and 1=open, the same with openvla 
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
    'qpos': torch.Tensor((chunk_size, state_dim), dtype=torch.float32),
    'raw_lang': str,
    'is_pad': torch.Tensor((chunk_size, action_dim), dtype=torch.bool),
    'reasoning': str,
}
```

To add customized datasets, please modify
```python
TASK_CONFIGS = {
    ...,
    'task_name':{
        'dataset_dir': [
            dataset_dir, # the path of the dataset
        ],
        'episode_len': int,
        'camera_names': List(str), # e.g., ['primary']
    },
}
```


