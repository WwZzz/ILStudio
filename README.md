# TODO
- rewrite evaluation pipeline
- add octo
- add roboflamingo
- add CALVIN benchmark
- add Installation Description
- intergrate RoboVerse with unified APIs for evaluation
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
```
--dataset_dir
    ├─ episode_00.hdf5
    ├─ episode_01.hdf5
    ├─...
```

The architecture of each .hdf5 file should be like



The dataset's item should be a dict like 
```
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
```
TASK_CONFIGS = {
    ...,
    'task_name':{
        'dataset_dir': [
            dataset_dir, # the path of the dataset
        ],
        'episode_len': int,
        'camera_names': List(str), # e.g., ['primary']
    },
```


