# Mask配置格式说明

## 配置位置

**重要**：`action_norm_mask` 和 `state_norm_mask` 应该与 `args` 同级，**不在** `args` 内部。

### ✅ 正确格式

```yaml
datasets:
  - name: "my_robot_dataset"
    class: "AlohaSimDataset"
    args:
      dataset_path_list: ['data/robot']
      camera_names: ['primary']
      chunk_size: 100
      ctrl_space: 'joint'
      ctrl_type: 'abs'
    # mask配置与args同级
    action_norm_mask: [-1]
    state_norm_mask: [-1]
```

### ❌ 错误格式

```yaml
datasets:
  - name: "my_robot_dataset"
    class: "AlohaSimDataset"
    args:
      dataset_path_list: ['data/robot']
      camera_names: ['primary']
      # ❌ 不要把mask放在args内部
      action_norm_mask: [-1]
      state_norm_mask: [-1]
```

## 完整示例

```yaml
# configs/task/my_task.yaml
datasets:
  # 示例1: 单臂机器人，gripper不归一化
  - name: "single_arm"
    class: "AlohaSimDataset"
    args:
      dataset_path_list: ['data/single_arm']
      camera_names: ['primary']
      ctrl_space: 'joint'
      ctrl_type: 'abs'
    action_norm_mask: [-1]  # 最后一维不归一化
    state_norm_mask: [-1]

  # 示例2: 双臂机器人，两个gripper都不归一化
  - name: "dual_arm"
    class: "EpisodicDataset"
    args:
      dataset_path_list: ['data/dual_arm']
      camera_names: ['front', 'side']
    action_norm_mask: [6, 13]  # 第7和第14维不归一化
    state_norm_mask: [6, 13]

  # 示例3: 使用Boolean数组
  - name: "robot_with_bool_mask"
    class: "EpisodicDataset"
    args:
      dataset_path_list: ['data/robot']
      camera_names: ['top']
    # 7维action: 前6维归一化，最后1维不归一化
    action_norm_mask: [true, true, true, true, true, true, false]

  # 示例4: 不指定mask（所有维度都归一化）
  - name: "standard_dataset"
    class: "EpisodicDataset"
    args:
      dataset_path_list: ['data/standard']
      camera_names: ['primary']
    # 不指定mask，使用默认行为

action_dim: 14
state_dim: 14
action_normalize: "zscore"
state_normalize: "zscore"
```

## 代码读取方式

在 `data_utils/utils.py` 中，mask是从 `dataset_config` 顶层直接读取的：

```python
# 从dataset_config顶层读取（与args同级）
action_norm_mask = dataset_config.get('action_norm_mask', None)
state_norm_mask = dataset_config.get('state_norm_mask', None)

# 而不是从args内部读取
# ❌ dataset_config['args'].get('action_norm_mask')  # 错误！
```

## 为什么要这样设计？

1. **清晰的职责分离**
   - `args`: 传递给dataset类构造函数的参数
   - `action_norm_mask/state_norm_mask`: normalizer的配置参数

2. **避免混淆**
   - mask不是dataset的参数，而是normalizer的参数
   - 将它们分开可以清楚地表明各自的作用域

3. **更好的可读性**
   - 配置结构一目了然
   - 便于理解哪些参数传递给dataset，哪些用于normalizer

## 验证配置格式

可以运行以下命令验证配置格式是否正确：

```bash
python -c "
import yaml
with open('configs/task/your_task.yaml', 'r') as f:
    config = yaml.safe_load(f)
dataset = config['datasets'][0]
print('args中的keys:', list(dataset['args'].keys()))
print('mask在args同级:', 'action_norm_mask' in dataset)
assert 'action_norm_mask' not in dataset['args'], 'mask不应该在args内部'
print('✓ 配置格式正确')
"
```

## 迁移指南

如果你之前将mask配置在args内部，需要将它们移出来：

### 之前的配置

```yaml
datasets:
  - name: "my_dataset"
    args:
      dataset_path_list: ['data/path']
      action_norm_mask: [-1]  # 在args内部
```

### 迁移后的配置

```yaml
datasets:
  - name: "my_dataset"
    args:
      dataset_path_list: ['data/path']
    action_norm_mask: [-1]  # 移到args外面，与args同级
```

## 常见问题

**Q: 我把mask放在args里面，会报错吗？**

A: 不会报错，但mask不会生效。因为代码从dataset_config顶层读取mask，而不是从args内部。

**Q: 每个dataset都必须配置mask吗？**

A: 不是。不配置mask时，所有维度都会被归一化（默认行为）。

**Q: action_norm_mask和state_norm_mask必须同时配置吗？**

A: 不是。可以只配置action_norm_mask，或只配置state_norm_mask，或都配置，或都不配置。

