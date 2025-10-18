# 数据集配置系统文档

## 概述

新的数据集配置系统支持灵活的多数据集配置，同时保持向后兼容性。系统支持两种配置格式：

1. **新的灵活格式**：支持多数据集、动态类加载、自定义参数
2. **传统格式**：完全向后兼容现有配置

## 新的灵活配置格式

### 基本结构

```yaml
# 数据集列表 - 每个数据集独立配置
datasets:
  - name: "dataset_name"
    class: "full.module.path.ClassName"
    args:
      # 数据集构造函数的参数
      dataset_path_list: ['path/to/data']  # EpisodicDataset 使用 dataset_path_list
      camera_names: ['primary']
      ctrl_space: 'ee'
      ctrl_type: 'delta'
      chunk_size: 64
      custom_param: value

# 其他任务参数
action_dim: 14
state_dim: 14
# ...
```

### 关键特性

#### 1. 动态类加载

支持多种类路径格式：

```yaml
datasets:
  # 完整路径
  - class: "data_utils.datasets.EpisodicDataset"
  
  # 简写（自动补全为 data_utils.datasets.ClassName）
  - class: "EpisodicDataset"
  
  # 自定义模块
  - class: "my_module.MyCustomDataset"
  
  # RLDS 包装器
  - class: "data_utils.datasets.rlds_wrapper.WrappedTFDSDataset"
```

#### 2. 灵活的参数配置

每个数据集可以有完全不同的构造参数：

```yaml
datasets:
  # 标准数据集
  - name: "standard"
    class: "EpisodicDataset"
    args:
      dataset_path_list: ['path1']
      camera_names: ['primary']
      chunk_size: 64
  
  # 自定义数据集
  - name: "custom"
    class: "MyCustomDataset"
    args:
      data_path: 'path2'
      custom_param1: 'value'
      custom_param2: 42
      any_other_param: [1, 2, 3]
```

#### 3. 独立配置

每个数据集都需要完整的独立配置，确保清晰和灵活：

```yaml
datasets:
  - name: "dataset1"
    class: "EpisodicDataset"
    args:
      dataset_path_list: ['path1']
      camera_names: ['primary']
      chunk_size: 64
      ctrl_space: 'ee'
      ctrl_type: 'delta'
  
  - name: "dataset2"
    class: "EpisodicDataset"
    args:
      dataset_path_list: ['path2']
      camera_names: ['primary', 'wrist']
      chunk_size: 100
      ctrl_space: 'joint'
      ctrl_type: 'abs'
```

## 传统配置格式（向后兼容）

原有的配置格式完全保持兼容：

```yaml
dataset_dir:
  - 'data/path1'
  - 'data/path2'

dataset_class: 'EpisodicDataset'
camera_names: ['primary']
ctrl_space: 'ee'
ctrl_type: 'delta'
chunk_size: 64

# 其他参数...
action_dim: 14
state_dim: 14
```

## 使用示例

### 示例 1: 多种数据集混合

```yaml
datasets:
  # 仿真数据
  - name: "sim_data"
    class: "AlohaSimDataset"
    args:
      dataset_path_list: ['data/sim']
      camera_names: ['primary']
      chunk_size: 64
      ctrl_space: 'joint'
      ctrl_type: 'abs'
  
  # 真实数据
  - name: "real_data"
    class: "EpisodicDataset"
    args:
      dataset_path_list: ['data/real']
      camera_names: ['primary', 'wrist']
      chunk_size: 100
      ctrl_space: 'ee'
      ctrl_type: 'delta'
  
  # RLDS 数据
  - name: "rlds_data"
    class: "data_utils.datasets.rlds_wrapper.WrappedTFDSDataset"
    args:
      dataset_path_list: ['data/rlds']
      camera_names: ['primary']
      chunk_size: 32
      ctrl_space: 'ee'
      ctrl_type: 'delta'

action_dim: 14
state_dim: 14
```

### 示例 2: 自定义数据集

```yaml
datasets:
  - name: "my_custom_dataset"
    class: "my_package.datasets.MyCustomDataset"
    args:
      data_root: '/path/to/data'
      preprocessing_config:
        resize_images: true
        target_size: [224, 224]
        normalize: true
      augmentation_config:
        random_crop: true
        color_jitter: 0.1
      custom_feature: 'special_value'

action_dim: 7
state_dim: 7
```

## 实现细节

### 类加载机制

`_import_class_from_path()` 函数负责动态加载类：

1. 如果类路径不包含 `.`，自动补全为 `data_utils.datasets.ClassName`
2. 使用 `importlib.import_module()` 动态导入模块
3. 使用 `getattr()` 获取类对象
4. 提供详细的错误信息

### 参数合并机制

`_create_dataset_from_config()` 函数处理参数合并：

1. 从 `global_dataset_config` 开始
2. 合并数据集特定的配置
3. 处理传统参数的兼容性映射
4. 添加必要的默认参数（如 `data_args`）

### 向后兼容性

`load_data()` 函数自动检测配置格式：

- 如果存在 `datasets` 键，使用新格式
- 否则使用传统格式处理

## 迁移指南

### 从传统格式迁移到新格式

**原配置：**
```yaml
dataset_dir: ['data/path1', 'data/path2']
dataset_class: 'EpisodicDataset'
camera_names: ['primary']
ctrl_space: 'ee'
ctrl_type: 'delta'
chunk_size: 64
```

**新配置：**
```yaml
datasets:
  - name: "dataset1"
    class: "EpisodicDataset"
    args:
      dataset_path_list: ['data/path1']
      camera_names: ['primary']
      ctrl_space: 'ee'
      ctrl_type: 'delta'
      chunk_size: 64
  
  - name: "dataset2"
    class: "EpisodicDataset"
    args:
      dataset_path_list: ['data/path2']
      camera_names: ['primary']
      ctrl_space: 'ee'
      ctrl_type: 'delta'
      chunk_size: 64
```

### 添加自定义数据集

1. 创建自定义数据集类
2. 在配置中指定完整的类路径
3. 在 `args` 中提供所需的构造参数

```yaml
datasets:
  - name: "my_dataset"
    class: "my_module.MyDataset"
    args:
      # 任何自定义参数
      param1: value1
      param2: value2
```

## 错误处理

系统提供详细的错误信息：

- 类导入失败：显示完整的导入路径和错误原因
- 数据集创建失败：显示类名、参数和具体错误
- 参数验证失败：显示缺失或无效的参数

## 最佳实践

1. **独立配置**：每个数据集都应该有完整的独立配置，确保清晰性
2. **明确命名**：为每个数据集提供有意义的 `name`
3. **完整路径**：对于自定义类，使用完整的模块路径
4. **参数文档**：在配置文件中添加注释说明参数用途
5. **必要参数**：确保每个数据集都包含必要的参数（camera_names, ctrl_space, ctrl_type等）
6. **渐进迁移**：可以逐步从传统格式迁移到新格式
