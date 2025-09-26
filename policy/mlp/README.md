# MLP Policy for IL-Studio

简单的多层感知机（MLP）策略，适用于模仿学习任务。

## 特性

- **纯状态输入**: 默认只使用状态向量作为输入
- **多模态支持**: 可选的相机图像输入支持
- **分块输出**: 支持输出多步动作序列 (chunk_size > 1)
- **灵活配置**: 可配置层数、维度、激活函数等参数
- **框架兼容**: 完全兼容 IL-Studio 框架接口

## 架构

### 输入
- **状态模式**: `state_dim` 维度的状态向量
- **多模态模式**: `state_dim + image_dim` 维度的拼接向量

### 网络结构
```
输入层: Linear(input_dim, hidden_dim) + Activation
隐藏层: [Linear(hidden_dim, hidden_dim) + Activation] × (num_layers - 2)
输出层: Linear(hidden_dim, chunk_size * action_dim)
输出重塑: reshape 为 (batch_size, chunk_size, action_dim)
```

### 输出
- 形状: `(batch_size, chunk_size, action_dim)`
- 支持多步动作预测

## 配置文件

### 基础配置 (`configs/policy/mlp.yaml`)
```yaml
name: mlp
module_path: policy.mlp
config_class: MLPPolicyConfig
model_class: MLPPolicy
data_processor: get_data_processor
data_collator: get_data_collator
trainer_class: null
pretrained_config:
  model_name_or_path: null
  is_pretrained: false
model_args:
  state_dim: 14          # 状态维度
  action_dim: 14         # 动作维度
  num_layers: 3          # 网络层数
  hidden_dim: 256        # 隐藏层维度
  activation: "relu"     # 激活函数
  dropout: 0.1           # Dropout 率
  use_camera: false      # 是否使用相机
  chunk_size: 1          # 动作序列长度
  learning_rate: 1e-3    # 学习率
```

### 多模态配置 (`configs/policy/mlp_camera.yaml`)
```yaml
name: mlp_camera
module_path: policy.mlp
model_args:
  use_camera: true       # 启用相机输入
  num_layers: 4          # 更多层处理图像
  hidden_dim: 512        # 更大的隐藏维度
```

## 使用方法

### 训练
```bash
python train.py --policy mlp --config configs/policy/mlp.yaml
```

### 评估
```bash
python eval.py --policy mlp --model_name_or_path /path/to/checkpoint
```

### 多模态训练
```bash
python train.py --policy mlp --config configs/policy/mlp_camera.yaml
```

## 框架接口

按照 IL-Studio 的策略规则，MLP 策略提供了三个必要接口：

### 1. `load_model(args)`
- 加载原始模型或训练过的检查点
- 返回包含 `model` 键的字典

### 2. `get_data_processor(args, model_components)`
- 返回样本级数据处理器
- 处理状态向量和可选的图像数据
- 确保数据格式符合模型要求

### 3. `get_data_collator(args, model_components)`
- 返回批处理函数
- 将多个样本组织成批次
- 处理不同数据类型的张量化
- **自动忽略文本模态**（raw_lang, instruction, task 等）

### 模型推理接口

### `select_action(obs)`
- 用于评估和推理时的动作选择
- 输入: 包含 `state` (和可选 `image`) 的观察字典
- 输出: numpy 数组格式的动作序列

## 数据处理策略

### 支持的模态
- ✅ **状态模态**: `state` - 机器人状态向量
- ✅ **图像模态**: `image` - 相机图像（可选）
- ✅ **动作模态**: `action` - 目标动作序列

### 忽略的模态
- ❌ **文本模态**: 自动忽略所有文本相关字段
  - `raw_lang`, `lang`, `language`, `text`
  - `instruction`, `task_description`
  - `task`, `episode_id`, `trajectory_id`
  - `dataset_name` 等

### 处理流程
1. **数据处理器**: 从输入样本中提取相关模态，忽略文本
2. **批处理器**: 只对 `state`, `action`, `image` 进行张量化
3. **模型推理**: 仅使用状态和图像信息进行动作预测

## 设计原则

1. **简单性**: MLP 是最简单的神经网络架构，适合快速原型和基准测试
2. **模块化**: 支持可选的图像模态，便于扩展
3. **框架兼容**: 严格遵循 IL-Studio 的接口规范
4. **配置灵活**: 通过 YAML 配置文件轻松调整参数

## 适用场景

- **状态空间简单**: 低维度状态观察的任务
- **快速原型**: 新任务的基线模型
- **基准测试**: 与复杂模型比较的简单基准
- **调试验证**: 验证数据处理管道的正确性

## 注意事项

1. MLP 不处理序列信息，每个时间步独立预测
2. 图像输入会被展平，丢失空间结构信息
3. 适合相对简单的控制任务
4. 对于复杂的视觉-运动任务，建议使用 CNN 或 Transformer 策略
