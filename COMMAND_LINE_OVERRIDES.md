# 命令行参数覆写功能

IL-Studio 支持通过命令行参数覆写任意深度的配置文件参数，提供灵活的配置管理机制。

## 支持的配置类型

系统支持以下6种配置类型的覆写：
- `task` - 任务配置
- `policy` - 策略配置 
- `training` - 训练配置
- `robot` - 机器人配置
- `teleop` - 遥操作配置
- `env` - 环境配置

## 覆写语法

### 基本语法

```bash
--<config_type>.<parameter_path> <value>
```

或者使用等号语法：

```bash
--<config_type>.<parameter_path>=<value>
```

### 支持的嵌套深度

系统支持任意深度的参数嵌套，例如：

- **2层嵌套**: `--policy.camera_names`
- **3层嵌套**: `--policy.model_args.backbone` 
- **4层嵌套**: `--training.optimizer.lr_scheduler.type`
- **5层嵌套**: `--task.env.simulation.physics.timestep`
- **更深嵌套**: `--config.a.b.c.d.e.f.value`

## 覆写示例

### Policy 配置覆写

```bash
# 基本参数覆写
python train.py --policy act --policy.camera_names '["primary", "wrist"]'

# 嵌套参数覆写
python train.py --policy act --policy.model_args.backbone resnet50
python train.py --policy act --policy.model_args.hidden_dim 1024
python train.py --policy act --policy.model_args.enc_layers 6

# 深度嵌套覆写
python train.py --policy act --policy.model_args.optimizer.lr 0.001
python train.py --policy act --policy.model_args.optimizer.lr_scheduler.type cosine
```

### Training 配置覆写

```bash
# 训练参数覆写
python train.py --training.learning_rate 0.0001
python train.py --training.num_train_epochs 10

# 优化器配置覆写
python train.py --training.optimizer.weight_decay 0.01
python train.py --training.optimizer.lr_scheduler.warmup_steps 1000
python train.py --training.optimizer.lr_scheduler.type linear
```

### Task 配置覆写

```bash
# 任务基本参数
python train.py --task.action_dim 14
python train.py --task.state_dim 14

# 环境参数覆写
python train.py --task.env.simulation.physics.timestep 0.01
python train.py --task.env.simulation.physics.gravity -9.81
python train.py --task.env.simulation.rendering.width 1920
python train.py --task.env.simulation.rendering.height 1080
```

### 组合使用

```bash
python train.py \
  --policy act \
  --task sim_transfer_cube_scripted \
  --training default \
  --policy.model_args.backbone resnet50 \
  --policy.model_args.hidden_dim 1024 \
  --policy.camera_names '["primary", "wrist", "overhead"]' \
  --training.learning_rate 0.0001 \
  --training.optimizer.lr_scheduler.type cosine \
  --task.env.simulation.physics.timestep 0.01
```

## 数据类型处理

系统会自动尝试将字符串值转换为合适的数据类型：

- **整数**: `"100"` → `100`
- **浮点数**: `"0.001"` → `0.001`
- **布尔值**: `"true"` → `True`, `"false"` → `False`
- **列表**: `'["a", "b", "c"]'` → `["a", "b", "c"]`
- **字典**: `'{"key": "value"}'` → `{"key": "value"}`
- **字符串**: 无法转换的值保持为字符串

## 覆写优先级

参数覆写的优先级顺序（从高到低）：

1. **命令行覆写** (最高优先级)
2. **YAML 配置文件参数**
3. **默认值** (最低优先级)

这意味着命令行参数总是会覆盖配置文件中的相同参数。

## 实际应用场景

### 1. 快速实验不同的模型架构

```bash
# 尝试不同的backbone
python train.py --policy act --policy.model_args.backbone resnet34
python train.py --policy act --policy.model_args.backbone resnet50

# 调整网络层数
python train.py --policy act --policy.model_args.enc_layers 6
python train.py --policy act --policy.model_args.dec_layers 8
```

### 2. 调试和测试

```bash
# 快速减少训练步数进行测试
python train.py --training.max_steps 100

# 调整batch size
python train.py --training.per_device_train_batch_size 4

# 修改相机配置
python train.py --policy.camera_names '["primary"]'
```

### 3. 超参数搜索

```bash
# 学习率搜索
python train.py --training.learning_rate 0.001
python train.py --training.learning_rate 0.0001
python train.py --training.learning_rate 0.00001

# 网络大小搜索
python train.py --policy.model_args.hidden_dim 256
python train.py --policy.model_args.hidden_dim 512
python train.py --policy.model_args.hidden_dim 1024
```

## 注意事项

1. **引用复杂值**: 对于包含空格、特殊字符或复杂结构的值，请使用引号包围：
   ```bash
   --policy.camera_names '["primary", "wrist"]'
   --policy.description "Multi-camera policy"
   ```

2. **布尔值**: 使用 `true`/`false` 字符串：
   ```bash
   --training.do_eval true
   --policy.model_args.use_pretrained false
   ```

3. **路径参数**: 确保路径的正确性：
   ```bash
   --training.output_dir "/path/to/output"
   --policy.model_args.pretrained_path "/path/to/checkpoint"
   ```

4. **验证参数**: 系统会验证参数的有效性，无效的参数会被报告或忽略。

## 错误处理

- 如果指定了无效的配置类型，系统会忽略该覆写
- 如果参数路径不存在，系统会创建必要的嵌套结构
- 如果数据类型转换失败，系统会保持原始字符串值
- 错误和警告信息会在训练开始时显示

这个强大的覆写机制让你可以在不修改配置文件的情况下灵活地调整任何参数，极大地提高了实验效率！
