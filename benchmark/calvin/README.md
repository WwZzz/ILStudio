# CALVIN 环境使用指南

## 快速开始

### 1. 安装依赖

CALVIN 环境需要在独立的虚拟环境中运行（已配置）：

```bash
# 虚拟环境位置
benchmark/calvin/.venv

# 如需重新安装 imageio-ffmpeg（必需）
benchmark/calvin/.venv/bin/pip install imageio-ffmpeg==0.6.0
```

### 2. 运行评估

使用 dummy policy 测试环境：

```bash
# 使用零动作 dummy policy
benchmark/calvin/.venv/bin/python eval_sim.py \
  -m __dummy-zero7 \
  -e calvin_task_d \
  -n 1 \
  -bs 0 \
  -o results/calvin_eval

# 使用随机动作 dummy policy
benchmark/calvin/.venv/bin/python eval_sim.py \
  -m __dummy-random7 \
  -e calvin_task_d \
  -n 1 \
  -bs 0 \
  -o results/calvin_eval
```

### 3. 使用真实模型

```bash
benchmark/calvin/.venv/bin/python eval_sim.py \
  -m /path/to/your/model \
  -e calvin_task_d \
  -n 1 \
  -bs 0 \
  -o results/calvin_eval
```

## 环境配置

### 可用任务
- `calvin_task_abc`: CALVIN ABC 任务集
- `calvin_task_d`: CALVIN D 任务集（默认）

### 配置文件
- `configs/env/calvin_task_abc.yaml`
- `configs/env/calvin_task_d.yaml`

### 配置参数
```yaml
args:
  task: task_D              # 任务类型
  show_gui: False           # 是否显示GUI
  num_sequences: 1          # 评估序列数量
  sequence_idx: -1          # 序列索引（-1=自动分配）
  max_timesteps: 1800       # 最大步数
  ctrl_space: ee            # 控制空间
  ctrl_type: delta          # 控制类型
  use_wrist: False          # 是否启用腕部相机
```

**腕部相机说明**:
- `use_wrist=False`: 仅使用主相机（static camera, 200×200）
- `use_wrist=True`: 同时使用主相机和腕部相机（gripper camera, 缩放到 200×200）
- 腕部相机自动缩放以匹配主相机分辨率
- 与 libero 环境保持一致的设计

## Dummy Policy 配置

### 格式
```
__dummy-[mode][dim]-chunk[size]
```

### 示例
- `__dummy-7`: 默认 7 维零动作
- `__dummy-zero7`: 7 维零动作
- `__dummy-random7`: 7 维随机动作
- `__dummy-14`: 14 维零动作
- `__dummy-zero7-chunk50`: 7 维零动作，chunk_size=50

## 输出说明

### 目录结构
```
results/calvin_eval/
└── calvin/
    ├── video/
    │   └── task_D_roll0_1.mp4  # 评估视频
    └── example_data/
        ├── image_cam0.png       # 初始帧图像
        ├── state_raw.csv        # 状态数据
        ├── action_raw.csv       # 动作数据
        └── info.txt             # 评估信息
```

### 评估指标
- **Success Rate**: 任务成功率
- **Total Success**: 成功任务数
- **Avg Horizon**: 平均成功步数

## 注意事项

1. **建议使用 `-bs 0`**: CALVIN 环境推荐使用 `SequentialVectorEnv`（设置 `batch_size=0`），已修复多 rollout 支持问题。

2. **多 rollout 支持**: ✅ 已修复！现在支持 `num_rollout > 1`，无需额外配置。

3. **imageio-ffmpeg 必需**: 视频录制需要 `imageio-ffmpeg==0.6.0`，否则会出现编码错误。

4. **图像调整警告**: 由于 CALVIN 图像大小为 200x200（不是16的倍数），imageio 会自动调整到 208x208。这是正常行为，不影响功能。

5. **腕部相机**: 启用 `use_wrist=True` 后，腕部相机会自动从 84×84 缩放到 200×200 以匹配主相机。

## 环境特性

### 观察空间
- **State**: robot_obs (15维) - 机器人状态
- **Image**: (200, 200, 3) - static 相机 RGB 图像

### 动作空间  
- **维度**: 7 (TCP位置 3D, TCP方向 3D, 夹爪 1D)
- **格式**: 连续动作 + 离散夹爪 {-1, 1}

### 视频输出
- **分辨率**: 200x200 (调整到 208x208)
- **帧率**: 默认 10 FPS
- **编码**: H.264 (libx264)

## 故障排除

### ModuleNotFoundError: No module named 'imageio_ffmpeg'
**解决方案**:
```bash
benchmark/calvin/.venv/bin/pip install imageio-ffmpeg==0.6.0
```

### error: Not connected to physics server
**状态**: ✅ 已修复！

如果仍然遇到此错误，请确保：
- 使用最新的代码（包含 `CalvinEnv.close()` 修复）
- 使用 `batch_size=0` 运行（`-bs 0`）

### ValueError: could not broadcast input array
**解决方案**: 确保已安装 `imageio-ffmpeg==0.6.0`

### 腕部相机图像大小不匹配
**解决方案**: 腕部相机已自动缩放到 200×200，无需额外配置

## 更多信息

详细的集成状态和技术细节，请参阅 `INTEGRATION_STATUS.md`
