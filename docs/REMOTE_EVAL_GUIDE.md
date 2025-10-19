# 远程策略评估指南

## 概述

远程策略评估允许你使用运行在远程服务器上的策略模型进行环境评估，而无需在本地加载模型和归一化器。这对于以下场景非常有用：

- **分布式推理**: 模型运行在高性能服务器上，评估在不同机器上进行
- **资源优化**: 避免在评估机器上重复加载大型模型
- **服务化部署**: 将策略作为服务提供给多个客户端
- **跨平台评估**: 在不同硬件平台上进行评估

## 架构

```
Evaluation Client          Policy Server
     |                         |
     |-- MetaObs (network) ---->|-- Local Model
     |                         |-- Normalizers  
     |<-- MetaAction (network)--|-- GPU Inference
     |                         |
   Env Loop                 Action Generation
```

## 快速开始

### 1. 启动策略服务器

```bash
# 启动服务器
python start_policy_server.py \
    --model_name_or_path ckpt/act_sim_transfer_cube_scripted_zscore_example \
    --host 0.0.0.0 \
    --port 5000 \
    --chunk_size 64
```

### 2. 运行远程评估

```bash
# 远程评估
python eval_remote.py \
    --model_name_or_path localhost:5000 \
    --env aloha \
    --num_rollout 10 \
    --num_envs 2 \
    --chunk_size 64
```

### 3. 使用便捷脚本

```bash
# 使用示例脚本
./scripts/example_eval_remote.sh
```

## 详细使用

### 服务器端设置

**1. 启动策略服务器**

```bash
# 基本启动
python start_policy_server.py

# 自定义配置
python start_policy_server.py \
    --model_name_or_path ckpt/my_model \
    --host 0.0.0.0 \
    --port 5000 \
    --dataset_id my_dataset \
    --chunk_size 64 \
    --device cuda
```

**2. 使用启动脚本**

```bash
./scripts/start_policy_server.sh \
    --model_path ckpt/my_model \
    --host 0.0.0.0 \
    --port 5000
```

### 客户端评估

**1. 远程模式 (推荐)**

```bash
python eval_remote.py \
    --model_name_or_path "server_host:port" \
    --env aloha \
    --num_rollout 10 \
    --chunk_size 64
```

**2. 本地模式 (兼容性)**

```bash
python eval_remote.py \
    --model_name_or_path "ckpt/local_model" \
    --dataset_id my_dataset \
    --env aloha \
    --num_rollout 10
```

### 参数说明

#### 服务器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name_or_path` | `ckpt/act_sim_transfer_cube_scripted_zscore_example` | 模型路径 |
| `--host` | `0.0.0.0` | 绑定地址 |
| `--port` | `5000` | 监听端口 |
| `--dataset_id` | `''` | 数据集ID |
| `--chunk_size` | `64` | 动作序列长度 |
| `--device` | `cuda` | 计算设备 |

#### 客户端参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name_or_path` | `localhost:5000` | 服务器地址或本地模型路径 |
| `--env` | `aloha` | 环境配置 |
| `--num_rollout` | `4` | 评估轮数 |
| `--num_envs` | `2` | 并行环境数 |
| `--chunk_size` | `64` | 动作序列长度 |
| `--output_dir` | `results/...` | 结果保存目录 |

## 工作原理

### 1. 地址检测

`eval_remote.py` 自动检测 `--model_name_or_path` 参数：

- **服务器地址格式**: `host:port` (如 `localhost:5000`, `192.168.1.100:5001`)
- **本地模型格式**: 文件路径 (如 `ckpt/my_model`, `/path/to/checkpoint`)

```python
# 自动检测示例
if is_server_address(args.model_name_or_path):
    # 使用远程服务器
    host, port = parse_server_address(args.model_name_or_path)
    policy = RemotePolicyClient(host, port, chunk_size)
else:
    # 使用本地模型
    policy = MetaPolicy(local_model, normalizers, ...)
```

### 2. 远程策略客户端

`RemotePolicyClient` 类模拟 `MetaPolicy` 的接口：

```python
class RemotePolicyClient:
    def select_action(self, mobs: MetaObs, t: int):
        # 当需要新chunk时请求服务器
        if t % self.chunk_size == 0 or self.is_action_queue_empty():
            mact_list = self._send_meta_obs(mobs)
            self.action_queue.extend(mact_list)
        
        # 从队列返回动作
        return self.action_queue.popleft()
    
    def reset(self):
        # 清空动作队列
        self.action_queue.clear()
```

### 3. 通信协议

**数据格式**: `[4 bytes length] + [pickled data]`

**请求流程**:
1. 客户端发送 `MetaObs` (观察数据)
2. 服务器返回 `List[MetaAction]` (动作序列)
3. 客户端缓存动作并逐步使用

### 4. 错误处理

- **连接错误**: 自动重连一次
- **服务器错误**: 返回空动作列表时抛出异常
- **网络中断**: 优雅处理并提供错误信息

## 使用示例

### 示例1: 基本远程评估

```bash
# 1. 启动服务器
python start_policy_server.py --port 5000

# 2. 运行评估
python eval_remote.py --model_name_or_path localhost:5000
```

### 示例2: 跨机器评估

```bash
# 服务器机器 (192.168.1.100)
python start_policy_server.py \
    --host 0.0.0.0 \
    --port 5000 \
    --model_name_or_path ckpt/my_model

# 客户端机器
python eval_remote.py \
    --model_name_or_path 192.168.1.100:5000 \
    --num_rollout 20 \
    --output_dir results/remote_eval
```

### 示例3: 多环境并行评估

```bash
python eval_remote.py \
    --model_name_or_path localhost:5000 \
    --env aloha \
    --num_rollout 50 \
    --num_envs 10 \
    --chunk_size 32
```

### 示例4: 使用便捷脚本

```bash
# 修改脚本中的配置
vim scripts/example_eval_remote.sh

# 运行评估
./scripts/example_eval_remote.sh
```

## 性能优化

### 1. 网络优化

- **本地网络**: 使用 `localhost` 或 `127.0.0.1`
- **局域网**: 确保网络稳定，避免防火墙阻挡
- **广域网**: 考虑网络延迟对评估速度的影响

### 2. 并发优化

```bash
# 增加并行环境数
--num_envs 8

# 减少chunk_size降低延迟
--chunk_size 32

# 使用spawn模式避免共享内存问题
--use_spawn
```

### 3. 服务器优化

```bash
# 使用高性能GPU
--device cuda

# 优化模型路径
--model_name_or_path /fast_ssd/models/my_model
```

## 故障排除

### 1. 连接问题

**问题**: `ConnectionRefusedError`

**解决方案**:
```bash
# 检查服务器是否运行
netstat -tlnp | grep 5000

# 检查防火墙
sudo ufw allow 5000

# 检查地址格式
ping server_host
```

### 2. 性能问题

**问题**: 评估速度慢

**解决方案**:
- 减少 `chunk_size`
- 增加 `num_envs`
- 使用更快的网络连接
- 检查服务器GPU利用率

### 3. 内存问题

**问题**: 服务器内存不足

**解决方案**:
```bash
# 减少并发连接
# 优化模型大小
# 使用CPU模式
--device cpu
```

### 4. 数据格式问题

**问题**: `pickle.PickleError`

**解决方案**:
- 检查客户端和服务器版本一致
- 验证 `MetaObs` 数据格式
- 确保numpy数组类型正确

## 高级用法

### 1. 自定义客户端

```python
from eval_remote import RemotePolicyClient

# 创建自定义客户端
client = RemotePolicyClient('localhost', 5000, chunk_size=64)

# 发送观察并获取动作
mobs = MetaObs(state=state, image=image, raw_lang=instruction)
action = client.select_action(mobs, timestep=0)

# 清理
client.reset()
```

### 2. 批量评估

```bash
# 评估多个环境
for env in aloha real_world simulation; do
    python eval_remote.py \
        --model_name_or_path localhost:5000 \
        --env $env \
        --output_dir results/remote_eval_$env
done
```

### 3. 配置文件

```yaml
# eval_config.yaml
server:
  host: "192.168.1.100"
  port: 5000
  
evaluation:
  env: "aloha"
  num_rollout: 20
  num_envs: 4
  chunk_size: 64
  output_dir: "results/remote_eval"
```

## 最佳实践

### 1. 开发环境

- 使用 `localhost` 进行本地测试
- 先验证本地模式再切换到远程模式
- 使用小规模测试验证连接

### 2. 生产环境

- 使用专用服务器运行策略服务
- 配置防火墙和网络安全
- 监控服务器资源使用情况

### 3. 调试技巧

```bash
# 启用详细日志
python eval_remote.py --model_name_or_path localhost:5000 -v

# 测试连接
python policy_client_example.py --host localhost --port 5000 --num_requests 1

# 检查服务器状态
curl -v telnet://localhost:5000
```

## 与本地评估的对比

| 特性 | 本地评估 | 远程评估 |
|------|----------|----------|
| **模型加载** | 需要本地加载 | 服务器端加载 |
| **内存使用** | 客户端高 | 客户端低 |
| **网络依赖** | 无 | 有 |
| **扩展性** | 受限 | 高 |
| **延迟** | 低 | 中等 |
| **部署复杂度** | 低 | 中等 |

## 总结

远程策略评估提供了灵活的分布式推理解决方案：

✅ **资源优化** - 服务器端统一推理，客户端轻量化  
✅ **易于扩展** - 支持多客户端并发评估  
✅ **部署灵活** - 支持跨机器、跨平台部署  
✅ **向下兼容** - 自动检测并支持本地模式  
✅ **错误恢复** - 完善的错误处理和重连机制  

适用于大规模评估、分布式系统、服务化部署等场景。
