# Policy Server 使用指南

## 概述

Policy Server 是一个网络服务，允许客户端发送观察数据（MetaObs）并接收预测的动作序列（MetaAction列表）。这对于机器人控制、远程推理等场景非常有用。

## 架构

```
Client                    Server
  |                         |
  |-- MetaObs (pickle) ---->|
  |                         |-- Policy Inference
  |                         |-- MetaAction List
  |<-- MetaAction (pickle)--|
```

## 快速开始

### 1. 启动服务器

**方法1: 直接使用Python**
```bash
python start_policy_server.py \
    --model_name_or_path ckpt/act_sim_transfer_cube_scripted_zscore_example \
    --host 0.0.0.0 \
    --port 5000 \
    --chunk_size 64
```

**方法2: 使用启动脚本**
```bash
./scripts/start_policy_server.sh \
    --model_path ckpt/act_sim_transfer_cube_scripted_zscore_example \
    --host 0.0.0.0 \
    --port 5000 \
    --chunk_size 64
```

### 2. 测试客户端

```bash
python policy_client_example.py \
    --host localhost \
    --port 5000 \
    --num_requests 5 \
    --delay 1.0
```

## 详细使用

### 服务器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 绑定的主机地址 |
| `--port` | `5000` | 监听端口 |
| `--model_name_or_path` | `ckpt/act_sim_transfer_cube_scripted_zscore_example` | 模型路径 |
| `--dataset_id` | `''` | 数据集ID（默认使用第一个） |
| `--chunk_size` | `64` | 动作序列长度 |
| `--device` | `cuda` | 计算设备 |

### 客户端参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `localhost` | 服务器地址 |
| `--port` | `5000` | 服务器端口 |
| `--num_requests` | `5` | 请求数量 |
| `--delay` | `1.0` | 请求间隔（秒） |

## 通信协议

### 数据格式

**发送 (Client → Server)**:
```
[4 bytes length (big-endian)] + [pickled MetaObs data]
```

**接收 (Server → Client)**:
```
[4 bytes length (big-endian)] + [pickled list of MetaAction]
```

### MetaObs 结构

```python
@dataclass
class MetaObs:
    state: np.ndarray = None           # Robot state
    state_ee: np.ndarray = None        # End-effector state
    state_joint: np.ndarray = None     # Joint state
    state_obj: np.ndarray = None       # Object state
    image: np.ndarray = None           # Camera images (K, C, H, W)
    depth: np.ndarray = None           # Depth images (K, H, W)
    pc: np.array = None                # Point cloud (n, 3)
    raw_lang: str = ''                 # Language instruction
    timestep: int = -1                 # Timestep
```

### MetaAction 结构

```python
@dataclass
class MetaAction:
    ctrl_space: str = 'ee'             # 'ee' or 'joint'
    ctrl_type: str = 'delta'           # 'abs', 'delta', 'relative'
    action: np.ndarray = None          # Action vector
    gripper_continuous: bool = False   # Gripper control type
```

## 使用示例

### 1. 基本服务器启动

```bash
# 使用默认参数
python start_policy_server.py

# 自定义参数
python start_policy_server.py \
    --model_name_or_path ckpt/my_model \
    --host 192.168.1.100 \
    --port 5001 \
    --dataset_id my_robot_dataset \
    --chunk_size 32
```

### 2. 使用启动脚本

```bash
# 基本用法
./scripts/start_policy_server.sh

# 自定义参数
./scripts/start_policy_server.sh \
    --host 192.168.1.100 \
    --port 5001 \
    --model_path ckpt/my_model \
    --dataset_id my_robot_dataset \
    --chunk_size 32
```

### 3. 客户端连接

```python
from policy_client_example import PolicyClient
from benchmark.base import MetaObs
import numpy as np

# 创建客户端
client = PolicyClient('localhost', 5000)
client.connect()

# 创建观察数据
meta_obs = MetaObs(
    state=np.random.randn(14),
    image=np.random.randint(0, 255, (1, 3, 480, 640), dtype=np.uint8),
    raw_lang="Pick up the red cube",
    timestep=0
)

# 发送请求并获取动作
mact_list = client.send_meta_obs(meta_obs)
print(f"Received {len(mact_list)} actions")

# 断开连接
client.disconnect()
```

## 高级用法

### 1. 多客户端支持

服务器支持多个客户端同时连接，每个客户端在独立线程中处理：

```bash
# 启动服务器
python start_policy_server.py --port 5000

# 在多个终端中启动客户端
python policy_client_example.py --port 5000 &
python policy_client_example.py --port 5000 &
python policy_client_example.py --port 5000 &
```

### 2. 自定义客户端

```python
import socket
import pickle
import struct
from benchmark.base import MetaObs, MetaAction

class CustomPolicyClient:
    def __init__(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
    
    def get_actions(self, obs_data):
        # 创建MetaObs
        meta_obs = MetaObs(
            state=obs_data['state'],
            image=obs_data['image'],
            raw_lang=obs_data['instruction']
        )
        
        # 发送数据
        data_bytes = pickle.dumps(meta_obs)
        length_bytes = struct.pack('>I', len(data_bytes))
        self.socket.sendall(length_bytes + data_bytes)
        
        # 接收响应
        length_bytes = self.socket.recv(4)
        data_length = struct.unpack('>I', length_bytes)[0]
        data_bytes = self.socket.recv(data_length)
        mact_list = pickle.loads(data_bytes)
        
        return mact_list
    
    def close(self):
        self.socket.close()
```

### 3. 错误处理

服务器具有完善的错误处理机制：

- **连接错误**: 自动重试和清理
- **推理错误**: 返回空动作列表
- **序列化错误**: 记录错误并断开连接
- **网络错误**: 优雅关闭连接

### 4. 性能监控

服务器提供详细的日志输出：

```
🚀 Policy Server started on 0.0.0.0:5000
   Ctrl+C to stop the server
   Waiting for connections...

✓ Client #1 connected from ('127.0.0.1', 54321)
  Client #1: 10 requests processed
  Client #1: 20 requests processed
  Client #1 connection closed (processed 25 requests)
```

## 故障排除

### 1. 连接被拒绝

**问题**: `ConnectionRefusedError: [Errno 111] Connection refused`

**解决方案**:
- 检查服务器是否正在运行
- 确认端口号正确
- 检查防火墙设置

### 2. 模型加载失败

**问题**: `FileNotFoundError: No normalize.json found`

**解决方案**:
- 确认模型路径正确
- 检查normalize.json文件是否存在
- 验证模型是否已训练完成

### 3. 内存不足

**问题**: `CUDA out of memory`

**解决方案**:
- 使用CPU: `--device cpu`
- 减少chunk_size: `--chunk_size 32`
- 关闭其他GPU程序

### 4. 序列化错误

**问题**: `pickle.PickleError`

**解决方案**:
- 检查MetaObs数据格式
- 确保numpy数组类型正确
- 验证数据维度匹配

## 最佳实践

### 1. 服务器配置

- **生产环境**: 使用专用服务器，配置防火墙
- **开发环境**: 使用localhost，便于调试
- **多GPU**: 使用负载均衡分发请求

### 2. 客户端设计

- **连接池**: 复用连接减少开销
- **错误重试**: 实现指数退避重试
- **数据验证**: 验证MetaObs数据完整性

### 3. 性能优化

- **批处理**: 一次发送多个观察
- **异步处理**: 使用异步客户端
- **数据压缩**: 压缩图像数据

## 扩展功能

### 1. 添加认证

```python
# 在PolicyServer中添加
def authenticate_client(self, client_socket):
    # 实现认证逻辑
    pass
```

### 2. 添加监控

```python
# 添加性能监控
import time
import psutil

def log_performance(self):
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    print(f"CPU: {cpu_usage}%, Memory: {memory_usage}%")
```

### 3. 添加配置

```python
# 支持配置文件
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```

## 总结

Policy Server 提供了一个灵活、高效的网络推理服务，支持：

✅ **多客户端并发** - 同时处理多个连接  
✅ **自动错误处理** - 优雅处理各种异常  
✅ **灵活配置** - 支持多种参数组合  
✅ **易于集成** - 简单的客户端API  
✅ **高性能** - 优化的网络通信协议  

适用于机器人控制、远程推理、分布式系统等场景。
