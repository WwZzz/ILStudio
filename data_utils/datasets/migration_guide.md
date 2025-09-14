# 数据集优化迁移指南

本指南将帮助您从原始的`EpisodicDataset`迁移到优化的`OptimizedEpisodicDataset`，以解决HDF5数据加载的IO瓶颈和多卡内存超额问题。

## 问题背景

### 原始实现的问题
1. **IO瓶颈**: 每次数据访问都需要打开HDF5文件进行IO操作
2. **内存超额**: 在`accelerate launch`多卡环境下，每个进程都会加载完整数据集到内存
3. **缓存效率低**: 没有有效的数据缓存机制

### 优化方案的优势
1. **内存映射**: 使用HDF5内存映射减少IO操作
2. **共享内存**: 多进程间共享数据，避免重复加载
3. **智能缓存**: LRU缓存和自适应缓存策略
4. **自动优化**: 根据系统资源自动选择最优配置

## 快速迁移

### 1. 最简单的迁移方式

如果您只想快速获得性能提升，只需要更换导入：

```python
# 原始代码
from data_utils.datasets.base import EpisodicDataset

# 优化后代码  
from data_utils.datasets.optimized_base import OptimizedEpisodicDataset
```

然后将您的数据集类继承从`OptimizedEpisodicDataset`：

```python
# 原始代码
class YourDataset(EpisodicDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# 优化后代码
class YourDataset(OptimizedEpisodicDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
```

### 2. 启用自动优化

使用自动优化器获得更好的性能：

```python
from data_utils.datasets.auto_optimizer import auto_optimize_dataset
from data_utils.datasets.optimized_usage_example import create_optimized_dataloader

# 自动优化配置
config, analysis = auto_optimize_dataset(
    dataset_paths=your_dataset_paths,
    camera_names=your_camera_names,
    batch_size=8,
    target_memory_usage=0.7  # 使用70%的可用内存
)

# 创建优化的数据加载器
dataloader, dataset = create_optimized_dataloader(
    dataset_paths=your_dataset_paths,
    camera_names=your_camera_names,
    batch_size=8,
    num_workers=config['dataloader']['num_workers'],
    use_distributed=True,  # 多卡环境
    **config['optimization']
)
```

## 详细迁移步骤

### 步骤1: 实现子类

创建您的优化数据集类：

```python
from data_utils.datasets.optimized_base import OptimizedEpisodicDataset

class OptimizedYourDataset(OptimizedEpisodicDataset):
    def __init__(self, *args, **kwargs):
        # 添加您特定的参数
        self.your_specific_param = kwargs.pop('your_param', default_value)
        super().__init__(*args, **kwargs)
    
    def get_language_instruction(self):
        """实现语言指令获取"""
        # 根据您的数据集返回相应的指令
        return "your task instruction"
    
    def _fallback_load_onestep(self, dataset_path: str, start_ts: int):
        """实现回退加载方法"""
        # 当内存映射失败时的回退逻辑
        # 可以复制您原始的load_onestep_from_episode逻辑
        return self._original_load_method(dataset_path, start_ts)
```

### 步骤2: 配置优化参数

根据您的数据集特点配置优化参数：

```python
# 小数据集配置 (< 1GB)
small_dataset_config = {
    'use_shared_memory': True,
    'memory_map_cache_size': 50,
    'preload_critical_data': True
}

# 大数据集配置 (> 10GB)  
large_dataset_config = {
    'use_shared_memory': True,
    'memory_map_cache_size': 200,
    'preload_critical_data': True
}

dataset = OptimizedYourDataset(
    dataset_path_list=paths,
    camera_names=cameras,
    **large_dataset_config
)
```

### 步骤3: 多卡环境配置

对于`accelerate launch`环境：

```python
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# 在训练脚本中
def create_distributed_dataloader(dataset_paths, camera_names, batch_size):
    # 自动优化配置
    config, _ = auto_optimize_dataset(
        dataset_paths, camera_names, batch_size
    )
    
    # 创建数据集
    dataset = OptimizedYourDataset(
        dataset_path_list=dataset_paths,
        camera_names=camera_names,
        use_shared_memory=True,  # 重要：启用共享内存
        **config['optimization']
    )
    
    # 分布式采样器
    sampler = DistributedSampler(dataset)
    
    # 数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    
    return dataloader, dataset
```

## 性能调优

### 内存使用监控

```python
import psutil
import gc

def monitor_memory_usage(dataset):
    """监控内存使用"""
    process = psutil.Process()
    
    print(f"内存使用: {process.memory_info().rss / 1024**3:.2f} GB")
    print(f"共享内存块数: {len(dataset.shared_memory_manager.shared_blocks)}")
    
    # 定期清理缓存
    if process.memory_info().rss > 16 * 1024**3:  # 超过16GB
        dataset.clear_cache()
        gc.collect()
```

### 缓存策略调优

```python
# 根据数据访问模式调整缓存
if your_data_access_is_sequential:
    cache_size = 50  # 顺序访问用较小缓存
else:
    cache_size = 200  # 随机访问用较大缓存

dataset = OptimizedYourDataset(
    memory_map_cache_size=cache_size,
    # ... 其他参数
)
```

## 故障排除

### 常见问题

1. **共享内存错误**
   ```
   FileExistsError: [Errno 17] File exists
   ```
   解决方案：清理遗留的共享内存
   ```python
   # 在脚本开始时
   import shutil
   shutil.rmtree('/dev/shm/shared_memory_*', ignore_errors=True)
   ```

2. **内存映射失败**
   ```
   OSError: Unable to open file
   ```
   解决方案：检查文件权限和路径
   ```python
   # 确保文件可读
   import os
   for path in dataset_paths:
       assert os.access(path, os.R_OK), f"无法读取文件: {path}"
   ```

3. **多进程锁死**
   解决方案：设置合适的worker数量
   ```python
   # 减少worker数量
   num_workers = min(4, os.cpu_count() // 2)
   ```

### 性能对比测试

```python
import time

def benchmark_datasets(original_dataset, optimized_dataset, num_batches=100):
    """对比性能"""
    
    # 原始数据集
    start_time = time.time()
    for i, batch in enumerate(original_dataset):
        if i >= num_batches:
            break
    original_time = time.time() - start_time
    
    # 优化数据集  
    start_time = time.time()
    for i, batch in enumerate(optimized_dataset):
        if i >= num_batches:
            break
    optimized_time = time.time() - start_time
    
    print(f"原始数据集加载时间: {original_time:.2f}秒")
    print(f"优化数据集加载时间: {optimized_time:.2f}秒") 
    print(f"性能提升: {original_time/optimized_time:.2f}x")
```

## 最佳实践

1. **根据数据集大小选择策略**
   - < 1GB: 启用完全预加载
   - 1-10GB: 使用内存映射 + 适中缓存
   - > 10GB: 使用内存映射 + 共享内存

2. **多卡环境优化**
   - 总是启用共享内存
   - 使用分布式采样器
   - 适当增加prefetch_factor

3. **内存管理**
   - 设置合理的内存限制
   - 定期清理缓存
   - 监控内存使用

4. **调试模式**
   ```python
   # 开发时启用详细日志
   dataset = OptimizedYourDataset(
       ...,
       verbose=True,
       enable_monitoring=True
   )
   ```

通过以上迁移步骤，您应该能够显著提升数据加载性能，并解决多卡环境下的内存问题。
