# 数据集 Normalizer 包装器指南

## 📌 概述

新的数据集包装器系统通过**外部包装**的方式为数据集添加归一化功能，而不需要每个数据集类实现 `set_action_normalizers` 和 `set_state_normalizers` 方法。

## ✨ 核心优势

1. **无侵入性**: 数据集类不需要实现任何特定的归一化方法
2. **灵活支持**: 使用鸭子类型（duck typing），支持任何实现标准接口的数据集
3. **保持原有性质**: 
   - Map-style 数据集（有 `__getitem__` 和 `__len__`）→ 包装后仍是 map-style
   - Iterable 数据集（有 `__iter__`）→ 包装后仍是 iterable
4. **透明转发**: 包装器透明地转发所有属性和方法调用到原始数据集

## 🔧 实现原理

### 鸭子类型检测

包装器使用鸭子类型来检测数据集类型，而不是检查继承关系：

```python
# 检查数据集实现了哪些方法
has_getitem = hasattr(dataset, '__getitem__') and callable(getattr(dataset, '__getitem__'))
has_iter = hasattr(dataset, '__iter__') and callable(getattr(dataset, '__iter__'))
has_len = hasattr(dataset, '__len__') and callable(getattr(dataset, '__len__'))

# 根据方法判断类型
if has_getitem and has_len:
    # Map-style 数据集
    return NormalizedMapDataset(...)
elif has_iter:
    # Iterable 数据集
    return NormalizedIterableDataset(...)
```

### 两种包装器类

#### 1. `NormalizedMapDataset` - Map-style 数据集包装器

用于有 `__getitem__` 和 `__len__` 方法的数据集：

```python
class NormalizedMapDataset(Dataset):
    def __getitem__(self, idx):
        sample = self.dataset[idx]  # 从原始数据集获取样本
        # 应用归一化
        if self.action_normalizer:
            sample['action'] = self.action_normalizer.normalize(sample['action'])
        if self.state_normalizer:
            sample['state'] = self.state_normalizer.normalize(sample['state'])
        return sample
    
    def __len__(self):
        return len(self.dataset)
```

#### 2. `NormalizedIterableDataset` - Iterable 数据集包装器

用于有 `__iter__` 方法的数据集：

```python
class NormalizedIterableDataset(IterableDataset):
    def __iter__(self):
        for sample in self.dataset:  # 迭代原始数据集
            # 应用归一化
            if self.action_normalizer:
                sample['action'] = self.action_normalizer.normalize(sample['action'])
            if self.state_normalizer:
                sample['state'] = self.state_normalizer.normalize(sample['state'])
            yield sample
```

## 💻 使用方法

### 自动包装（推荐）

使用 `wrap_dataset_with_normalizers` 函数自动检测并包装：

```python
from data_utils.dataset_wrappers import wrap_dataset_with_normalizers

# 创建原始数据集
dataset = MyDataset(...)

# 准备 normalizers
action_normalizers = {dataset_name: action_normalizer}
state_normalizers = {dataset_name: state_normalizer}

# 自动包装
wrapped_dataset = wrap_dataset_with_normalizers(
    dataset=dataset,
    action_normalizers=action_normalizers,
    state_normalizers=state_normalizers,
    dataset_name=dataset_name
)

# 包装后的数据集可以直接使用
sample = wrapped_dataset[0]  # 返回已归一化的样本
```

### 手动包装

如果你知道数据集类型，也可以直接使用对应的包装器：

```python
from data_utils.dataset_wrappers import NormalizedMapDataset, NormalizedIterableDataset

# Map-style 数据集
wrapped_map = NormalizedMapDataset(
    dataset=my_map_dataset,
    action_normalizers=action_normalizers,
    state_normalizers=state_normalizers
)

# Iterable 数据集
wrapped_iterable = NormalizedIterableDataset(
    dataset=my_iterable_dataset,
    action_normalizers=action_normalizers,
    state_normalizers=state_normalizers
)
```

## 📝 支持的数据集类型

### ✅ 原生支持

1. **继承自 `torch.utils.data.Dataset` 的类**
   - 例如：`EpisodicDataset`, `AlohaSimDataset`, `RobomimicDataset` 等

2. **继承自 `torch.utils.data.IterableDataset` 的类**
   - 例如：`WrappedTFDSDataset`, `DroidRLDSDataset` 等

3. **任何实现标准接口的自定义类**（无需继承 PyTorch 基类）
   - Map-style: 实现 `__getitem__` 和 `__len__`
   - Iterable: 实现 `__iter__`

### 示例：自定义数据集类

```python
# 不需要继承任何基类，只要实现正确的方法即可
class MyCustomMapDataset:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 这个自定义类会被自动识别为 map-style 并正确包装
wrapped = wrap_dataset_with_normalizers(
    dataset=MyCustomMapDataset(data),
    action_normalizers=normalizers,
    state_normalizers=normalizers
)
```

## 🔄 与现有代码的集成

在 `data_utils/utils.py` 中，原来的代码：

```python
# ❌ 旧方式：需要数据集实现特定方法
for dataset in datasets:
    dataset.set_action_normalizers(action_normalizers)
    dataset.set_state_normalizers(state_normalizers)
```

已经被替换为：

```python
# ✅ 新方式：使用包装器，无需数据集实现特定方法
from data_utils.dataset_wrappers import wrap_dataset_with_normalizers

wrapped_datasets = []
for dataset in datasets:
    dataset_name = dataset.get_dataset_dir() if hasattr(dataset, 'get_dataset_dir') else None
    wrapped_dataset = wrap_dataset_with_normalizers(
        dataset=dataset,
        action_normalizers=action_normalizers,
        state_normalizers=state_normalizers,
        dataset_name=dataset_name
    )
    wrapped_datasets.append(wrapped_dataset)
```

## 🎯 关键特性

### 1. 属性和方法透明转发

包装器会将所有未知的属性和方法调用转发给原始数据集：

```python
# 可以直接调用原始数据集的方法
wrapped_dataset.get_dataset_dir()  # 转发到 dataset.get_dataset_dir()
wrapped_dataset.initialize()        # 转发到 dataset.initialize()
wrapped_dataset.custom_method()     # 转发到 dataset.custom_method()
```

### 2. 自动数据集名称推断

包装器会尝试自动推断数据集名称来查找对应的 normalizer：

```python
# 尝试按优先级获取数据集名称
if hasattr(dataset, 'dataset_dir'):
    dataset_name = dataset.dataset_dir
elif hasattr(dataset, 'dataset_path_list'):
    dataset_name = dataset.dataset_path_list[0]
elif hasattr(dataset, 'get_dataset_dir'):
    dataset_name = dataset.get_dataset_dir()
```

### 3. 只归一化需要的字段

只有当 `action` 或 `state` 字段存在时才会应用归一化：

```python
# 只归一化存在的字段
if self.action_normalizer is not None and 'action' in sample:
    sample['action'] = self.action_normalizer.normalize(sample['action'])

if self.state_normalizer is not None and 'state' in sample:
    sample['state'] = self.state_normalizer.normalize(sample['state'])
```

## 🧪 测试

运行测试以验证包装器功能：

```bash
cd /home/wz/project/IL-Studio
python -m data_utils.test_dataset_wrappers
```

测试涵盖：
- ✅ Map-style 数据集包装
- ✅ Iterable 数据集包装
- ✅ 自动类型检测
- ✅ 鸭子类型对自定义类的支持
- ✅ 属性转发
- ✅ 归一化正确性

## 📚 相关文件

- **`data_utils/dataset_wrappers.py`** - 包装器实现
- **`data_utils/test_dataset_wrappers.py`** - 测试代码
- **`data_utils/utils.py`** - 在数据加载流程中使用包装器

## 🎓 最佳实践

1. **新数据集开发**: 只需实现 `__getitem__`+`__len__` 或 `__iter__`，无需关心归一化逻辑
2. **已有数据集**: 无需修改，包装器自动处理
3. **自定义归一化**: 如果需要特殊的归一化逻辑，可以在数据集内部处理，包装器只会添加额外的归一化层
4. **调试**: 包装器完全透明，可以直接访问原始数据集的所有方法和属性

## 🔮 未来扩展

包装器设计允许轻松添加更多功能：
- 数据增强包装器
- 缓存包装器
- 采样权重包装器
- 多模态数据处理包装器

每个包装器都可以独立工作，也可以链式组合使用。

