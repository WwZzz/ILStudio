# Normalizer Mask功能说明

## ✅ 功能已完成

Normalizer现在完全支持mask功能，确保训练和推理时的归一化行为一致。

## 快速使用

### 1. 训练时配置

在 `configs/task/xxx.yaml` 中：

```yaml
datasets:
  - name: "my_robot_dataset"
    class: "AlohaSimDataset"
    args:
      dataset_path_list: ['data/robot']
      camera_names: ['primary']
    # mask配置与args同级
    action_norm_mask: [-1]  # 最后一维（gripper）不归一化
    state_norm_mask: [-1]
```

### 2. 自动保存

训练时会自动保存到 `{output_dir}/normalize.json`：

```json
{
  "datasets": [
    {
      "dataset_id": "my_robot_dataset",
      "action_norm_mask": [-1],
      "state_norm_mask": [-1]
    }
  ]
}
```

### 3. 自动加载

推理时自动从 `normalize.json` 加载mask配置，无需额外操作。

## Mask格式

```yaml
# 索引数组（推荐）- 指定不归一化的维度
action_norm_mask: [-1]        # 最后一维
action_norm_mask: [6, 13]     # 第7和第14维
action_norm_mask: [-2, -1]    # 最后两维

# Boolean数组 - 显式指定每个维度
action_norm_mask: [true, true, true, false]  # 最后一维不归一化

# 不配置 - 所有维度都归一化（默认）
```

## 实现细节

### 已修改的文件

1. **data_utils/normalize.py**
   - 移除了 `gripper_indices` 参数
   - 添加了 `mask` 参数支持
   - 实现了 `_build_mask()` 和 `_apply_mask()` 方法

2. **data_utils/utils.py**
   - 训练时从config读取mask并传递给normalizer
   - 保存mask到 `normalize.json`
   - 推理时从 `normalize.json` 加载mask
   - 添加了透明的日志输出

3. **configs/task/example_with_norm_mask.yaml**
   - 完整的使用示例和说明

### 日志输出

**训练时**：
```
Creating normalizers with mask configuration for dataset 'my_robot_dataset':
  - action_norm_mask: [-1]
  - state_norm_mask: [-1]

Saving normalizer metadata with mask configurations to: ./ckpt/model/normalize.json
  Dataset 'my_robot_dataset':
    - action_norm_mask: [-1]
```

**推理时**：
```
Loading normalizers with mask configuration for dataset 'my_robot_dataset':
  - action_norm_mask: [-1]
  - state_norm_mask: [-1]
```

## 验证

查看保存的配置：
```bash
cat ckpt/your_model/normalize.json
```

查看训练/推理日志：
```bash
grep "mask" train.log
python eval.py --model_name_or_path ckpt/your_model 2>&1 | grep "mask"
```

## 向后兼容

- 不配置mask时，保持原有行为（全部归一化）
- 现有的normalize.json文件仍然可以正常工作
- 已删除未使用的 `gripper_indices` 参数

## 核心优势

✅ **配置即生效** - 只需在config中配置，无需修改代码  
✅ **自动保存加载** - 训练推理自动同步  
✅ **透明日志** - 清晰显示mask配置状态  
✅ **灵活配置** - 支持多种格式和每个数据集独立配置  
✅ **完全一致** - 确保训练和推理使用相同的mask

