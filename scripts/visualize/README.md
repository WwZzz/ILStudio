# Episode Data Visualizer

一个基于Web的交互式可视化工具，用于查看和分析IL-Studio收集的HDF5格式episode数据。

## 功能特性

- 📊 **交互式可视化**: 使用滑块逐帧浏览episode数据
- 📷 **图像显示**: 支持多相机视角的实时图像展示
- 🤖 **机器人轨迹**: 3D轨迹可视化和位置数据分析
- 📈 **数据图表**: 实时展示动作、观察和传感器数据
- 📁 **批量处理**: 支持单个文件或整个目录的HDF5文件
- 🔍 **数据探索**: 详细的数据信息和统计展示

## 安装依赖

```bash
# 安装可视化工具所需的依赖
pip install -r scripts/visualize/requirements.txt
```

或者单独安装：

```bash
pip install dash plotly h5py numpy Pillow pandas
```

## 使用方法

### 方法1: 使用启动脚本（推荐）

```bash
# 可视化整个目录的HDF5文件
python scripts/visualize/run_visualizer.py /path/to/your/data/

# 可视化单个HDF5文件
python scripts/visualize/run_visualizer.py /path/to/episode.hdf5

# 自定义端口
python scripts/visualize/run_visualizer.py /path/to/data/ --port 8080

# 列出目录中的所有HDF5文件
python scripts/visualize/run_visualizer.py /path/to/data/ --list-files
```

### 方法2: Windows批处理脚本

```batch
REM 双击运行或在命令行中使用
scripts\visualize\visualize_data.bat C:\path\to\your\data\
```

### 方法3: 直接调用

```bash
python scripts/visualize/episode_visualizer.py /path/to/data/
```

## 界面说明

启动后，在浏览器中打开 `http://127.0.0.1:8050` 查看可视化界面。

### 主要组件

1. **Episode选择器**: 在多个episode文件间切换
2. **帧滑块**: 逐帧浏览episode数据
3. **相机图像区域**: 显示各个相机的实时图像
4. **3D轨迹图**: 机器人末端执行器的3D轨迹
5. **数据图表**: 位置、动作和观察数据的时间序列图
6. **数据信息面板**: 详细的数据统计和当前值

### 支持的数据格式

可视化工具自动识别以下HDF5数据结构：

- **图像数据**: `/observations/images/*` 或包含 `image`/`camera` 关键字的数据集
- **动作数据**: 包含 `action` 关键字的数据集
- **观察数据**: `/observations/*` 路径下的数据集
- **关节位置**: 包含 `qpos`、`joint`、`shoulder`、`elbow`、`wrist`、`gripper` 等关键字
- **末端执行器**: 包含 `eef`、`end_effector`、`tcp`、`pose` 等关键字

## 数据结构示例

工具支持多种HDF5数据结构，例如：

```
episode.hdf5
├── /observations/
│   ├── /images/
│   │   ├── front_camera    # (T, H, W, 3) 前置相机图像
│   │   └── wrist_camera    # (T, H, W, 3) 手腕相机图像
│   ├── shoulder_pan.pos    # (T,) 关节位置
│   ├── shoulder_lift.pos   # (T,) 关节位置
│   └── ...
├── /action                 # (T, N) 动作数据
└── /episode_len           # 该episode的长度
```

## 性能优化

- **大文件处理**: 工具使用内存映射技术处理大型HDF5文件
- **缓存机制**: 自动缓存频繁访问的数据以提升响应速度
- **懒加载**: 只在需要时加载数据，减少内存占用

## 故障排除

### 常见问题

1. **无法打开HDF5文件**
   ```
   解决方案: 检查文件权限和路径是否正确
   ```

2. **图像无法显示**
   ```
   解决方案: 确认图像数据是 (T, H, W, 3) 格式，数值范围在 [0, 255] 或 [0, 1]
   ```

3. **内存不足**
   ```
   解决方案: 使用 --no-debug 参数关闭调试模式，或处理较小的数据集
   ```

4. **端口占用**
   ```
   解决方案: 使用 --port 参数指定其他端口
   ```

### 调试模式

启用详细日志输出：

```bash
python scripts/visualize/run_visualizer.py /path/to/data/ --debug
```

## 技术细节

- **前端**: Dash + Plotly (基于React和D3.js)
- **后端**: Python + HDF5
- **数据处理**: NumPy + Pandas
- **图像处理**: Pillow + Base64编码

## 扩展功能

### 添加自定义数据类型

修改 `episode_visualizer.py` 中的数据提取方法：

```python
def _extract_common_data_patterns(self):
    # 添加自定义数据类型识别逻辑
    for key, data in self.data.items():
        if 'your_custom_data' in key:
            # 处理自定义数据
            pass
```

### 自定义可视化

在回调函数中添加新的图表类型：

```python
@callback(Output('custom-plot', 'figure'), ...)
def update_custom_plot(frame_idx, episode_idx):
    # 创建自定义图表
    pass
```

## 贡献

欢迎提交Issue和Pull Request来改进这个可视化工具。

## 许可证

与IL-Studio项目保持一致。

