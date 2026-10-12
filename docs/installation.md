# 分层安装与可选依赖

VLAStudio 把“能导入框架”和“能运行某个数据集/模型”分开。不要在一个环境里
安装所有模型：部分模型要求不同的 PyTorch、Transformers、JAX 或 TensorFlow
版本，强行合并容易产生不可解的冲突。

## 1. 轻量核心

```bash
python -m pip install -e .
python -c "import vlastudio; print(vlastudio.__version__)"
```

核心只包含配置解析、扩展注册、环境管理和公开 API；导入 `vlastudio` 不会导入
PyTorch、Transformers、仿真器或模型代码。

## 2. 共用数据层与 PyTorch 基础层

| 层 | 安装命令 | 用途 |
| --- | --- | --- |
| 数据 | `python -m pip install -e ".[data]"` | NumPy、HDF5、图像/视频 IO、SciPy |
| PyTorch | `python -m pip install -e ".[torch]"` | 统一的 Torch 2.4 基础版本 |
| 常用基础 | `python -m pip install -e ".[base]"` | 同时安装 data + torch |

发布包安装时去掉 `-e .`，例如 `python -m pip install "vlastudio[base]"`。
GPU 对应的 CUDA wheel 可按 PyTorch 官方索引安装；VLAStudio 的 `torch` extra
固定经过项目验证的 Python 包版本，不安装显卡驱动或 CUDA 系统运行库。

## 3. 模型按需安装

每个模型 extra 已包含它所需的数据/PyTorch 基础，不必先装 `base`：

```bash
python -m pip install -e ".[act]"
python -m pip install -e ".[diffusion]"
python -m pip install -e ".[smolvla]"
python -m pip install -e ".[openpi]"
python -m pip install -e ".[fastwam]"
```

其他内置族包括 `qwen`、`openvla`、`xvla`、`roboflamingo`、`groot` 和 `octo`。
仿真器依赖仍独立安装，例如 ALOHA 使用 `.[act,aloha]`。不同模型要求冲突时，
为每个模型创建单独虚拟环境，或使用 `vlastudio env create --policy <name>`。

FastWAM 需要 Linux、NVIDIA GPU/EGL、LIBERO 资源以及模型权重。`.[fastwam]`
安装可锁定的 Python 依赖；固定版本上游源码和完整资源由下列命令准备（上游包的
依赖元数据与项目验证栈冲突，因此 setup 会用 `--no-deps` 安装它）：

```bash
bash examples/08_fastwam_libero/setup.sh
```

## 4. 缺失依赖提示

`load_dataset` 和内置模型的 `load_policy` 在导入重型实现之前做预检：

```python
import vlastudio as vla

dataset = vla.load_dataset("sim_transfer_cube_scripted")
policy = vla.load_policy("act")
```

若当前解释器缺依赖，会抛出 `vlastudio.OptionalDependencyError`，列出缺失的
import 名称，并给出可复制的发布包与源码安装命令。例如 ACT 会提示安装
`vlastudio[act]`，数据集会提示 `vlastudio[data,torch]`，FastWAM 会提示
`vlastudio[fastwam]`。

自定义 policy 类型的依赖由插件自身声明，因此 VLAStudio 不猜测其安装命令。
如果脚本只创建句柄，后续交给 VLAStudio 托管环境执行，可显式跳过当前解释器预检：

```python
policy = vla.load_policy("smolvla", check_dependencies=False)
dataset = vla.load_dataset("task.yaml", check_dependencies=False)
```

这只跳过预检，不会隐藏 worker 中真实的导入错误。

## 5. 推荐工作流

日常数据开发使用一个 `.[base]` 环境；模型开发从干净虚拟环境开始，只安装对应
模型 extra。CI 至少保留两项检查：核心安装后 `import vlastudio`，以及模拟缺失依赖
时的安装命令契约。完整 GPU、数据和仿真验证另行在具备资源的机器上运行。
