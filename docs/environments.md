# 作为 Python 库使用：安装环境与编写 run.py

VLAStudio 的主要用法是 `import vlastudio as vla`，在自己的 Python 文件里组合
数据、模型、训练、评估和设备操作，然后运行 `python run.py`。
工作脚本使用当前 Python 环境。环境准备是独立步骤，导入库和调用加载函数不会自动安装包。

## 1. 创建普通虚拟环境，安装库

在仓库根目录执行，以 ACT 的 Python 3.10 环境为例：

```bash
python -m pip install uv
python -m uv venv --python 3.10 --seed .venv
source .venv/bin/activate
python -m pip install -e .
```

Windows 激活方式为 `.venv\Scripts\Activate.ps1`。已有合适的 Python 时，也可以
使用 `python -m venv .venv`。不需要可编辑安装时使用 `python -m pip install .`。
未发布到 PyPI 的开发版本可以通过仓库的 Git 地址安装，见首页。

uv 负责选择 Python 和安装依赖；VLAStudio 作为库装在这个环境中。
安装包只包含轻量运行依赖，不会一起安装 Torch、模型、仿真器或数据集。
即使没有这些依赖，`import vlastudio` 也只使用 Python 标准库。

## 2. 写一个独立的 setup_environment.py

```python
import vlastudio as vla

vla.install_dependencies(policy="act", environment="aloha_sim")
```

```bash
python setup_environment.py
```

这个函数使用 uv 增量安装到运行该文件的 `sys.executable`，不会切换解释器、
创建另一个环境或修改其他虚拟环境。它可能升级或降级当前环境已有的包，
因此不同模型应分别使用自己的虚拟环境。它会检查 Python 和平台要求，
并应用模型清单中声明的必要补丁。权重、数据、设备 SDK 和仿真资源按各组件说明另外准备。

不传选择器时，`vla.install_dependencies()` 安装 Torch 2.4 base。
其他选择器示例（每次选择自己需要的组合）：

```python
import vlastudio as vla

# 只做数据加载、变换和缓存
vla.install_dependencies(dataset="sim_transfer_cube_scripted")

# SmolVLA 模型环境；覆盖 base 的 Torch/Transformers 版本
# vla.install_dependencies(policy="smolvla")

# 同一个脚本中训练并在 LIBERO 评估
# vla.install_dependencies(policy="smolvla", environment="libero.object")

# 独立仿真客户端，不安装远端模型的依赖
# vla.install_dependencies(remote=True, environment="metaworld.easy")

# 自定义完整环境清单
# vla.install_dependencies(manifest="my-runtime.yaml")
```

源码示例提供了可编辑的安装脚本，内部调用同一个 Python API：

```bash
python examples/install_dependencies.py --policy act --env aloha_sim
```

也可以完全自己编写 setup 文件，不需要使用这个辅助脚本。

## 3. 写自己的 run.py

```python
import vlastudio as vla


def main():
    dataset = vla.load_dataset("sim_transfer_cube_scripted")
    policy = vla.load_policy("act")
    vla.train(policy, dataset, "default", output_dir="checkpoints/my_act")
    benchmark = vla.load_env("aloha_transfer")
    result = benchmark.evaluate(policy, output_dir="results/my_act")
    print(result.metrics)


if __name__ == "__main__":
    main()
```

```bash
python run.py
```

数据路径、训练参数、策略服务、采集逻辑都可以在自己的文件里定义。
使用内置配置名或自己的 YAML 路径即可。相对路径按当前工作目录解析。
依赖缺失或版本不匹配时，`load_policy`、`load_dataset`、`load_env` 会提示
需要的包和 setup 代码。`DependencyError` 继承 `ImportError`，也可读取其 `plan` 字段。

## 查看环境需求，不执行安装

```python
import vlastudio as vla

plan = vla.environment_plan(policy="smolvla", environment="libero.object")
print(plan["required_python"])
print(plan["profile"]["requirements"])
print(plan["issues"])

# 也可预览一次安装调用；不会创建缓存或修改环境
preview = vla.install_dependencies(policy="act", dry_run=True)
```

检查只读取已安装包的元数据，不导入模型库，也不验证 CUDA、动态库、权重、
仿真资源、extras 或 Git 源码版本。包版本齐全不代表系统资源已准备完成。

## 共享层与模型层

清单位于 `src/vlastudio/environments/`，依赖关系由 `extends` 声明：

| 层或组件 | Python | 主要内容 |
| --- | --- | --- |
| data | 默认 3.10，允许 3.10–3.12 | HDF5、parquet、视频、图像与通用数据处理 |
| base / ACT / MLP | 3.10 | data + Torch 2.4.0 / torchvision 0.19.0 / Transformers 4.45.2 |
| Diffusion Policy | 3.10 | base + 扩散策略依赖 |
| SmolVLA | 3.10 | base 的版本覆盖：Torch 2.7.1、LeRobot 0.3.3；Linux glibc ≥ 2.31 |
| OpenPI / Pi0 / Pi0.5 | 3.11 | OpenPI 固定版本、Torch 2.7.1 和 Transformers 补丁；Linux glibc ≥ 2.31 |
| FastWAM | 3.10 / Linux | Torch 2.7.1、视频模型依赖；另行准备源码和权重 |
| simulator | 按组件 | ALOHA、LIBERO、MetaWorld、Tabletop-Sim 等独立仿真依赖 |

子清单明确声明的同名包覆盖父层版本。并列选择的模型、数据集和仿真器需要
同时满足各自约束；冲突时使用不同虚拟环境，通过 `vla.serve` 和 `vla.connect_policy` 通信。
新环境继承的是共享配置，不会复制另一个虚拟环境中的文件。

## 自定义清单、下载源与缓存

```yaml
# my-runtime.yaml；extends 相对当前清单文件解析
extends: /path/VLAStudio/src/vlastudio/environments/base.yaml
python: '3.10'
requirements:
  - transformers==4.51.3
  - my-policy-package>=1
```

```python
import vlastudio as vla

vla.install_dependencies(
    manifest="my-runtime.yaml",
    index_url="https://pypi.org/simple/",
    cache_dir="/scratch/vlastudio",
    offline=False,
)
```

`extends` 支持路径列表和多层继承，会检测循环。完整 lockfile 不能参与继承，
应提供包含轻量运行依赖的最终完整锁定文件。`requires_python` 可声明允许的 Python 范围；
`overrides`、`overlays`、平台和 glibc 限制继续适用。
Python packages 的 CUDA 构建选择使用安装工具的索引配置；GPU 驱动仍由系统管理。

默认缓存为 `~/.cache/vlastudio`，可通过 `VLASTUDIO_CACHE` 或 `cache_dir` 修改。
数据和模型缓存可分别通过工作流 API 的 `data_cache_dir`、`model_cache_dir` 配置。
离线模式要求依赖包和源码已在缓存中，模型权重和仿真资源也要提前准备。

使用根目录 `uv sync` 后又增加模型包时，直接运行当前环境的 `python run.py`，
或使用 `python -m uv run --no-sync python run.py`，保留额外安装的包。
