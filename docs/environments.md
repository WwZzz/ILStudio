# 环境安装：先选功能，再复制命令

安装入口只需要 Python 3.10+：

```bash
python -m pip install -e .
python -c "import vlastudio"
```

顶层导入只使用 Python 标准库，不导入 Torch、模型、数据集或仿真器。
读取 YAML 和运行环境管理命令时才需要安装包声明的轻量依赖。
安装 VLAStudio 不会自动安装训练环境。

## 不知道该装什么

先查看功能清单和安装方案：

```bash
vlastudio env catalog
vlastudio env plan --policy act
vlastudio env plan --policy smolvla --dataset libero.libero_object --env libero.object
vlastudio env plan --dataset sim_transfer_cube_scripted
vlastudio env check --policy act
```

`plan` 显示当前解释器、缺失包、版本不匹配、Python/平台要求，以及可复制的
create、run、install 命令。`plan` 和 `catalog` 不安装、不登记环境、不创建缓存；
`check` 执行相同检查，有问题时退出码为 2，否则为 0。`--json` 输出结构化结果。
检查读取已安装发行包的元数据，不导入模型库。它不验证 CUDA、动态库、模型权重、
仿真资源、extras 或 Git 源码版本；包版本齐全仍可能需要组件的系统准备步骤。

推荐创建独立环境，再在其中执行脚本：

```bash
# Torch base；没有仿真器
vlastudio env create -n base
# ACT 训练并在 ALOHA 评测
vlastudio env create --policy act --env aloha_sim -n act_aloha
vlastudio env run -n act_aloha -- python my_training_script.py
# SmolVLA 与 LIBERO，同一进程使用的组件一起选择
vlastudio env create --policy smolvla --env libero.object -n smol_libero
vlastudio env run -n smol_libero -- python my_smolvla_script.py
# 只做共享数据加载、变换和缓存
vlastudio env create --dataset -n data
vlastudio env run -n data -- python inspect_data.py
```

`--dataset` 接受 task YAML 的名称或路径；不跟值时选择默认共享数据配置。
策略选择器接受名称或 YAML 路径，`dp` 对应 `diffusion_policy`，
`openpi` 对应 `pi0`。`aloha_sim` 对应 `aloha_transfer`。
不兼容的 Python/依赖请分开创建环境，通过 policy server 通信。

## 依赖分层

清单位于 `src/vlastudio/environments/`：

| 层 | 内容 | 选择方式 |
| --- | --- | --- |
| 轻量入口 | 配置、注册、环境管理 | 安装 vlastudio |
| data | HDF5、LeRobot parquet/video、图像、变换、缓存；Torch/NumPy 使用兼容范围 | `--dataset <task>` |
| base | 继承 data，固定 Torch 2.4 / torchvision 0.19 及通用训练依赖 | 无选择器，或 `--policy act/mlp` |
| 模型 | 继承共享层并声明模型差异 | `--policy smolvla/pi0/openvla/fastwam/dp` |
| 数据格式扩展 | RLDS/DROID 的 TensorFlow/dlimp；原生 LeRobot adapter 的 lerobot | 从 task 中的数据集类型或 runtime 选择 |
| 仿真 | ALOHA、LIBERO、MetaWorld 等独立依赖 | `--env <config>` |

base 不安装 LIBERO、MuJoCo、MetaWorld 等仿真器。DP 在 base 上增加 diffusers
和 robomimic；SmolVLA/FastWAM 改用 Torch 2.7.1；OpenPI 默认 Python 3.11，
并保留 transformers overlay 与 glibc 要求。

“基于 base 新建”通过清单继承实现：每个新环境重新解析完整依赖，并复用 uv 的下载缓存。
它不复制当前 base 中手工安装的包，也不通过 system-site-packages 共享可变环境。
因此升级某个模型环境不会更改 base。

其他历史模型仍可从组件自身的 `pyproject.toml` 生成安装方案；
输出会标记为 legacy 清单，不能把这些旧依赖视为已验证的分层环境。
FastWAM 的包清单只覆盖 Python 依赖；上游源码、权重、LIBERO 资源以及训练专用依赖
仍按 [example08](../examples/08_fastwam_libero/README.md) 的 setup 步骤准备。

## Python 加载入口的提示

```python
import vlastudio as vla             # 不要求 Torch 或仿真环境
policy = vla.load_policy("smolvla")  # 缺失/不兼容时抛出 vla.DependencyError
dataset = vla.load_dataset("sim_transfer_cube_scripted")
bench = vla.load_env("aloha_transfer")
```

错误包含所选配置对应的安装、创建和运行命令，且保留结构化的 `error.plan`。
这些函数仍返回配置句柄，不立即加载模型权重或数据，也不自动安装包。

只编排任务、准备托管任务时，可以显式跳过当前进程检查：

```python
dataset = vla.load_dataset("sim_transfer_cube_scripted", check_dependencies=False)
policy = vla.load_policy("act", runtime="managed")
# 或 vla.load_policy("act", check_dependencies=False)
vla.train(policy, dataset, output_dir="checkpoints/act")
```

`load_policy/load_env(runtime="managed")` 不检查调用者的 ML 环境；任务执行时准备独立环境。
`load_dataset(check_dependencies=False)` 用于相同的编排场景。
API 默认使用当前解释器；与以前的行为相比，load_* 默认增加了依赖预检，
仅创建配置句柄的旧代码应显式关闭检查。

也可直接获取只读方案：

```python
plan = vla.environment_plan(policy="smolvla", environment="libero.object")
print(plan["commands"]["create"])
```

## 自定义组件的共享清单

自定义 YAML 中声明 `runtime: /path/my-runtime.yaml`，或环境命令使用
`--runtime-manifest /path/my-runtime.yaml`。完整 manifest 与组件选择器二选一。

```yaml
# my-runtime.yaml；extends 相对当前 manifest 解析
extends: /path/VLAStudio/src/vlastudio/environments/base.yaml
python: '3.10'
requirements:
  - transformers==4.51.3
  - my-policy-package>=1
notes:
  - Prepare the model weights separately.
```

`extends` 支持一个路径或路径列表，可多层继承，会检测循环。子清单明确写出的同名
依赖覆盖父层约束；同时选择 policy、dataset、env 属于并列组合，保留各自的约束，
安装工具会报告冲突。Python 版本不同会直接报错。`requires_python` 可声明允许的
Python 范围，`python` 指定创建环境默认使用的版本。
完整 lockfile 不参与继承；请提供最终完整 lockfile，包含所有轻量运行依赖。
原有 `entrypoint`、`overrides`、`overlays`、`platforms`、`min_glibc` 继续支持。

## 当前环境安装、缓存和终端切换

```bash
# 增量安装到当前解释器，可能升级/降级已有包
vlastudio env install --policy act --env aloha_sim
# 已登记的环境可直接安装，无需激活
vlastudio env install -n smol_libero --policy smolvla --env libero.object
vlastudio env list
vlastudio env path --policy act
vlastudio env prepare --policy act --dry-run
```

安装前可使用 `--dry-run` 查看目标解释器及完整清单。`path` 不安装；
`prepare` 准备环境，`run` 准备后执行命令。
显式命名的新环境位于 `$VLASTUDIO_CACHE/envs/named-<名称>`；
不命名时按完整清单哈希复用。重复 create 复用已有环境并保留手动修改，
更新依赖请使用 `env install -n <名称> ...`，或新建另一个名称。

base 首次由启动器环境登记，不代表 Torch 已安装；`env create` 会准备托管 base，
已有用户登记的 base 则复用。`env install` 默认使用当前解释器并登记为 base，
指定已登记名称则安装到该环境，指定新名称则登记当前解释器。
`VLASTUDIO_BASE_PYTHON` 可指定已有 base，VLAStudio 不自动修改该外部环境。

默认缓存根目录为 `~/.cache/vlastudio`，用 `VLASTUDIO_CACHE` 修改。
下载源继承系统和安装工具配置，使用 `-i/--index-url URL` 仅覆盖本次安装：

```bash
vlastudio env create --policy smolvla -i https://pypi.org/simple/
```

`env run` 不需要激活。需要切换当前终端时，先加载 shell 支持：

```bash
eval "$(vlastudio env init --shell bash)"
# zsh: eval "$(vlastudio env init --shell zsh)"
vlastudio env activate smol_libero
vlastudio env deactivate
```

PowerShell：

```powershell
vlastudio env init --shell powershell | Out-String | Invoke-Expression
vlastudio env activate smol_libero
```

初始化后 create 会自动激活创建的环境；activate/deactivate 默认返回 base。
