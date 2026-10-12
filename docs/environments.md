# Python environments

VLAStudio is used from your own Python files, launched with `python run.py`.
Dependencies are installed externally with pip or uv. The library never creates
Python environments, installs packages, or repairs packages at runtime.

## Default: ACT, Diffusion Policy and ALOHA-Sim

From a checkout of main:

```bash
python -m pip install uv
python -m uv venv --python 3.10 --seed .venv
source .venv/bin/activate
python -m pip install -e .
python run.py
```

This installs the shared Torch/data stack, ACT, DP, and ALOHA-Sim dependencies.
Training needs demonstrations, and the ResNet backbones may download weights.
GPU drivers and native OpenGL/EGL/OSMesa libraries belong to the host system.
On Windows activate `.venv\Scripts\Activate.ps1` instead.

`import vlastudio` uses only standard-library modules, even when optional model
packages are absent. The default dependency metadata installs the useful base;
Python imports remain lazy.

## Optional models and simulators

Create a separate environment, then install the project and selected requirements
in the same command. From the repository root, for example:

```bash
deactivate
python -m uv venv --python 3.10 --seed .venv-smolvla
source .venv-smolvla/bin/activate
python -m pip install -e . -r requirements/smolvla.txt -r requirements/libero.txt
```

| Component | Python | External requirements |
| --- | --- | --- |
| ACT / DP / ALOHA-Sim | 3.10 recommended | Included in `pip install -e .` |
| SmolVLA | 3.10 | `requirements/smolvla.txt` |
| FastWAM | 3.10, Linux | `requirements/fastwam.txt`; Example 08 also supplies `requirements.txt` |
| OpenPI / Pi0.5 | 3.11, Linux | `requirements/openpi.txt` and its uv overrides; see Example 05 |
| OpenVLA | 3.10, Linux | `requirements/openvla.txt` |
| LIBERO | 3.10 | `requirements/libero.txt`, plus source checkout/assets |
| MetaWorld | 3.10 | `requirements/metaworld.txt` |
| Tabletop-Sim | 3.10 | `requirements/tabletop_sim.txt`, plus source/submodules/assets |
| RLDS loaders | 3.10 or 3.11 | `requirements/rlds.txt` |

LeRobot 0.3.3 needs glibc >= 2.31 on Linux. OpenPI also requires Linux/glibc >= 2.31.
The requirements files contain inherited shared data/model requirements and
component-specific pins. Models can choose their compatible versions within the
default package's base ranges. Simulator packages are included in the default
installation; other environments can select compatible simulator versions.

For Example 07/08 use the example's `requirements.txt`, which preserves its tested
MuJoCo version, rather than adding the generic LIBERO file. Their `setup.py` files
prepare sources, model weights and assets only. Installation commands stay in the
terminal (or the optional shell scripts).

For OpenPI, install uv inside the activated Python 3.11 environment, then:

```bash
python -m pip install uv
python -m uv pip install -e . -r requirements/openpi.txt --override requirements/openpi-overrides.txt
cp -r .venv-pi05/lib/python3.11/site-packages/openpi/models_pytorch/transformers_replace/. .venv-pi05/lib/python3.11/site-packages/transformers/
```

Adjust `.venv-pi05` if you chose another environment directory. The external copy
applies the patch required by the pinned OpenPI source. See [Example 05](../examples/05_finetune_vla_pi05).

## Inspect dependencies without changing anything

Loaders validate installed distribution metadata and raise `DependencyError`
with the missing packages, compatible Python version, and a pip command:

```python
import vlastudio as vla

plan = vla.environment_plan(policy="smolvla", environment="libero.object")
print(plan["issues"])
print(plan["commands"]["install_current"])
```

The command installs requirements into your activated interpreter. For complex
models such as OpenPI, follow the requirements/override/patch commands above.
Use `check_dependencies=False` when only inspecting configuration handles; it
does not install missing packages. All training/evaluation workers use the same
Python interpreter as your script. `runtime="managed"` is unsupported.

Profiles in `src/vlastudio/environments` and custom `runtime` YAML mappings are
read-only dependency descriptions. A custom policy can declare its Python version
and requirements there; install those packages with pip before running it.
Checks compare package versions and platform metadata. They do not prove CUDA
availability, asset availability, native rendering health or VCS revision identity.

## uv and CUDA

`python -m uv sync --python 3.10` installs the committed default lock. Install uv
in the active environment first if using `python -m uv` there. For optional model
environments, install their requirements files rather than syncing the default
lock, which would restore default versions/remove extra packages.

Select CUDA-specific Torch wheels with external pip/uv commands appropriate for
your driver. Install matching Torch/torchvision versions within the package's
declared version ranges. Ordinary `python run.py` uses that activated environment.
An alternative package index can be passed directly to pip with `--index-url URL`.
