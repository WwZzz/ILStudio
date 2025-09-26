# Installation
```shell
# Linux only or mac
pip install gymnasium-robotics
```

# TroubleShooting
for error `mujoco.FatalError: an OpenGL platform library has not been loaded into this process, this most likely means that a valid OpenGL context has not been created before mjr_makeContext was called` on Linux, please run
```shell
export MUJOCO_GL=egl
```