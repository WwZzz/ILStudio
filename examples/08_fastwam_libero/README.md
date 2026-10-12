# 08 — FastWAM on LIBERO

Run from the repository root on Linux with Python 3.10 and NVIDIA GPUs.

## Setup

```bash
bash examples/08_fastwam_libero/setup.sh --training
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/envs/example08/activate.sh"
```

## Evaluate released weights

```bash
python examples/08_fastwam_libero/evaluate.py
```

## Train and evaluate (4 GPUs)

```bash
python examples/08_fastwam_libero/prepare_data.py --text-device cuda
python examples/08_fastwam_libero/prepare_weights.py
python -m torch.distributed.run --standalone --nproc_per_node=4 examples/08_fastwam_libero/train.py
MUJOCO_EGL_DEVICE_ID=0 python examples/08_fastwam_libero/evaluate.py --checkpoint checkpoints/fastwam --device cuda:1 --output-dir results/example08_fastwam_trained
```

## Results

Both VLAStudio checkpoints were trained for 21,700 updates and evaluated over 500 trials per suite (MuJoCo 3.3.2, seed 42).

| Model | Spatial | Object | Goal | Long | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Paper](https://arxiv.org/html/2603.16666v1) | 98.2% | 100.0% | 97.0% | 95.2% | 97.6% |
| Released weights via VLAStudio | 96.0% | 99.4% | 96.6% | 94.0% | 96.5% |
| VLAStudio — Wan initialization | 97.6% | 99.2% | 97.4% | 94.2% | **97.1%** |
| VLAStudio — random expert initialization | 80.0% | 97.8% | 87.2% | 63.6% | **82.15%** |
