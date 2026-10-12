# 08 — FastWAM on LIBERO

Linux, Python 3.10 and an NVIDIA GPU with EGL support. From the repository root:

```bash
python -m pip install uv
python -m uv venv --python 3.10 --seed "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/envs/example08"
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/envs/example08/bin/activate"
python -m pip install -e . -r examples/08_fastwam_libero/requirements.txt
python examples/08_fastwam_libero/setup.py
python -m pip install --no-deps -e "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/sources/FastWAM"
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/fastwam/env.sh"
python examples/08_fastwam_libero/evaluate.py
```

The evaluation command uses the official FastWAM Base checkpoint by default. Cache: `VLASTUDIO_CACHE`
(default `~/.cache/vlastudio`). Pass `--index-url URL` to `python -m pip install`
to select a package index. The asset preparation script never installs packages. It accepts `--root` and `--assets` to reuse
the pinned LIBERO source and assets. Evaluation accepts
`--suite spatial|object|goal|10`, `--checkpoint PATH`, `--device cuda:0`,
and `--output-dir PATH`.
Use fresh outputs. Configuration files are in `config/`.

Native dependencies are Git, a C/C++ build toolchain, Python development headers,
ImageMagick Wand, OpenGL, and EGL. The optional [setup.sh](setup.sh) installs the
Ubuntu packages and runs the Python setup for you.

On multi-GPU hosts affected by NVIDIA EGL aborts, keep rendering and inference
on separate GPUs (validated on H100):

```bash
MUJOCO_EGL_DEVICE_ID=0 python examples/08_fastwam_libero/evaluate.py --device cuda:1
```

Both GPUs must be visible to the process.

## Paper comparison

50 initial states per task, 10 tasks per suite (2,000 trials total).
30 settling steps; 32 predicted / 10 executed actions; 10 denoising steps,
scheduler shift 5.0, seed 42. Two 256px cameras rotated 180 degrees, resized
to 224px and concatenated; 400-step limit (700 for Long). The simulator is
reused between trials to preserve the official reset behavior.

| Suite | Paper | Released weights via VLAStudio |
| --- | ---: | ---: |
| Spatial | 98.2% | 96.0% (480/500) |
| Object | 100.0% | 99.4% (497/500) |
| Goal | 97.0% | 96.6% (483/500) |
| Long | 95.2% | 94.0% (470/500) |
| Average | 97.6% | 96.5% (1930/2000) |

Validation uses eager BF16 inference on RTX 4090 48GB with PyTorch 2.7.1+cu126.
Reset images/states and predicted actions were checked against the official code.

Sources: [paper, Table 2](https://arxiv.org/html/2603.16666v1),
[official code](https://github.com/yuantianyuan01/FastWAM),
[released checkpoint](https://huggingface.co/yuanty/fastwam).

## Train through VLAStudio

```bash
# Reuse the environment prepared above
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/envs/example08/bin/activate"
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/fastwam/env.sh"
python -m pip install deepspeed==0.18.7 pyarrow==24.0.0 ninja tensorboard
python examples/08_fastwam_libero/prepare_data.py --text-device cuda
python examples/08_fastwam_libero/prepare_weights.py
python -m torch.distributed.run --standalone --nproc_per_node=4 examples/08_fastwam_libero/train.py
python examples/08_fastwam_libero/evaluate.py --checkpoint checkpoints/fastwam
```

Preparation downloads the pinned official dataset and converts it to standard
VLAStudio episode HDF5, preserving source pixels, actions and states. Set
`--source` / `--output` during preparation and `--data` / `--output-dir` during
training to use custom paths. Allow about 175 GB for the converted images.

The Python example calls `vla.load_dataset`, `vla.load_policy`, and `vla.train`.
Training uses VLAStudio's DataLoader and BaseTrainer, with FastWAM model/loss
and checkpoint hooks. It does not run the upstream trainer. Four 80 GB GPUs,
8 samples/GPU, accumulation 4, global batch 128, 21,700 updates. Training uses
the official Wan video initialization and interpolated, scaled Wan action
backbone; VAE/T5 remain frozen. Missing action initialization fails before training.
The earlier 2026-10-09 run used random experts and is a separate experiment.
Checkpoints retain model weights
only (`save_only_model: true`); change this for optimizer-state resumption.

Standard `image[2,9,3,512,512]` contains offsets 0,4,...,32. The policy processor
produces `video[3,9,224,448]`, `action[32,7]`, and `proprio[32,8]`. Twelve windows
across all four suites, including episode tails, matched the official loader
exactly for video, action, state, text embeddings and padding masks. A 4-H100
VLAStudio smoke run completed three optimizer updates (finite loss 3.5128) and
saved a 6.02B-parameter checkpoint. The complete conversion contains 1,712
episodes / 277,713 frames; action/state min-max statistics match the source.
Both VLAStudio training runs completed 21,700 updates. The table above
describes released weights only; results for our trained checkpoints follow.

HDF5 images are read as contiguous frames rather than a multi-frame fancy
selection. On the same H100 node, a 128-sample CPU DataLoader benchmark
(batch 8, four workers) improved from 55.05 s to 7.70 s; model input tensors
matched exactly. After restarting on another 4-H100 node, updates 30-220
averaged 2.02 s/update (previous run: 13.10 s/update); a 25-second sample
measured 92.7% mean GPU utilization. The end-to-end comparison spans two
nodes; the DataLoader comparison used the same node.


## VLAStudio-trained checkpoints

Completed October 12, 2026: both checkpoints trained through VLAStudio for
21,700 updates and evaluated with the protocol above. Each checkpoint has
500 trials per suite (2,000 total), using MuJoCo 3.3.2, seed 42 and eager BF16
inference on H100, with rendering and inference on separate GPUs.

| Initialization | Spatial | Object | Goal | Long | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wan pretrained | 97.6% (488/500) | 99.2% (496/500) | 97.4% (487/500) | 94.2% (471/500) | **97.1% (1942/2000)** |
| Random experts | 80.0% (400/500) | 97.8% (489/500) | 87.2% (436/500) | 63.6% (318/500) | **82.15% (1643/2000)** |

All 4,000 unique trials passed the checkpoint, seed and per-task summary audit;
only complete 50-trial tasks were retained across restarts. Incomplete and
diagnostic runs are excluded. Random initialization applies to the video/action
experts; VAE/T5 remain pretrained and frozen.

Wan initialization is 0.5 percentage points below the paper's 97.6%, 0.6 points
above our earlier released-weight evaluation (96.5%), and 14.95 points above
random initialization. The released weights were evaluated on RTX 4090 and
our trained checkpoints on H100; these are single-seed results.
