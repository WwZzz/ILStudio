# 08 — FastWAM on LIBERO

Linux, Python 3.10 and an NVIDIA GPU with EGL support. From the repository root:

```bash
bash examples/08_fastwam_libero/setup.sh
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/envs/example08/activate.sh"
python examples/08_fastwam_libero/evaluate.py
```

The evaluation command uses the official FastWAM Base checkpoint by default. Cache: `VLASTUDIO_CACHE`
(default `~/.cache/vlastudio`). Setup accepts `-i URL`. Evaluation accepts
`--suite spatial|object|goal|10`, `--checkpoint PATH`, `--device cuda:0`,
and `--output-dir PATH`.
Use fresh outputs. Configuration files are in `config/`.

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
bash examples/08_fastwam_libero/setup.sh --training
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/envs/example08/activate.sh"
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

Audited results as of October 12, 2026: 500 trials per completed suite,
MuJoCo 3.3.2, seed 42, eager BF16 inference on H100.

| Initialization | Spatial | Object | Goal | Long |
| --- | ---: | ---: | ---: | ---: |
| Wan pretrained | 97.6% (488/500) | 99.2% (496/500) | Pending | Pending |
| Random experts | 80.0% (400/500) | 97.8% (489/500) | Pending | Pending |

Goal and Long evaluation is still running. No four-suite average is claimed yet.
