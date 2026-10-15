# CARLA / Bench2Drive

Continue training TCP with VLAStudio's standard dataset and training interfaces,
then evaluate vehicle controls in the official Bench2Drive scenarios.

Install once (skip this when reusing an environment with these components):

```bash
vlastudio env create driving --policy tcp_b2d --env carla.bench2drive --system-packages
```

Activate the resulting environment. CARLA and its additional maps are installed
by the environment recipe; the host must provide working NVIDIA Vulkan graphics.
Training data and configured pretrained weights download automatically on first use.

```bash
python examples/10_autonomous_driving/carla/train.py
python examples/10_autonomous_driving/carla/evaluate.py --model checkpoints/example10/tcp --output results/example10/tcp
```

`config/task.yaml` selects ten scenario-diverse training routes. Set
`max_trajectories: null` to use all 1,100 official training routes (a large download).
`train.py --steps 100` limits a development run. `config/val.yaml` evaluates all
220 official validation routes; `--max-routes 1` is a small interface check only.
Evaluation resumes from its completed-route journal with the same command.

Observation: one RGB panorama `(1,3,256,900)`, speed in m/s, ego-frame next route
point and six route-command indicators. Action: `[steer, throttle, brake, reverse]`,
with ranges `[-1,1], [0,1], [0,1], {0,1}`. The default policy predicts waypoints
and uses the authors' PID controller to produce these controls.

The ordinary environment API also works with a saved checkpoint:

```python
import vlastudio as vla

policy = vla.load_policy('tcp_b2d', checkpoint='checkpoints/example10/tcp')
env = vla.load_env('carla.bench2drive')
result = env.evaluate(policy, output_dir='results/example10/api', num_rollout=1, batch_size=0)
```

[Installation details, sources and metric definitions](../../../docs/driving.md)
