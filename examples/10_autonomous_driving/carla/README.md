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

The default run performs 100 fine-tuning updates. `config/task.yaml` selects ten
scenario-diverse training routes. Set
`max_trajectories: null` to use all 1,100 official training routes (a large download).
`train.py --steps 100` limits a development run. `config/val.yaml` evaluates all
220 official validation routes; `--max-routes 1` is a small interface check only.
Evaluation resumes from its completed-route journal with the same command.

Observation: one RGB panorama `(1,3,256,900)`, `state=[speed_mps]`, and `raw_lang`
containing both navigation intent and the local route waypoint. TCP's data
processor extracts the waypoint and six command indicators internally; a language
policy can read the complete task text directly. Action: `[steer, throttle, brake, reverse]`,
with ranges `[-1,1], [0,1], [0,1], {0,1}`. The default `merge_ctrl_traj` policy
uses the authors' 50/50 fusion of waypoint PID and learned control predictions.

## What is in a training sample?

The public data combines `anno/*.json.gz` (vehicle measurements and navigation),
three front-camera JPG streams, and `expert_assessment/*.npz` (expert actions,
features and value). One dataset item is one current frame with future labels
from that same route. Example values below are illustrative:

```python
{
    "image": rgb,                    # uint8 (1,3,256,900), left/front/right panorama
    "state": [3.0],                  # current speed, m/s
    "raw_lang": ("turn left\n"
        "Local route waypoint in meters (ego frame; x=forward, y=right): [12.0, -4.0]"),
    "action": [[-0.3, 0.3, 0, 0]],    # expert steer/throttle/brake/reverse
    "is_pad": [False],
    # Training-only: waypoints (4,2), action_index, future_action_index (4,),
    # expert_features (1536,), expert_value, future_features (4,1536).
    "episode_id": "Scenario_TownXX_RouteID_WeatherID", "frame_id": "00042",
}
```

The route target describes where navigation wants the vehicle to go; it is not
the final destination or an expert future position. Here the text means 12 m
forward and 4 m left. The coordinates use lossless float32-to-JSON conversion;
the text declares units and axes and is refreshed every observation.
Future waypoint/action labels
use frame offsets 5/10/15/20 and never cross episode boundaries. The live environment
returns current image/state/complete task text and timestep, without expert labels.
TCP's data processor builds a private `tcp_state` nine-value vector from the text
and speed. The public dataset state remains one-dimensional. Arbitrary free-form
instructions require a language-capable policy; TCP accepts the six declared
phrases plus the documented waypoint text template.

[Seven-framework comparison, complete fields and coordinate conventions](../../../docs/driving_data_contract.md).

The ordinary environment API also works with a saved checkpoint:

```python
import vlastudio as vla

policy = vla.load_policy('tcp_b2d', checkpoint='checkpoints/example10/tcp')
env = vla.load_env('carla.bench2drive')
result = env.evaluate(policy, output_dir='results/example10/api', num_rollout=1, batch_size=0)
```

[Installation details, sources and metric definitions](../../../docs/driving.md)
