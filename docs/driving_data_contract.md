# Driving task inputs and dataset composition

VLAStudio separates **physical state**, **task conditions** and **training targets**.
The driving dataset's `state` describes the vehicle; `raw_lang` describes the task,
including both navigation intent and its local route waypoint. That waypoint is
a target condition, not a measurement of the vehicle or the expert's future
trajectory. Encoding task text into command indicators, numeric target features
or model tokens belongs to the policy's data processor.

## What existing pipelines do

The following comparison inspects seven public implementations on 2026-10-16.
They cover learned driving policies and planning/benchmark frameworks; their
protocols and command vocabularies differ. This is a design comparison, not a
claim that they all implement the schema below or use natural-language inputs.

| Implementation | Observations and ego state | Navigation condition and adaptation | Prediction / execution |
|---|---|---|---|
| [TCP / Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/8a08b07883f10b7d83f6bf5dd475bda91a91c50a/team_code/tcp_b2d_agent.py) | Front cameras; measured speed; GNSS/IMU used by route preprocessing | Supplied route produces local target point and six-way command. Network concatenates speed, target and command indicators | Four waypoints and control classes; waypoint PID and/or learned controls |
| [TransFuser, 2022 agent](https://github.com/autonomousvision/transfuser/blob/2022/team_code_transfuser/submission_agent.py) | RGB/LiDAR, speed and localization preprocessing | Route planner produces `target_point` and `next_command`; agent separately prepares goal coordinates/image and velocity | Predicted waypoints passed to driving control, with agent-level checks |
| [UniAD, planning head](https://github.com/OpenDriveLab/UniAD/blob/main/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py) | BEV features and ego motion/track queries from upstream perception/prediction | Three-way `command` selects a navigation embedding in the planner | Ego trajectory; optional occupancy-based collision optimization |
| [VAD, original nuScenes pipeline](https://github.com/hustvl/VAD/blob/main/projects/mmdet3d_plugin/VAD/VAD.py) | Camera/BEV features; history and ego low-level features | `ego_fut_cmd` identifies the command-conditioned ego trajectory. The inspected test path selects the corresponding predicted mode for evaluation | Vectorized perception/motion and ego trajectories; original evaluation is open loop |
| [LMDrive](https://github.com/opendilab/LMDrive#dataset) | Temporal multi-view/multimodal sensor inputs and driving information | Navigation and notice instructions are actual text inputs; instruction fine-tuning aligns language, vision and driving outputs | Predicted waypoints converted to controls for closed-loop CARLA evaluation |
| [NAVSIM, input dataclasses](https://github.com/autonomousvision/navsim/blob/main/navsim/common/dataclasses.py) | `AgentInput` contains current/past ego statuses, cameras and LiDAR; status includes pose, velocity and acceleration | `driving_command` is retained in `EgoStatus`; individual agents choose its encoding | Agent predicts a trajectory; evaluator computes planning metrics |
| [nuPlan, planner API](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/simulation/planner/abstract_planner.py) | Rolling history of observations/states and traffic-light data | Planner initialization separately supplies roadblock route IDs, mission goal and map API | Planner returns an ego trajectory to the simulation/controller |

**Synthesis for VLAStudio:** all seven pipelines distinguish navigation intent
or route/mission conditions from the vehicle's measured motion, although their
containers and encodings differ. None establishes a universal language schema.
NAVSIM bundles the command into ego status; nuPlan separates mission data.
Following this project's requested contract, we express the complete driving task
as text and keep ego measurements in `state`. The numeric waypoint is serialized
without float32 precision loss and with explicit units/axes, so a numeric policy
can recover it exactly. This is a VLAStudio design choice, not an industry standard.

UniAD/VAD's original nuScenes paths use recorded future trajectories for training
or evaluation, and should not be mistaken for online sensor interfaces. Never
derive a live navigation condition from a held-out future expert trajectory.
For CARLA, the target here is from the supplied route and current localization.

## Standard sample: meanings before encodings

| Field | Meaning | Policy input? |
|---|---|---|
| `image`, optional `depth` / `pc` | Sensor pixels or precomputed image features; `reasoning.image_type` distinguishes RGB/features | Yes |
| `state` | Measured ego variables declared by the task; this preset supplies only `[speed_mps]` | Yes |
| `raw_lang` | Complete task condition: original language where available; for driving, deterministic command text plus exact local route waypoint and coordinate convention | Yes |
| `reasoning.navigation` | Optional simulator protocol/candidate metadata, used by the VLN backends; driving does not require a duplicate target here | Backend-specific |
| `action`, `is_pad` | Demonstrated action and padding mask in the declared control space | Training target |
| `reasoning.supervision` | Future waypoints and expert features/value/actions; additional training supervision | Training targets only |
| `episode_id`, `timestamp`, `reasoning.frame_id` | Alignment, split verification and provenance | Metadata |

Future backends can declare additional task conditions such as route points or a
mission goal, including their frames/units. They must not invent a local target
for data that supplies only language: an absent target is absent, not `[0, 0]`.
The current driving task text is the source of truth for its target. A policy may
tokenize it, rephrase it into a prompt, or extract exact numeric embeddings.

## Bench2Drive example

Illustrative values below show one sample; they are not a measured successful
episode. A sample is one current frame plus same-route future training labels.

```python
sample = {
    "image": uint8_array,       # (1, 3, 256, 900), RGB left/front/right panorama
    "state": [3.0],             # speed, m/s; no navigation one-hot in state
    "raw_lang": ("turn left\n"
        "Local route waypoint in meters (ego frame; x=forward, y=right): [12.0, -4.0]"),
    "action": [[-0.3, 0.3, 0.0, 0.0]],  # steer, throttle, brake, reverse
    "is_pad": [False],
    "reasoning": {
        "image_type": "rgb",
        "frame_id": "00042",
        "supervision": {
            "waypoints": float_array,           # (4, 2), future expert positions
            "action_index": 12,                 # expert class, training label only
            "future_action_index": int_array,   # (4,)
            "expert_features": float_array,     # (1536,)
            "expert_value": 0.5,
            "future_features": float_array,     # (4, 1536)
        },
    },
    "episode_id": "Scenario_TownXX_RouteID_WeatherID",
    "timestamp": 42,
}
```

`[12, -4]` means 12 m forward and 4 m left of the current vehicle. It is the next
navigation waypoint, not necessarily the final destination and not a promise
that the area is clear. The expert future `waypoints` are separate labels.

The source fields are `anno/*.json.gz` (position, heading, speed, route target,
`next_command`), three `camera/rgb_front*/*.jpg` streams, and
`expert_assessment/*.npz` (1536 features, value, action class). Source command IDs
1–6 become `turn left`, `turn right`, `go straight`, `follow the lane`,
`change to the left lane`, `change to the right lane`; the source's negative
sentinel retains the upstream lane-follow fallback. Unknown nonnegative IDs fail.
Future labels use frame offsets 5, 10, 15 and 20 without crossing a route boundary.
Negative warmup frames and terminal observations without expert actions are excluded.

The policy's `TCPDataProcessor`, not the dataset, parses the task text, maps its
canonical command to one-hot, and builds a private `tcp_state` containing
`[speed, target_x, target_y, command_6]`. It validates the frame/axes/unit template
and target dimensions/finiteness. Coordinates are JSON numbers produced from
float32 values without decimal truncation. The public `state_dim` is 1; the
pretrained TCP network still receives 9 values and its weight shapes are unchanged.
It also adapts `reasoning.supervision` into the private training-loss arguments;
these auxiliary labels do not become extra top-level public dataset fields.
TCP rejects unsupported free-form text or missing target declarations rather than
guessing a command or silently using a zero target.
Other language policies can consume `raw_lang` directly; this does not by itself
make arbitrary VLM policies compatible with the vehicle action space or expert losses.

The live environment returns the same `image`, `state`, complete `raw_lang`
contract, plus a timestep. It never returns expert/future training targets.
The task text is updated as the route progresses. The training processor and live
policy use the same conversion. Coordinate convention and
instruction granularity must remain consistent between training and rollout.

## Where a deployed route point comes from

A map/navigation service supplies a route polyline (or road/lane sequence);
localization supplies current pose and heading. A route-following adapter matches
the pose to progress on the route, chooses a point farther along that route, and
transforms it to the vehicle frame. Map coordinates and sensor poses must first
share a metric reference frame. Route progress matters at intersections or
overlapping roads; picking the globally closest point alone can jump branches.

For an illustrative east/north map, a car at `(100,200)` facing east and a route
point at `(112,204)` produce `[12,-4]` in forward/right axes. Lookahead distance
is a planner parameter, potentially dependent on speed and curvature; 15 m would
be an example, not a universal setting. A coarse navigation map can supply a
coarse route point; lane-level precision requires suitable mapping/localization.
The point's existence says nothing about obstacle clearance. In this CARLA
backend the benchmark's supplied route takes the place of the deployment map
route, while GNSS/IMU provide the pose used for the conversion.
