# Autonomous driving

The initial driving backend combines CARLA 0.9.15 vehicle physics and rendering
with Bench2Drive 0.0.4 scenario dynamics and scoring. TCP is the initial learned
baseline. [Example 10](../examples/10_autonomous_driving/README.md) uses the same
`vla.load_dataset`, `vla.load_policy`, `vla.train` and environment interfaces as
the other tasks.

## Installation and caching

The `carla` environment recipe installs the Python client, scenario dependencies,
native Linux server and additional maps. `--system-packages` installs the declared
Linux graphics dependencies and requires root. CUDA alone is insufficient:
the host/container must also expose NVIDIA Vulkan and EGL libraries compatible
with its driver. The installer never changes the host kernel driver.

Set `VLASTUDIO_CACHE` to a persistent directory before creating the environment.
Sources, CARLA archives and model weights are cached under `driving/`; standard
dataset storage follows `VLASTUDIO_DATA_CACHE_DIR`. Downloading and indexing the
selected demonstration shards happens inside `Bench2DriveDataset`. The policy
loads and verifies its configured official pretrained checkpoint internally.
No separate dataset preparation or demonstration collection is required.

An existing CARLA installation can be selected with `VLASTUDIO_RUNTIME_CARLA_ROOT`
or the environment's `carla_root`. For containers with an extracted matching
NVIDIA userspace runtime, `graphics_runtime` (or `VLASTUDIO_CARLA_GRAPHICS_RUNTIME`)
accepts a JSON file with `library_path`, `icd` and optional `egl_vendor` paths.
These configure the owned simulator child process.

Native installation is also available independently through
`python -m vlastudio.benchmark.carla.install`. `vlastudio env check driving --smoke`
starts an owned server, renders an RGB camera and steps a vehicle. Import success
alone does not establish simulator readiness. CARLA's UE4 server cannot run as
root; the adapter drops privileges for its own server child when necessary.
Its installation and parent directories must be traversable by that child.

## Data, observations and actions

`Bench2DriveDataset` reads the public official 0.0.4 demonstrations. Its default
ten-route selection is deterministic and spans different scenarios. Setting
`max_trajectories: null` selects all 1,100 routes. A local extracted dataset can
be selected with `dataset_dir`; `download: false` requires cached assets.
Only training demonstrations are exposed: held-out evaluation is closed loop.

Each standard sample contains RGB `image`, `state`, four-dimensional `action`,
`is_pad`, `raw_lang`, episode/frame identity and TCP's auxiliary supervision.
The latter includes four future ego-frame waypoints, expert actions, features
and value. Future/expert information is used for the training loss only.
Negative warmup frames and terminal observations without an expert action are
excluded. Future targets stay within a single route.

Live policy inputs are the three front RGB cameras composed into a
`(1,3,256,900)` panorama and a nine-value state:
`[speed_mps, next_point_x, next_point_y, command_one_hot_6]`. The next point and
command come from the supplied navigation route, GNSS and IMU, using the authors'
route planner. Other actors' privileged state, expert actions and future ego
positions are not policy inputs. The baseline uses categorical driving commands;
it is not a free-form language-conditioned VLA model.

The environment declares `ctrl_space: vehicle`, `ctrl_type: control` and
`action_order: [steer, throttle, brake, reverse]`. Controls have ranges
`[-1,1], [0,1], [0,1], {0,1}`. TCP's default `only_traj` planner predicts four
waypoints and applies its PID controller. `only_ctrl` selects a control from the
39 expert classes. Controllers are private to each episode/batch slot.

TCP initializes from the authors' complete released checkpoint without an
additional ImageNet download. VLAStudio saves ordinary HF-compatible model
weights plus policy and normalization metadata. The example continues training
with waypoint, speed, action, value and feature losses. BatchNorm running
statistics are frozen by default for small-batch fine-tuning; network parameters
remain trainable. This is a fine-tuning example, not a claim of reproducing the
authors' full training recipe from scratch.

## Closed-loop evaluation

The default evaluation selects the exact official 0.0.4 validation XML: 220
routes, covering 44 scenario types. The adapter retains the official scenario
manager, sensor geometry, 20 Hz simulation timing and statistics manager.
Only the unused debugging BEV camera is omitted from the SENSORS track.
Scenario setup errors invalidate a run rather than silently skipping a scenario.
Cleanup targets only the server launched by this adapter.

`evaluate.py` preserves native per-route records in `episodes.jsonl`, writes
source/model/config hashes to `run.json`, and writes aggregate results only
after every selected route completes. Resuming requires the same manifest and
a unique completed prefix of the selected routes. A custom/subset run is not
reported as the complete official validation split.

Metrics follow the official code: Driving Score and route completion are on a
0–100 scale; SR requires `Completed`/`Perfect` status and no counted infractions.
The authors exclude `min_speed_infractions` from SR and disable their DS penalty;
those records are nevertheless retained. Completing a route, reaching a time
limit or exiting a process does not by itself pass the task.

The agreed full-split integration gate is **SR ≥20% and DS ≥40** for a policy
fine-tuned through VLAStudio. This gate requires all 220 distinct held-out routes;
single-route smoke results do not meet it. The authors' 0.0.4 TCP reference is
DS 51.44 and SR 26.36%.

## Pinned upstream assets

| Component | Upstream / revision |
|---|---|
| CARLA server and maps | [CARLA 0.9.15](https://carla.readthedocs.io/en/0.9.15/) |
| Scenarios and official metrics | [Bench2Drive 0.0.4](https://github.com/Thinklab-SJTU/Bench2Drive/tree/7ec25d1c9f7522d923ce5f3420986cef1cb2d956) |
| TCP model and controller | [Bench2DriveZoo TCP](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/8a08b07883f10b7d83f6bf5dd475bda91a91c50a) |
| Demonstrations | [Bench2Drive-V0.0.4](https://huggingface.co/datasets/rethinklab/Bench2Drive-V0.0.4/tree/d84ae18cf20a88b9220df1bc15c073d8824b7ac1) |
| Official TCP checkpoint | [Bench2DriveZoo weights](https://huggingface.co/rethinklab/Bench2DriveZoo/tree/140aea58c02185e787c3df39d7ad79e910967b8b) |
| Reference table | [Official 0.0.4 update](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/docs/v004_update.md) |

Sources are downloaded unmodified into the cache rather than vendored into the
package. Upstream code, data and weights retain their own licenses and terms.
The official TCP checkpoint SHA-256 is
`e6573ff1f8ea910b9a53eddfb68f69cac469bf5bfa253a516578f6126110b4fe`.
