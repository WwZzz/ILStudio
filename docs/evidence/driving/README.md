# Driving evidence

**Full validation is pending.** The agreed acceptance gate requires a TCP policy
fine-tuned through VLAStudio to reach SR ≥20% and Driving Score ≥40 on all 220
official Bench2Drive 0.0.4 validation routes. Small interface checks below do not
establish that result. The fixed protocol is in [protocol.json](protocol.json).

| Check | Recorded observation |
|---|---|
| Native CARLA | Client/server 0.9.15; 20 physics steps; nonempty rendered RGB |
| Dataset | 1,581 valid windows from ten official training routes; route IDs disjoint from validation |
| Standard training interface | Two optimizer updates, frozen BatchNorm configuration; saved HF checkpoint |
| Acceptance checkpoint | 100 optimizer updates from official TCP, `merge_ctrl_traj`; 180 changed weight tensors, unchanged BatchNorm running statistics |
| Standard `env.evaluate` | Two-step `only_traj` checkpoint; official route 24721; 1/1 success, DS 100, completion 100 |
| Parking initialization | Original ParkingExit builds all 3 required actors after disabling duplicate stock parking; no scenario spawn failures |
| Regression checks | 122 passed, 2 optional-dependency skips |

[Native runtime](interface/native-server.json) ·
[Installed maps](interface/native-install.json) ·
[Graphics runtime](interface/graphics.json) ·
[Dataset](interface/data.json) ·
[Trainer state](interface/training-interface.json) ·
[Standard API closed loop](interface/standard-api.json) ·
[100-step training](training.json) ·
[Parking initialization](interface/managed-parking.json) ·
[Map overlay inspection](interface/map-overlay.json)

Two attempted full evaluations were rejected before any valid route was recorded:
stock Town15 parking collided with a required ParkingExit actor. Their logs and
output are retained separately on the experiment host. The current evaluation
explicitly selects `remove_map_parked_vehicles: true`, retaining Bench2Drive's
own parking catalog and scenario actors. This configuration and per-route
initialization coverage are recorded; an initialized subset cannot pass as a
complete scenario. Full-split metrics remain pending.

The single-route check retains 20 low-speed records. The official SR calculation
excludes `min_speed_infractions`, and their DS penalty is disabled upstream;
the route has no other infractions. The complete native record is preserved.
Private experiment paths are replaced with `${DRIVING_ROOT}`. Source archives,
data shards, native libraries and model weights are not included in Git.
