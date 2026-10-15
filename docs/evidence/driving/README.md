# Driving evidence

**Full validation is pending.** The agreed acceptance gate requires a TCP policy
fine-tuned through VLAStudio to reach SR ≥20% and Driving Score ≥40 on all 220
official Bench2Drive 0.0.4 validation routes. Small interface checks below do not
establish that result. The fixed protocol is in [protocol.json](protocol.json).

| Check | Recorded observation |
|---|---|
| Native CARLA | Client/server 0.9.15; 20 physics steps; nonempty rendered RGB |
| Dataset | 180 valid windows from one official training route; route ID disjoint from validation |
| Standard training interface | Two optimizer updates, frozen BatchNorm configuration; saved HF checkpoint |
| Standard `env.evaluate` | Trained two-step checkpoint; official route 24721; 1/1 success, DS 100, completion 100 |
| Regression checks | 116 passed, 2 optional-dependency skips |

[Native runtime](interface/native-server.json) ·
[Installed maps](interface/native-install.json) ·
[Graphics runtime](interface/graphics.json) ·
[Dataset](interface/data.json) ·
[Trainer state](interface/training-interface.json) ·
[Standard API closed loop](interface/standard-api.json)

The single-route check retains 20 low-speed records. The official SR calculation
excludes `min_speed_infractions`, and their DS penalty is disabled upstream;
the route has no other infractions. The complete native record is preserved.
Private experiment paths are replaced with `${DRIVING_ROOT}`. Source archives,
data shards, native libraries and model weights are not included in Git.
