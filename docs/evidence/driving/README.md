# Driving evidence

**Full validation has not passed.** The original run stopped at an upstream
`CrossingBicycleFlow` initialization error. The agreed acceptance gate requires a TCP policy
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
| Regression checks | 124 passed, 2 optional-dependency skips |

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

The interrupted evaluation used three disjoint contiguous route ranges
(74/73/73) on one GPU. Planned aggregation requires the exact 220-route sequence,
identical model and adapter hashes, matching environment parameters, and complete
initialization coverage. The incomplete per-range journals are retained.

The incomplete original run retained 23 completed routes (6 successes, partial
SR 26.09%, partial DS 48.32). Route 35418 could not initialize its requested
bicycle scenario and was rejected rather than scored as an easier scenario.
These partial values do not satisfy the full-split gate. Both official native
Python 3.7 and current Python 3.10 clients reproduce the same topology gap.

The opt-in bicycle repair initializes all four bicycle-route configurations:
one native path stays unchanged, and three receive the same bounded 27.12 m
junction connection. An isolated native witness recorded 300 physics steps,
6 generated bicycles, 5 with valid motion over 3 m, maximum speed 11.05 m/s,
and maximum per-frame displacement 0.552 m. A single-route learned-policy
compatibility diagnostic is separate from formal acceptance. Using patched
scenario construction for formal acceptance requires an explicit protocol choice.

The canonical release XML has 220 routes and 42 distinct scenario types with
unequal counts. The upstream release note describes 44 types with five routes
each; the integration uses the actual released route IDs without resampling.

[Incomplete run](partial-validation.json) · [Partial native episodes](partial-episodes.json) ·
[Rejected initialization](interface/invalid-bicycle-route.json) ·
[Native/client comparison](interface/bicycle-client-comparison.json) ·
[Route-file audit](validation-release-audit.json) ·
[Offline repair coverage](interface/bicycle-offline-repair.json) ·
[Native repair witness](interface/bicycle-native-repair.json)

The single-route check retains 20 low-speed records. The official SR calculation
excludes `min_speed_infractions`, and their DS penalty is disabled upstream;
the route has no other infractions. The complete native record is preserved.
Private experiment paths are replaced with `${DRIVING_ROOT}`. Source archives,
data shards, native libraries and model weights are not included in Git.
