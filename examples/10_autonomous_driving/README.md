# 10 — Autonomous driving

Run from the repository root on Linux with Python 3.10 and an NVIDIA GPU.

| Simulator | Task / policy | Example |
|---|---|---|
| CARLA 0.9.15 | Bench2Drive 0.0.4 / TCP | [carla/](carla/README.md) |

The dataset exposes current sensors, physical `state`, complete task text in
`raw_lang`, and demonstrated actions. The local route goal is part of the task
text; TCP's data processor extracts it and command indicators internally.
[Sample fields and example values](carla/README.md#what-is-in-a-training-sample) ·
[Comparison of seven driving pipelines](../../docs/driving_data_contract.md).

[Interfaces and assets](../../docs/driving.md) · [Recorded checks](../../docs/evidence/driving/README.md)
