# SO101 Plus Follower (single arm)

Stock SO101 follower upgraded with an extra `wrist_yaw` joint.

## Motor IDs

| Joint | ID | Notes |
|-------|----|--------|
| shoulder_pan | 1 | unchanged |
| shoulder_lift | 2 | unchanged |
| elbow_flex | 3 | unchanged |
| wrist_flex | 4 | unchanged |
| wrist_roll | 5 | unchanged |
| gripper | 6 | unchanged |
| **wrist_yaw** | **7** | **new** |

Logical qpos / action order keeps gripper last:

```
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, wrist_yaw, gripper]
```

## Setup motors (IDs)

Connect **one** motor at a time when prompted:

```bash
python -m deploy.robot.so101_plus --port /dev/so101_left_arm --id so101_plus_left --setup-motors
```

If IDs 1–6 are already programmed from the stock SO101 arm, you still need to run this for the new `wrist_yaw` (ID 7). You can skip reconnecting motors that already have the correct ID, or re-run the full sequence.

## Calibrate (all 7 joints)

```bash
python -m deploy.robot.so101_plus --port /dev/so101_left_arm --id so101_plus_left --calibrate
```

During ROM recording, move **every** joint including `wrist_yaw`. Calibration is saved under:

```
~/.cache/huggingface/lerobot/calibration/robots/so101_plus/<robot_id>.json
```

Use a distinct `robot_id` (default in YAML: `so101_plus_left`) so it does not overwrite stock SO101 calibration files.

Connecting via the ILStudio robot config will also prompt for calibration if the file/motors do not match.

## Run with ILStudio

```bash
python eval_real.py -r configs/robot/so101_plus.yaml -m __dummy-random_7x16
```

Config: `configs/robot/so101_plus.yaml`  
Class: `deploy.robot.so101_plus.So101Plus`

## Notes

- Control mode is **qpos only** for now (no delta_ee / IK until wrist_yaw kinematics are modeled).
- Does not modify the stock `so101` module or lerobot `SO101Follower`.
