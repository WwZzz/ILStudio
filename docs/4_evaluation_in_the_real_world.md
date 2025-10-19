# 4. Evaluation in the Real World

This guide covers how to deploy and evaluate a trained policy on a physical robot using the `eval_real.py` script.

## ⚠️ Safety First!

**Warning**: Running policies on a real robot can be dangerous. Real-world hardware can behave unexpectedly.
*   **Always be prepared to stop the robot.** Keep the emergency stop button within reach.
*   **Clear the workspace.** Ensure the robot's operating area is free of any obstacles or personnel.
*   **Start with low speeds.** If possible, test at a reduced speed before running at full speed.

## System Architecture

The `eval_real.py` script uses a multi-threaded architecture to handle the different rates of sensing, inference, and action:

1.  **Sensing Thread**: Runs at a high frequency (`--sensing_rate`) to capture the latest observations (e.g., camera images, joint states) from the robot. It puts these observations into a queue.
2.  **Inference Thread**: Waits for an observation from the queue. When it receives one, it runs the policy model to produce a *chunk* of future actions. This runs at a lower frequency, as inference can be slow. The resulting action chunk is passed to the Action Manager.
3.  **Main Control Loop (Action Thread)**: Runs at the robot's required control frequency (`--publish_rate`). It continuously queries the **Action Manager** for the next action and sends it to the robot hardware.

## Example Usage

This example shows how to run an evaluation on a hypothetical "agilex_aloha" robot for the "transfer_cube" task.

```bash
python eval_real.py \
    --model_name_or_path ckpt/act_sim_transfer_cube_scripted_zscore_example \
    --robot_config agilex_aloha \
    --task agilex_transfer_cube \
    --publish_rate 50 \
    --sensing_rate 25 \
    --action_manager OlderFirstManager
```

## Key Arguments

*   `--model_name_or_path` (string):
    *   **Description**: Path to the model checkpoint *or* a server address (`host:port`) if using a remote Policy Server.
    *   **Example (local)**: `ckpt/act_sim_transfer_cube_scripted_zscore_example`
    *   **Example (remote)**: `192.168.1.101:5000`

*   `--robot_config` (string):
    *   **Description**: The name of the robot configuration YAML file from `configs/robot/`. This defines the robot's hardware interface, camera setup, and teleop device.
    *   **Example**: `agilex_aloha` (refers to `configs/robot/agilex_aloha.yaml`)

*   `--task` (string):
    *   **Description**: The name of the task configuration YAML file from `configs/task/`. This defines the datasets, normalization, and policy settings used for training, which are needed to load the model correctly.
    *   **Example**: `agilex_transfer_cube`

*   `--publish_rate` (int):
    *   **Description**: The frequency (Hz) at which the main control loop sends action commands to the robot. This should match the robot's expected control rate.
    *   **Default**: `25`

*   `--sensing_rate` (int):
    *   **Description**: The frequency (Hz) at which the sensing thread polls the robot for new observations.
    *   **Default**: `20`

*   `--action_manager` (string):
    *   **Description**: The name of the Action Manager class to use. See the Action Manager documentation for more details.
    *   **Default**: `OlderFirstManager`

## Pre-flight Checklist

1.  ✅ **Robot On**: The robot is powered on and initialized.
2.  ✅ **Network Connection**: Your machine can communicate with the robot (e.g., via ROS, Ethernet).
3.  ✅ **Drivers Running**: The robot's low-level control software/drivers are running.
4.  ✅ **Correct Configs**: The `--robot_config` and `--task` files accurately reflect your setup.
5.  ✅ **Safety**: The workspace is clear and you are ready to stop the robot if needed.
