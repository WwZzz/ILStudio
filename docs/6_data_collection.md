# 6. Data Collection

High-quality demonstration data is crucial for imitation learning. IL-Studio provides a powerful, decoupled system for collecting expert data via teleoperation.

## System Architecture

The data collection system consists of two independent processes that communicate via **shared memory**:

1.  **`start_teleop_controller.py` (The Controller)**:
    *   This process runs in a separate terminal.
    *   It connects to your physical teleoperation device (e.g., VR controller, keyboard).
    *   It continuously reads the device's state and writes the resulting robot action into a shared memory block.
    *   It is hardware-focused and does not interact with the robot directly.

2.  **`start_teleop_recorder.py` (The Recorder)**:
    *   This is the main script you will interact with.
    *   It connects to the robot hardware to read observations (camera images, joint states).
    *   It also connects to the *same shared memory block* to read the actions being written by the controller.
    *   It pairs the observations and actions by timestamp and saves them as an episode.

This decoupled design allows the teleoperation device and the robot to run on different machines, as long as they have access to the same shared memory.

## Step 1: Start the Teleop Controller

In **Terminal 1**, start the controller process. This process will create the shared memory block.
```bash
python start_teleop_controller.py \
    --teleop_config <teleop_config_name> \
    --shm_name "ilstd_teleop_shm"
```
*   `--teleop_config`: Defines your leader device (e.g., `so101_leader` for a VR controller). Found in `configs/teleop/`.

## Step 2: Start the Teleop Recorder

In **Terminal 2**, start the recorder process.
```bash
python start_teleop_recorder.py \
    -c <robot_config_name> \
    -o data/my_new_task \
    --shm_name "ilstd_teleop_shm" \
    --frequency 30
```
The recorder will connect to the shared memory created by the controller.

## The Recording Process

1.  With both scripts running, press **Enter** in the Recorder terminal (`Terminal 2`) to **start** recording an episode.
2.  Use your teleoperation device to perform the desired task. The robot should move according to your inputs.
3.  When you are finished with the task, press **Enter** again in the Recorder terminal to **stop** recording.
4.  The script will ask if you want to save the episode. Press **Enter** to save, or type any character and press Enter to discard.
5.  Repeat the process to collect more episodes.

## Key Arguments for `start_teleop_recorder.py`

*   `-c, --config` (string):
    *   **Description**: The name of the robot configuration from `configs/robot/`. This defines the robot hardware that is being controlled.
    *   **Example**: `agilex_aloha`

*   `-o, --output_dir` (string):
    *   **Description**: The directory where the collected HDF5 episode files will be saved.
    *   **Default**: `data/teleop_recordings`

*   `-shm, --shm_name` (string):
    *   **Description**: The name for the shared memory block. This **must** match the name used by `start_teleop_controller.py`.
    *   **Default**: `ilstd_teleop_controller`

*   `-f, --frequency` (int):
    *   **Description**: The target frequency (Hz) for recording observation-action pairs.
    *   **Default**: `25`

*   `-s, --start_idx` (int):
    *   **Description**: The starting index for episode numbering (e.g., `episode_0000.hdf5`). Useful for resuming a data collection session.
    *   **Default**: `0`
