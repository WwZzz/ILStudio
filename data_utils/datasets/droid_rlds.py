"""
This dataset implementation is from openpi.
"""

from enum import Enum
from enum import auto
import json
from pathlib import Path
import tqdm
try:
    import dlimp as dl
    import tensorflow as tf
    import tensorflow_datasets as tfds
except ImportError:
    pass
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)
    
class DroiRLDSDataset:
    def __init__(
        self,
        dataset_path_list: list, 
        camera_names: list=[], 
        action_normalizers: dict = {},  
        state_normalizers: dict = {}, 
        data_args=None, 
        chunk_size: int = 16,  
        ctrl_space: str = 'ee', 
        ctrl_type: str = 'delta',
        *args, 
        **kwargs,
    ):
        assert len(dataset_path_list)==1
        self.dataset_dir = dataset_path_list[0]
        builder = tfds.builder("droid", data_dir=self.dataset_dir, version="1.0.1")
        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False)
        # Filter out any unsuccessful trajectories -- we use the file name to check this
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )
        dataset = dataset.repeat() # Repeat dataset so we never run out of data.

        self.filter_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
        )
        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # Important: we use joint *position* action space -- easier to simulate!
            actions = tf.concat(
                (
                    traj["action_dict"]["joint_position"],
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
            # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            exterior_img = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: traj["observation"]["exterior_image_1_left"],
                lambda: traj["observation"]["exterior_image_2_left"],
            )
            wrist_img = traj["observation"]["wrist_image_left"]
            # Randomly sample one of the three language instructions
            instruction = tf.random.shuffle(
                [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
            )[0]
            traj_len = tf.shape(traj["action"])[0]
            indices = tf.as_string(tf.range(traj_len))
            # Data filtering:
            # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
            # and each step's time step index. This will index into the filter hash table, and if it returns true,
            # then the frame passes the filter.
            step_id = (
                traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                + "--"
                + traj["traj_metadata"]["episode_metadata"]["file_path"]
                + "--"
                + indices
            )
            passes_filter = self.filter_table.lookup(step_id)
            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction,
                "step_id": step_id,
                "passes_filter": passes_filter,
            }
        dataset = dataset.traj_map(restructure)
        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["actions"])[0]

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(chunk_size)[None],
                [traj_len, chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        dataset = dataset.traj_map(chunk_actions)
        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten()
        # Filter data that doesn't pass the filter
        def filter_from_dict(frame):
            return frame["passes_filter"]
        dataset = dataset.filter(filter_from_dict)
        # Remove "passes_filter" key from output
        def remove_passes_filter(frame):
            frame.pop("passes_filter")
            return frame
        dataset = dataset.map(remove_passes_filter)
        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["wrist_image"] = tf.io.decode_image(
                traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            return traj
        dataset = dataset.frame_map(decode_images, num_parallel_calls)
        # Shuffle, batch
        # dataset = dataset.shuffle(shuffle_buffer_size)
        # dataset = dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)
        self.dataset = dataset

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000

if __name__=='__main__':
    print('ok')
    