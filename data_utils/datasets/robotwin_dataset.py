import os
import sys
import numpy as np
import cv2
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import requests
import torch
import h5py
from collections import OrderedDict

try:
    from .base import EpisodicDataset
except ImportError:
    from data_utils.datasets.base import EpisodicDataset


# ============================================================================
# RoboTwin's load_hdf5 function - adapted from process_data.py
# ============================================================================
def load_hdf5(path):
    """
    Load RoboTwin HDF5 file.
    Returns: (left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict)
    
    Images are stored as encoded JPEG bytes in '/observation/{camera_name}/rgb'
    """
    with h5py.File(path, 'r') as root:
        # Load joint states
        left_gripper_all = root['/joint_action/left_gripper'][()]
        left_arm_all = root['/joint_action/left_arm'][()]
        right_gripper_all = root['/joint_action/right_gripper'][()]
        right_arm_all = root['/joint_action/right_arm'][()]
        
        # Load images - stored in '/observation/{camera_name}/rgb' as encoded JPEG bytes
        image_dict = {}
        for camera_name in ['head_camera', 'left_camera', 'right_camera', 'front_camera']:
            try:
                rgb_path = f'/observation/{camera_name}/rgb'
                if rgb_path in root:
                    image_dict[camera_name] = root[rgb_path][()]
            except KeyError:
                pass  # Camera not available
    
    return left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict


class RoboTwinDataset(EpisodicDataset):
    """
    RoboTwin dataset for ILStudio.
    Directly loads raw RoboTwin HDF5 data and converts to ILStudio's standard format.
    
    Usage in config:
        datasets:
          - type: data_utils.datasets.robotwin_dataset.RoboTwinDataset
            name: robotwin_data
            args:
              dataset_path: /path/to/data  # or dataset_name for HuggingFace
              image_size: [480, 640]
              chunk_size: 16
              camera_names: ['head_camera', 'left_camera', 'right_camera']
              ctrl_space: qpos
              ctrl_type: abs
              preload_data: false
    """

    HF_REPO_ID = "TianxingChen/RoboTwin2.0"
    HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    CACHE_DIR = Path.home() / ".cache" / "ilstudio" / "robotwin"

    def __init__(
        self,
        dataset_path: str = None,
        dataset_name: str = None,
        image_size: tuple = (480, 640),
        chunk_size: int = 16,
        camera_names: list = None,
        ctrl_space: str = 'qpos',
        ctrl_type: str = 'abs',
        preload_data: bool = False,
    ):
        """
        Initialize RoboTwin dataset.
        
        Args:
            dataset_path: Local path to dataset (.zip or extracted directory)
            dataset_name: Hugging Face dataset name (e.g., 'adjust_bottle/aloha-agilex_clean_50')
            image_size: (height, width) for image resizing
            chunk_size: Number of timesteps per sample
            camera_names: List of camera names to load
            ctrl_space: Control space ('qpos', 'ee', etc.)
            ctrl_type: Control type ('abs', 'rel', etc.)
            preload_data: Whether to preload all data into memory
        """
        if camera_names is None:
            camera_names = ['head_camera', 'left_camera', 'right_camera']
        
        if isinstance(image_size, list):
            image_size = tuple(image_size)

        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._preload_data = preload_data
        self._episode_cache = {}  # Cache loaded episodes

        # Get raw data directory
        if dataset_name:
            self.raw_data_dir = self._download_and_extract_hf(dataset_name)
        elif dataset_path:
            self.raw_data_dir = self._extract_local_data(dataset_path)
        else:
            raise ValueError("Either 'dataset_path' or 'dataset_name' must be provided.")

        # Find all raw episode HDF5 files
        raw_episode_files = sorted(list(self.raw_data_dir.glob('episode*.hdf5')))
        if not raw_episode_files:
            raise FileNotFoundError(f"No episode HDF5 files found in: {self.raw_data_dir}")

        self.raw_episode_files = raw_episode_files

        # Infer action and state dimensions from first episode
        try:
            _, left_arm, _, right_arm, _ = load_hdf5(str(raw_episode_files[0]))
            left_dof = left_arm.shape[1]  # Number of joints per arm
            right_dof = right_arm.shape[1]
            # state_dim = left_joints + left_gripper + right_joints + right_gripper
            inferred_state_dim = left_dof + 1 + right_dof + 1
        except Exception as e:
            logger.warning(f"Could not infer dimensions from data: {e}. Using default 14.")
            inferred_state_dim = 14  # Default for aloha-agilex (6+1+6+1)

        # Initialize base class with raw file paths
        super().__init__(
            dataset_path_list=[str(p) for p in raw_episode_files],
            camera_names=camera_names,
            chunk_size=chunk_size,
            ctrl_space=ctrl_space,
            ctrl_type=ctrl_type,
            image_size=image_size,
            preload_data=preload_data
        )

        # Store inferred dimensions
        self.state_dim = inferred_state_dim
        self.action_dim = inferred_state_dim

        logger.info(f"RoboTwinDataset initialized with {len(self)} frames from {self.raw_data_dir}")
        logger.info(f"  - State dim: {self.state_dim}, Action dim: {self.action_dim}")
        logger.info(f"  - Episodes: {len(self.raw_episode_files)}, Total frames: {len(self)}")

    def get_episode_len(self):
        """Get lengths of all episodes from raw RoboTwin data."""
        all_episode_len = []
        for dataset_path in self.dataset_path_list:
            try:
                left_gripper_all, _, _, _, _ = load_hdf5(dataset_path)
                # Number of valid transitions is num_frames - 1 (action points to next state)
                episode_len = max(0, left_gripper_all.shape[0] - 1)
                all_episode_len.append(episode_len)
            except Exception as e:
                logger.error(f"Error getting episode length for {dataset_path}: {e}")
                all_episode_len.append(0)
        return all_episode_len

    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """
        Load one timestep from a raw RoboTwin episode.
        Converts RoboTwin's native format to ILStudio's standard format.
        """
        # Cache episode data
        if dataset_path not in self._episode_cache:
            self._episode_cache[dataset_path] = load_hdf5(dataset_path)

        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = self._episode_cache[dataset_path]

        # State: current qpos (7 left joints + 1 left gripper + 7 right joints + 1 right gripper)
        left_gripper = left_gripper_all[start_ts]
        left_arm = left_arm_all[start_ts]
        right_gripper = right_gripper_all[start_ts]
        right_arm = right_arm_all[start_ts]
        
        state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0).astype(np.float32)

        # Action: next state (or zeros if at episode end)
        action = np.zeros(self.action_dim, dtype=np.float32)
        if start_ts + 1 < left_gripper_all.shape[0]:
            next_left_gripper = left_gripper_all[start_ts + 1]
            next_left_arm = left_arm_all[start_ts + 1]
            next_right_gripper = right_gripper_all[start_ts + 1]
            next_right_arm = right_arm_all[start_ts + 1]
            action = np.concatenate((next_left_arm, [next_left_gripper], next_right_arm, [next_right_gripper]), axis=0).astype(np.float32)

        # Images: decode from bytes and resize
        images = OrderedDict()
        for cam_name in self.camera_names:
            if cam_name in image_dict and len(image_dict[cam_name]) > start_ts:
                camera_bits = image_dict[cam_name][start_ts]
                img = cv2.imdecode(np.frombuffer(camera_bits, np.uint8), cv2.IMREAD_COLOR)
                img_resized = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
                images[cam_name] = img_resized  # HWC format
            else:
                # Provide black image if missing
                images[cam_name] = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

        # Return data dict in ILStudio format
        data_dict = {
            'action': action[np.newaxis, :],  # (1, action_dim)
            'image': images,  # OrderedDict of camera images
            'state': state,  # (state_dim,)
            'language_instruction': "",  # Not available in RoboTwin
            'reasoning': "",
            'timestamp': start_ts,
        }
        return data_dict

    def _download_and_extract_hf(self, dataset_name):
        """Download and extract dataset from Hugging Face."""
        url = f"{self.HF_ENDPOINT}/{self.HF_REPO_ID}/resolve/main/dataset/{dataset_name}.zip"
        download_path = self.CACHE_DIR / f"{dataset_name.replace('/', '_')}.zip"
        extract_dir = self.CACHE_DIR / dataset_name

        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"RoboTwin data already extracted to {extract_dir}")
            return self._find_data_dir(extract_dir)

        if not download_path.exists():
            logger.info(f"Downloading from {url}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                with open(download_path, 'wb') as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8192), 
                                     total=total_size // 8192, unit='KB', 
                                     desc=f"Downloading {dataset_name}"):
                        f.write(chunk)
                logger.info("Download complete.")
            except Exception as e:
                logger.error(f"Download failed: {e}")
                if download_path.exists():
                    download_path.unlink()
                raise

        logger.info(f"Extracting to {extract_dir}...")
        try:
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info("Extraction complete.")
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            raise

        return self._find_data_dir(extract_dir)

    def _extract_local_data(self, dataset_path):
        """Extract local zip or use directory directly."""
        path = Path(dataset_path)
        
        if path.is_dir():
            return self._find_data_dir(path)
        elif path.is_file() and path.suffix == '.zip':
            extract_dir = self.CACHE_DIR / path.stem
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            if extract_dir.exists() and any(extract_dir.iterdir()):
                logger.info(f"Already extracted to {extract_dir}")
                return self._find_data_dir(extract_dir)

            logger.info(f"Extracting {path} to {extract_dir}...")
            try:
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                logger.info("Extraction complete.")
            except Exception as e:
                logger.error(f"Extraction failed: {e}")
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
                raise

            return self._find_data_dir(extract_dir)
        else:
            raise ValueError(f"Invalid dataset_path: {dataset_path}")

    def _find_data_dir(self, root_path):
        """Find directory containing episode*.hdf5 files."""
        root_path = Path(root_path)
        
        # Check if root has episodes directly
        if list(root_path.glob('episode*.hdf5')):
            logger.info(f"Found episodes in {root_path}")
            return root_path
        
        # Check for 'data' subdirectory
        data_dir = root_path / 'data'
        if data_dir.exists() and list(data_dir.glob('episode*.hdf5')):
            logger.info(f"Found episodes in {data_dir}")
            return data_dir
        
        # Recursively search subdirectories
        for subdir in root_path.rglob('*'):
            if subdir.is_dir() and list(subdir.glob('episode*.hdf5')):
                logger.info(f"Found episodes in {subdir}")
                return subdir
        
        raise FileNotFoundError(f"No episode*.hdf5 files found in {root_path} or subdirectories")

    def get_dataset_statistics(self):
        """Compute dataset statistics for normalization."""
        logger.info("Computing dataset statistics...")
        
        all_states = []
        all_actions = []
        
        for dataset_path in tqdm(self.dataset_path_list, desc="Computing statistics"):
            try:
                left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, _ = load_hdf5(dataset_path)
                
                # Collect all states and actions
                for j in range(left_gripper_all.shape[0]):
                    left_gripper = left_gripper_all[j]
                    left_arm = left_arm_all[j]
                    right_gripper = right_gripper_all[j]
                    right_arm = right_arm_all[j]
                    
                    state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper])).astype(np.float32)
                    all_states.append(state)
                    
                    # Action is next state (skip last frame)
                    if j + 1 < left_gripper_all.shape[0]:
                        next_left_gripper = left_gripper_all[j + 1]
                        next_left_arm = left_arm_all[j + 1]
                        next_right_gripper = right_gripper_all[j + 1]
                        next_right_arm = right_arm_all[j + 1]
                        action = np.concatenate((next_left_arm, [next_left_gripper], next_right_arm, [next_right_gripper])).astype(np.float32)
                        all_actions.append(action)
                        
            except Exception as e:
                logger.warning(f"Error processing {dataset_path}: {e}")
                continue
        
        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        
        # Return statistics in the format expected by ILStudio normalizers
        # Format: {key: {stat_name: value}} or {key: {stat_name: value} for each statistic}
        stats = {
            'state': {
                'mean': all_states.mean(axis=0),
                'std': all_states.std(axis=0),
                'min': all_states.min(axis=0),
                'max': all_states.max(axis=0),
            },
            'action': {
                'mean': all_actions.mean(axis=0),
                'std': all_actions.std(axis=0),
                'min': all_actions.min(axis=0),
                'max': all_actions.max(axis=0),
            }
        }
        
        logger.info("Statistics computed.")
        return stats
