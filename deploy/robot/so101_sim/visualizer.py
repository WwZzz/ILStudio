"""
SO101 Simulation Visualizer

MuJoCo-based visualization for SO101 simulation robot.
Reads robot state from shared memory and renders the robot in a MuJoCo viewer.
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path
from typing import Optional

from deploy.visualizer.base import BaseVisualizer


# Joint names in MuJoCo model (must match robot.py)
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


class Visualizer(BaseVisualizer):
    """
    MuJoCo Visualizer for SO101 Simulation Robot.
    
    Reads 'qpos' from shared memory and renders the robot state.
    """
    
    def __init__(self, 
                 shm_name: str, 
                 fps: float = 60.0,
                 xml_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize the visualizer.
        
        Args:
            shm_name: Name of the shared memory to read robot data from
            fps: Target visualization frame rate
            xml_path: Path to MuJoCo XML model file (uses default if None)
        """
        super().__init__(shm_name=shm_name, fps=fps, **kwargs)
        
        # Locate XML model
        if xml_path is None:
            module_dir = Path(__file__).parent
            xml_path = str(module_dir / "mujoco_model" / "scene.xml")
        self.xml_path = xml_path
        
        # MuJoCo objects
        self.mjmodel = None
        self.mjdata = None
        self.viewer = None
        self.qpos_indices = None
    
    def setup(self) -> bool:
        """
        Load MuJoCo model and launch the viewer.
        """
        try:
            print(f"[Visualizer] Loading MuJoCo model from {self.xml_path}")
            self.mjmodel = mujoco.MjModel.from_xml_path(self.xml_path)
            self.mjdata = mujoco.MjData(self.mjmodel)
            
            # Get joint indices
            self.qpos_indices = np.array([
                self.mjmodel.jnt_qposadr[self.mjmodel.joint(name).id] 
                for name in JOINT_NAMES
            ])
            
            # Launch passive viewer
            self.viewer = mujoco.viewer.launch_passive(
                self.mjmodel, 
                self.mjdata,
                show_left_ui=True,
                show_right_ui=True,
            )
            
            print("[Visualizer] MuJoCo viewer launched")
            return True
            
        except Exception as e:
            print(f"[Visualizer] Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def visualize(self, data: dict) -> bool:
        """
        Update the visualization with new robot state.
        
        Args:
            data: Data from shared memory, expected to contain 'qpos'
            
        Returns:
            True to continue, False if viewer was closed
        """
        # Check if viewer is still running
        if self.viewer is None or not self.viewer.is_running():
            return False
        
        # Get qpos from data
        qpos = data.get('qpos', None)
        if qpos is None:
            return True  # No qpos data, but continue
        
        qpos = np.array(qpos)
        if len(qpos) != 6:
            return True  # Invalid qpos, but continue
        
        # Update MuJoCo state
        self.mjdata.qpos[self.qpos_indices] = qpos
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        
        # Sync viewer
        with self.viewer.lock():
            pass  # State already updated
        self.viewer.sync()
        
        return True
    
    def cleanup(self):
        """
        Close the viewer.
        """
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        self.mjmodel = None
        self.mjdata = None


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SO101 Visualizer")
    parser.add_argument("--shm", "-s", type=str, default="so101_sim_test",
                        help="Shared memory name to connect to")
    args = parser.parse_args()
    
    visualizer = Visualizer(shm_name=args.shm)
    visualizer.start()
