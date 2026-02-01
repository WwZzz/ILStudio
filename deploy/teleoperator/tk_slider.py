"""
Tkinter Slider Teleoperator
Uses GUI sliders to generate actions for teleoperation
"""

import tkinter as tk
import numpy as np
import multiprocessing as mp
from typing import Optional, List, Tuple

from deploy.teleoperator.base import BaseTeleopDevice


class SliderTeleop(BaseTeleopDevice):
    """
    Teleoperator using Tkinter sliders to generate actions.
    The GUI runs in a separate process.
    """

    def __init__(self,
                 name: str = "slider_teleop",
                 max_size_mb: int = 1,
                 fps: float = 30.0,
                 action_dim: int = 7,
                 slider_ranges: Optional[List[Tuple[float, float]]] = None,
                 slider_labels: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize the Slider teleoperation device
        
        Args:
            name: Name of the shared memory segment
            max_size_mb: Maximum size of shared memory in MB
            fps: Control frequency in Hz
            action_dim: Number of action dimensions
            slider_ranges: List of (min, max) tuples for each slider
            slider_labels: List of labels for each slider (optional)
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps)
        
        self.action_dim = action_dim
        
        # Default slider ranges if not provided
        if slider_ranges is None:
            slider_ranges = [(-1.0, 1.0)] * action_dim
        self.slider_ranges = slider_ranges
        
        # Default slider labels if not provided
        if slider_labels is None:
            slider_labels = [f"Action {i}" for i in range(action_dim)]
        self.slider_labels = slider_labels

        # Shared array for slider values
        self._slider_values = mp.Array('d', action_dim)  # double precision for generality

        # Start the GUI process
        self._gui_proc = mp.Process(
            target=self._slider_gui_process,
            args=(self._slider_values, self.slider_ranges, self.slider_labels)
        )
        self._gui_proc.daemon = True
        self._gui_proc.start()

    def _slider_gui_process(self, shared_array, slider_ranges, slider_labels):
        root = tk.Tk()
        root.title("Teleop Sliders")
        scales = []
        for i, (low, high) in enumerate(slider_ranges):
            frame = tk.Frame(root)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Use custom label if provided
            label_text = slider_labels[i] if i < len(slider_labels) else f"Action {i}"
            label = tk.Label(frame, text=f"{label_text}:", width=12, anchor='e')
            label.pack(side=tk.LEFT)
            
            scale = tk.Scale(frame, from_=low, to=high, resolution=0.01, orient=tk.HORIZONTAL, length=300)
            scale.set((low + high) / 2)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            scales.append(scale)
        
        # Add reset button
        def reset_all():
            for i, scale in enumerate(scales):
                low, high = slider_ranges[i]
                scale.set((low + high) / 2)
        
        reset_btn = tk.Button(root, text="Reset All to Center", command=reset_all)
        reset_btn.pack(pady=10)

        def update_shared():
            for i, scale in enumerate(scales):
                shared_array[i] = scale.get()
            root.after(50, update_shared)  # update every 50 ms

        root.after(50, update_shared)
        root.mainloop()

    def get_data(self) -> Optional[dict]:
        """Read the current slider values from the shared array"""
        arr = np.frombuffer(self._slider_values.get_obj()).copy()
        return {"sliders": arr}

    def convert_data_to_action(self, data: dict) -> np.ndarray:
        """Convert slider values to action"""
        return data.get("sliders", np.zeros(self.action_dim))

    def close(self):
        """Close the teleoperation device"""
        super().close()
        if hasattr(self, "_gui_proc") and self._gui_proc.is_alive():
            self._gui_proc.terminate()
            self._gui_proc.join()


# ==============================================================================
# Test (start slider teleop in subprocess, read from SHM and print)
# ==============================================================================

if __name__ == "__main__":
    import time
    import yaml
    from pathlib import Path

    from deploy.base import start_device
    from deploy.shm_utils import SharedMemoryChannel

    # Load config
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "teleop" / "tk_slider.yaml"
    with open(cfg_path, "r") as f:
        device_config = yaml.safe_load(f)

    shm_name = device_config["args"]["name"]
    proc = mp.Process(target=start_device, args=(device_config,))
    proc.start()

    time.sleep(0.5)
    print("Reading from SHM (Ctrl+C to stop)...")
    print("Move the sliders to see action updates.\n")

    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=10.0)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None and "action" in data:
                arr = data["action"]
                print(f"  action: {arr}", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
        print("Done.")