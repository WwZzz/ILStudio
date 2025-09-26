import tkinter as tk
import numpy as np
import multiprocessing as mp
from deploy.teleoperator.base import BaseTeleopDevice

class SliderTeleop(BaseTeleopDevice):
    """
    Teleoperator using Tkinter sliders to generate actions.
    The GUI runs in a separate process.
    """

    def __init__(self, shm_name, shm_shape, shm_dtype, action_dim, action_dtype, slider_ranges, frequency=10, **kwargs):
        super().__init__(
            shm_name=shm_name,
            shm_shape=shm_shape,
            shm_dtype=shm_dtype,
            action_dim=action_dim,
            action_dtype=action_dtype,
            frequency=frequency,
            **kwargs
        )
        self.slider_ranges = slider_ranges
        self.frequency = frequency
        self.action_dtype = action_dtype

        # Shared array for slider values
        self._slider_values = mp.Array('d', action_dim)  # double precision for generality

        # Start the GUI process
        self._gui_proc = mp.Process(
            target=self._slider_gui_process,
            args=(self._slider_values, self.slider_ranges)
        )
        self._gui_proc.daemon = True
        self._gui_proc.start()

    def _slider_gui_process(self, shared_array, slider_ranges):
        root = tk.Tk()
        root.title("Teleop Sliders")
        scales = []
        for i, (low, high) in enumerate(slider_ranges):
            frame = tk.Frame(root)
            frame.pack()
            label = tk.Label(frame, text=f"Action {i}")
            label.pack(side=tk.LEFT)
            scale = tk.Scale(frame, from_=low, to=high, resolution=0.01, orient=tk.HORIZONTAL, length=300)
            scale.set((low + high) / 2)
            scale.pack(side=tk.LEFT)
            scales.append(scale)

        def update_shared():
            for i, scale in enumerate(scales):
                shared_array[i] = scale.get()
            root.after(50, update_shared)  # update every 50 ms

        root.after(50, update_shared)
        root.mainloop()

    def get_observation(self):
        # Read the current slider values from the shared array
        arr = np.frombuffer(self._slider_values.get_obj())
        return arr.astype(self.action_dtype)

    def observation_to_action(self, observation):
        return observation

    def stop(self):
        if hasattr(self, "_gui_proc") and self._gui_proc.is_alive():
            self._gui_proc.terminate()
            self._gui_proc.join()

    def get_doc(self):
        return "SliderTeleop: Use Tkinter GUI sliders (in a separate process) to generate actions for teleoperation."