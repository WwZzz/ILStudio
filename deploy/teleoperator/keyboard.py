import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
from pynput import keyboard
from deploy.teleoperator.base import BaseTeleopDevice

class KeyboardTeleop(BaseTeleopDevice):
    """
    Concrete implementation of teleoperation using keyboard input
    """

    def __init__(self, 
                 shm_name: str, 
                 shm_shape: tuple, 
                 shm_dtype: type, 
                 action_dim: int = 7,
                 action_dtype = np.float64,
                 frequency: int = 100, 
                 gripper_index: int = -1, 
                 delta_scale=1.0):
        """
        Initialize the keyboard teleoperation device
        
        Args:
            shm_name: Name of the shared memory segment
            shm_shape: Shape of the shared memory array
            shm_dtype: Data type of the shared memory array
            action_dim: The dim of the flattened action
            frequency: Control frequency in Hz
            gripper_index: Index of the gripper control in the action array
            gripper_width: Maximum width of the gripper in meters
        """
        super().__init__(shm_name, shm_shape, shm_dtype, action_dim, action_dtype, frequency)
        self.pressed_keys = set()

        # Define control sensitivity as class attributes
        self.MIN_TRANS_STEP = 0.005*delta_scale  # Minimum translation step per key press (meters)
        self.MIN_ROT_STEP = np.deg2rad(1.5)*delta_scale  # Minimum rotation step per key press (radians)
        self.gripper_index = gripper_index
        self.gripper_index = [gripper_index] if isinstance(gripper_index, int) else gripper_index
        self._start_keyboard_listener()

    def _on_press(self, key):
        """Callback function when a key is pressed"""
        try:
            self.pressed_keys.add(key.char)
        except AttributeError:
            # Handle special keys like arrow keys, space, etc.
            self.pressed_keys.add(key)

    def _on_release(self, key):
        """Callback function when a key is released"""
        try:
            self.pressed_keys.remove(key.char)
        except AttributeError:
            self.pressed_keys.remove(key)
        except KeyError:
            pass  # Ignore keys that are not in the set

    def _start_keyboard_listener(self):
        """Start a separate thread to listen for keyboard events"""
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.daemon = True  # Set as daemon thread, exits when main process exits
        listener.start()
        print("Keyboard listener started.")

    def get_doc(self):
        return "\n--- Control Instructions ---\n \
            Translation: W/S (forward/backward), A/D (left/right), \
            Q/E (up/down)\nRotation: U/O (around X-axis), I/K (around Y-axis),\
            J/L (around Z-axis)\nGripper: Hold SPACE to close, \
            release to open\n\
            Press Ctrl+C to exit the program.\
            \n-----------------\n"
    
    def get_observation(self):
        """Get all currently pressed keys"""
        return self.pressed_keys.copy()

    def observation_to_action(self, observation):
        """Convert key states to robot end-effector delta pose and gripper action"""
        action = self.get_zero_action()

        # Translation control (X, Y, Z)
        if 'a' in observation: action[0] -= self.MIN_TRANS_STEP
        if 'd' in observation: action[0] += self.MIN_TRANS_STEP
        if 'w' in observation: action[1] += self.MIN_TRANS_STEP
        if 's' in observation: action[1] -= self.MIN_TRANS_STEP
        if 'q' in observation: action[2] += self.MIN_TRANS_STEP
        if 'e' in observation: action[2] -= self.MIN_TRANS_STEP

        # Rotation control (Roll, Pitch, Yaw)
        if 'j' in observation: action[3] += self.MIN_ROT_STEP
        if 'l' in observation: action[3] -= self.MIN_ROT_STEP
        if 'i' in observation: action[4] += self.MIN_ROT_STEP
        if 'k' in observation: action[4] -= self.MIN_ROT_STEP
        if 'u' in observation: action[5] += self.MIN_ROT_STEP
        if 'o' in observation: action[5] -= self.MIN_ROT_STEP

        # Gripper control
        if keyboard.Key.space in observation:
            action[6] = 0.  # Close signal
        else:
            action[6] = 1.0  # Open signal
        print('create_action', action)
        return action