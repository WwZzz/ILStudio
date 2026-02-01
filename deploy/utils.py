import numpy as np
import time


class RateLimiter:
    """
    A class to manage the rate for a single thread.
    Each thread should have its own instance.
    """

    def __init__(self):
        self._last_sleep_time = time.perf_counter()

    def sleep(self, rate: float):
        """
        Sleeps for a duration that maintains the desired loop rate.

        Args:
            rate (float): The desired loop frequency in Hz.
        """
        if rate <= 0:
            return

        target_period = 1.0 / rate
        current_time = time.perf_counter()
        elapsed_time = current_time - self._last_sleep_time
        sleep_duration = target_period - elapsed_time

        if sleep_duration > 0:
            time.sleep(sleep_duration)

        # Update the timestamp for the next iteration
        self._last_sleep_time = time.perf_counter()