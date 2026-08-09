"""Shared scheduling for on-policy rollout algorithms."""

from rl.buffer import RolloutBuffer


class FullRolloutUpdates:
    """Reuse a sealed rollout for a configured number of full-batch epochs."""

    def iter_update_batches(
        self,
        buffer,
        *,
        batch_size,
        num_updates,
        rng,
    ):
        del batch_size, rng
        if not isinstance(buffer, RolloutBuffer):
            raise TypeError("on-policy algorithms require RolloutBuffer")
        if not buffer.sealed:
            raise RuntimeError("rollout buffer must be sealed before updates")
        if isinstance(num_updates, bool) or not isinstance(num_updates, int) or num_updates < 0:
            raise ValueError("num_updates must be a non-negative integer")
        for _ in range(num_updates):
            if len(buffer):
                yield buffer.get_batch(range(len(buffer)))
