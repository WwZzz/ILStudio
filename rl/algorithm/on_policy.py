"""Shared scheduling for on-policy rollout algorithms."""

from collections import OrderedDict

from rl.base import RL_LIKELIHOOD_GROUP_KEY, RL_LIKELIHOOD_GROUP_SIZE_KEY

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


def iter_likelihood_micro_batches(decisions, max_decisions):
    """Pack decisions without splitting one batched-likelihood forward.

    Mixed-precision transformer logits can depend materially on batch shape.
    Decisions sampled by one batched policy call must therefore be recomputed
    together when PPO/GRPO forms the new likelihood ratio.  Ungrouped decisions
    remain independently packable for adapters that use scalar inference.
    """

    decisions = tuple(decisions)
    if not decisions:
        return
    if (
        isinstance(max_decisions, bool)
        or not isinstance(max_decisions, int)
        or max_decisions <= 0
    ):
        raise ValueError("max_decisions must be a positive integer")

    grouped = OrderedDict()
    expected_sizes = {}
    for decision in decisions:
        group_id = decision.extras.get(RL_LIKELIHOOD_GROUP_KEY)
        if group_id is None:
            key = ("decision", decision.decision_id)
            expected_size = 1
        else:
            key = ("likelihood", group_id)
            expected_size = decision.extras.get(RL_LIKELIHOOD_GROUP_SIZE_KEY)
            if (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size <= 0
            ):
                raise ValueError("likelihood group size must be a positive integer")
        previous_size = expected_sizes.setdefault(key, expected_size)
        if previous_size != expected_size:
            raise ValueError("likelihood group members disagree on group size")
        grouped.setdefault(key, []).append(decision)

    atomic_groups = []
    for key, members in grouped.items():
        if len(members) != expected_sizes[key]:
            raise ValueError("likelihood recompute group is incomplete")
        atomic_groups.append(tuple(members))

    pending = []
    for group in atomic_groups:
        if pending and len(pending) + len(group) > max_decisions:
            yield tuple(pending)
            pending = []
        pending.extend(group)
        if len(pending) >= max_decisions:
            yield tuple(pending)
            pending = []
    if pending:
        yield tuple(pending)
