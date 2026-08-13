"""TD3+BC with the paper's batch-adaptive Q normalization."""

from numbers import Real

from .td3 import TD3Algorithm
from .utils import detached_metric


class TD3BCAlgorithm(TD3Algorithm):
    """TD3 actor objective: behavior cloning minus normalized dataset Q."""

    METRIC_PREFIX = "td3_bc"

    def __init__(self, *, bc_alpha=2.5, **kwargs):
        if not isinstance(bc_alpha, Real) or float(bc_alpha) <= 0.0:
            raise ValueError("bc_alpha must be positive")
        if "actor_bc_coef" in kwargs:
            raise TypeError("TD3+BC fixes the BC coefficient to one; use bc_alpha")
        super().__init__(actor_bc_coef=1.0, **kwargs)
        self.bc_alpha = float(bc_alpha)

    def _q_coefficient(self, actor_q1):
        return self.bc_alpha / actor_q1.detach().abs().mean().clamp_min(1e-6)

    def _combine_actor_loss(self, actor_q1, bc_loss):
        coefficient = self._q_coefficient(actor_q1)
        return -coefficient * actor_q1.mean() + bc_loss

    def _extra_actor_metrics(self, actor_q1):
        return {
            "td3_bc/q_coefficient": detached_metric(
                self._q_coefficient(actor_q1)
            )
        }


__all__ = ["TD3BCAlgorithm"]
