"""Optimizer execution behind the policy-adapter boundary."""

from .optimizer import (
    OptimizerTrainerAdapter,
    TrainerAdapter,
    TrainerStepResult,
    build_grouped_trainer_adapter,
    build_trainer_adapter,
    register_trainer_adapter,
)

__all__ = [
    "OptimizerTrainerAdapter",
    "TrainerAdapter",
    "TrainerStepResult",
    "build_grouped_trainer_adapter",
    "build_trainer_adapter",
    "register_trainer_adapter",
]
