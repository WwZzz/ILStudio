"""Reward composition with namespacing, weighting, and collision checks."""

import math
from numbers import Real
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from rl.base import MetaTransition

from .base import BaseReward, RewardContext, RewardDict, _validate_path
from .env import TOTAL_REWARD_KEY


class RewardComposer:
    """Combine raw and derived rewards without mutating source transitions."""

    STATE_VERSION = 1

    def __init__(
        self,
        modules: Optional[Iterable[BaseReward]] = None,
        *,
        weights: Optional[Mapping[str, float]] = None,
        default_weight: float = 1.0,
    ) -> None:
        self.modules = tuple(modules or ())
        for module in self.modules:
            if not isinstance(module, BaseReward):
                raise TypeError("reward modules must inherit BaseReward")

        namespaces = [module.namespace for module in self.modules]
        if len(namespaces) != len(set(namespaces)):
            raise ValueError("reward modules contain a duplicate namespace")

        self.weights = self._validate_weights(weights or {})
        self.default_weight = self._validate_weight(
            default_weight,
            label="default reward weight",
        )

    @staticmethod
    def _validate_weight(value: float, *, label: str) -> float:
        if not isinstance(value, Real):
            raise TypeError(f"{label} must be a real number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
        return value

    @classmethod
    def _validate_weights(cls, weights: Mapping[str, float]) -> Dict[str, float]:
        if not isinstance(weights, Mapping):
            raise TypeError("reward weights must be a mapping")
        validated = {}
        for key, value in weights.items():
            key = cls._validate_full_key(key)
            validated[key] = cls._validate_weight(
                value,
                label=f"reward weight for '{key}'",
            )
        return validated

    @staticmethod
    def _validate_full_key(key: str) -> str:
        key = _validate_path(key, label="reward component key")
        if "/" not in key:
            raise ValueError(f"reward component key must be namespaced: {key}")
        return key

    @classmethod
    def _qualify_module_result(
        cls,
        module: BaseReward,
        result: RewardDict,
    ) -> Dict[str, Any]:
        if not isinstance(result, dict):
            raise TypeError(f"reward module '{module.namespace}' must return a dict")

        qualified = {}
        for local_key, value in result.items():
            local_key = _validate_path(
                local_key,
                label=f"local reward key from '{module.namespace}'",
            )
            full_key = cls._validate_full_key(
                f"{module.namespace}/{local_key}"
            )
            qualified[full_key] = value
        return qualified

    @staticmethod
    def _split_batch_value(value: Any, batch_size: int, *, key: str):
        if batch_size == 1:
            try:
                value_length = len(value)
            except TypeError:
                return (value,)
            if value_length != 1:
                raise ValueError(
                    f"reward component '{key}' has leading size {value_length}, "
                    f"expected batch size 1"
                )
            return (value[0],)

        try:
            value_length = len(value)
        except TypeError as exc:
            raise ValueError(
                f"reward component '{key}' has no leading batch dimension; "
                f"expected batch size {batch_size}"
            ) from exc
        if value_length != batch_size:
            raise ValueError(
                f"reward component '{key}' has leading size {value_length}, "
                f"expected batch size {batch_size}"
            )
        return tuple(value[index] for index in range(batch_size))

    def _add_total(self, components: RewardDict) -> RewardDict:
        result = dict(components)
        result.pop(TOTAL_REWARD_KEY, None)
        total = None
        for key, value in result.items():
            weight = self.weights.get(key, self.default_weight)
            if weight == 0.0:
                continue
            try:
                weighted_value = value * weight
                total = weighted_value if total is None else total + weighted_value
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"reward component '{key}' cannot be weighted and summed"
                ) from exc
        result[TOTAL_REWARD_KEY] = 0.0 if total is None else total
        return result

    def compute_step(
        self,
        transition: MetaTransition,
        *,
        context: RewardContext = None,
    ) -> RewardDict:
        """Compute all reward components for one transition."""

        return self.compute_batch((transition,), context=context)[0]

    def compute_batch(
        self,
        transitions: Sequence[MetaTransition],
        *,
        context: RewardContext = None,
    ) -> Sequence[RewardDict]:
        """Compute modules once per batch and return one dict per transition."""

        transitions = tuple(transitions)
        for transition in transitions:
            if not isinstance(transition, MetaTransition):
                raise TypeError("reward inputs must be MetaTransition")
        if not transitions:
            return []

        component_dicts = []
        for transition in transitions:
            components = {
                self._validate_full_key(key): value
                for key, value in transition.reward.items()
                if key != TOTAL_REWARD_KEY
            }
            component_dicts.append(components)

        for module in self.modules:
            module_result = module.compute_batch(transitions, context=context)
            qualified = self._qualify_module_result(module, module_result)
            for key, batch_value in qualified.items():
                split_values = self._split_batch_value(
                    batch_value,
                    len(transitions),
                    key=key,
                )
                for index, value in enumerate(split_values):
                    if key in component_dicts[index]:
                        raise KeyError(
                            f"reward component '{key}' already exists; "
                            "reward modules may not overwrite components"
                        )
                    component_dicts[index][key] = value

        return [self._add_total(components) for components in component_dicts]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "weights": dict(self.weights),
            "default_weight": self.default_weight,
            "module_states": {
                module.namespace: module.state_dict() for module in self.modules
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("reward composer state must be a dict")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported reward composer state version")

        weights = self._validate_weights(state.get("weights", {}))
        default_weight = self._validate_weight(
            state.get("default_weight"),
            label="default reward weight",
        )
        module_states = state.get("module_states")
        if not isinstance(module_states, dict):
            raise TypeError("reward module_states must be a dict")

        expected_namespaces = {module.namespace for module in self.modules}
        if set(module_states) != expected_namespaces:
            raise ValueError("reward module namespaces do not match checkpoint")
        for module in self.modules:
            module_state = module_states[module.namespace]
            if not isinstance(module_state, dict):
                raise TypeError(
                    f"state for reward module '{module.namespace}' must be a dict"
                )

        for module in self.modules:
            module.load_state_dict(module_states[module.namespace])
        self.weights = weights
        self.default_weight = default_weight
