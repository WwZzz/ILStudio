"""Base interface for environment, learned, and model-based rewards."""

from typing import Any, Dict, Mapping, Optional, Sequence

from rl.base import MetaTransition


RewardDict = Dict[str, Any]
RewardContext = Optional[Mapping[str, Any]]
RESERVED_REWARD_NAMESPACES = frozenset({"env", "train"})


def _validate_path(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if value != value.strip() or not value:
        raise ValueError(f"{label} must be a non-empty stripped path")
    if value.startswith("/") or value.endswith("/") or "//" in value:
        raise ValueError(f"{label} contains an empty path segment: {value}")
    return value


class BaseReward:
    """A namespaced reward source.

    Subclasses may implement ``compute_step`` for simple rewards or override
    ``compute_batch`` for GPU-efficient learned/world-model rewards.  A module
    always returns a dictionary of local component names to reward values.
    """

    def __init__(self, namespace: str) -> None:
        namespace = _validate_path(namespace, label="reward namespace")
        root_namespace = namespace.split("/", 1)[0]
        if root_namespace in RESERVED_REWARD_NAMESPACES:
            raise ValueError(
                f"reward namespace '{root_namespace}' is reserved by ILStudio"
            )
        self.namespace = namespace

    def compute_step(
        self,
        transition: MetaTransition,
        *,
        context: RewardContext = None,
    ) -> RewardDict:
        """Compute local reward components for one transition."""

        del transition, context
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement compute_step or compute_batch"
        )

    def compute_batch(
        self,
        transitions: Sequence[MetaTransition],
        *,
        context: RewardContext = None,
    ) -> RewardDict:
        """Compute a dict whose values have one leading item per transition.

        The default implementation collects ``compute_step`` results.  A model
        reward can override this method and return NumPy arrays or tensors with
        leading dimension ``len(transitions)``.
        """

        transitions = tuple(transitions)
        if not transitions:
            return {}

        step_results = []
        expected_keys = None
        for transition in transitions:
            result = self.compute_step(transition, context=context)
            if not isinstance(result, dict):
                raise TypeError(
                    f"reward module '{self.namespace}' must return a dict"
                )
            keys = tuple(result.keys())
            if expected_keys is None:
                expected_keys = keys
            elif set(keys) != set(expected_keys):
                raise ValueError(
                    f"reward module '{self.namespace}' returned inconsistent keys"
                )
            step_results.append(result)

        return {
            key: [result[key] for result in step_results]
            for key in expected_keys
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return model/module state required for checkpointing."""

        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore model/module state; stateless rewards accept only ``{}``."""

        if not isinstance(state, dict):
            raise TypeError("reward module state must be a dict")
        if state:
            raise ValueError(
                f"stateless reward module '{self.namespace}' received state"
            )
