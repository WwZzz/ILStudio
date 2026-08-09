"""Thin policy-facing contracts for reinforcement learning."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Optional

from benchmark.base import MetaAction, MetaObs
from rl.base import PolicyOutput


class BasePolicyAdapter(ABC):
    """Expose policy-specific RL operations without owning the RL loop.

    The adapter keeps model inputs, action construction, token metadata and
    train-time forward calls policy-specific.  Optimizer stepping and rollout
    scheduling deliberately live outside this class.
    """

    STATE_VERSION = 1

    def __init__(self, policy: Any, *, capabilities: Iterable[str] = ("action",)):
        capabilities = frozenset(capabilities)
        if not all(isinstance(item, str) and item for item in capabilities):
            raise TypeError("policy adapter capabilities must be non-empty strings")
        if "action" not in capabilities:
            raise ValueError("policy adapter must provide the action capability")
        self.policy = policy
        self._capabilities = capabilities
        self._policy_version = 0

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def policy_version(self) -> int:
        return self._policy_version

    def require_capabilities(self, required: Iterable[str]) -> None:
        missing = set(required) - set(self.capabilities)
        if missing:
            available = ", ".join(sorted(self.capabilities))
            raise ValueError(
                "policy adapter is missing required capabilities: "
                f"{', '.join(sorted(missing))}; available: {available}"
            )

    def bump_policy_version(self) -> int:
        self._policy_version += 1
        return self._policy_version

    def parameters(self):
        """Return adapter-owned trainable parameters, excluding the policy.

        Generic adapters own no additional parameters. Policy-local adapters
        may override this hook for value, Q, temperature, or similar heads.
        """
        return ()

    def set_training(self, training: bool) -> None:
        method_name = "train" if training else "eval"
        method = getattr(self.policy, method_name, None)
        if callable(method):
            method()

    @abstractmethod
    def select_action(
        self,
        obs: MetaObs,
        *,
        deterministic: bool = False,
        context: Optional[Mapping[str, Any]] = None,
    ) -> PolicyOutput:
        """Produce one ILStudio action plus algorithm-specific metadata."""

    def select_actions(
        self,
        observations,
        *,
        deterministic: bool = False,
        context: Optional[Mapping[str, Any]] = None,
    ):
        """Produce a batch of actions, with a safe sequential fallback.

        Policy-local adapters should override this method when their native
        model can batch inference. The fallback preserves compatibility for
        existing policies while allowing parallel collectors to share one
        executor contract.
        """

        observations = tuple(observations)
        if not observations:
            raise ValueError("observations cannot be empty")
        for obs in observations:
            self._validate_obs(obs)
        return tuple(
            self.select_action(
                obs,
                deterministic=deterministic,
                context=context,
            )
            for obs in observations
        )

    def training_forward(
        self,
        batch: Any,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        raise NotImplementedError(
            "this policy adapter does not implement training_forward"
        )

    def algorithm_forward(
        self,
        operation: str,
        batch: Any,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Run one named algorithm-facing policy operation.

        Concrete adapters own observation/action tensorization and model-head
        selection. Algorithms own return, target and loss construction.
        """

        del operation, batch, context
        raise NotImplementedError(
            "this policy adapter does not implement algorithm_forward"
        )

    def algorithm_post_step(
        self,
        operation: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Run policy-specific maintenance such as a target-network update."""

        del operation, context
        raise NotImplementedError(
            "this policy adapter does not implement algorithm_post_step"
        )

    def critic_features(
        self,
        obs: Any,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Return policy-specific visual/language features for a critic.

        Policy-local ``rl_adapter.py`` implementations may override this hook.
        The generic policy-feature critic detaches returned actor features by
        default, so critic optimization cannot silently update the policy.
        """

        del obs, context
        raise NotImplementedError(
            "this policy adapter does not expose critic features"
        )

    def save_pretrained(self, output_dir):
        """Delegate policy persistence without owning checkpoint scheduling."""

        save = getattr(self.policy, "save_pretrained", None)
        if not callable(save):
            raise TypeError(
                "policy must provide save_pretrained() or override the adapter hook"
            )
        return save(output_dir)

    def _validate_obs(self, obs: MetaObs) -> None:
        if not isinstance(obs, MetaObs):
            raise TypeError("policy adapter observation must be MetaObs")

    def _finalize_output(self, output: Any) -> PolicyOutput:
        if isinstance(output, PolicyOutput):
            action = output.action
            policy_info = dict(output.policy_info)
        elif isinstance(output, MetaAction):
            action = output
            policy_info = {}
        elif isinstance(output, Mapping) and "action" in output:
            action = output["action"]
            policy_info = dict(output.get("policy_info", {}))
            policy_info.update(
                {
                    key: value
                    for key, value in output.items()
                    if key not in {"action", "policy_info"}
                }
            )
            if not isinstance(action, MetaAction):
                raise TypeError("mapping policy output action must be MetaAction")
        else:
            raise TypeError(
                "policy output must be PolicyOutput, MetaAction, or an action mapping"
            )

        policy_info.setdefault("policy_version", self.policy_version)
        return PolicyOutput(action=action, policy_info=policy_info)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "policy_version": self.policy_version,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("policy adapter state must be a mapping")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported policy adapter state version")
        policy_version = state.get("policy_version")
        if not isinstance(policy_version, int) or policy_version < 0:
            raise ValueError("policy_version must be a non-negative integer")
        self._policy_version = policy_version
