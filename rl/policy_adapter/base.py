"""Meta-level RL policy contract and its default composed implementation."""

import json
import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from benchmark.base import MetaAction, MetaObs, MetaPolicy
from rl.base import PolicyOutput

from .action import build_action_adapter


_STATE_FILENAME = "rl_adapter.pt"


class MetaPolicyAdapter(ABC):
    """Stable MetaObs/MetaAction boundary used by algorithms and runners.

    This is the shared interface. ``BasePolicyAdapter`` below is the default
    implementation and delegates action semantics to a configured
    ``ActionAdapter``.
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
        self.checkpoint_path = None

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
        """Return adapter-owned parameters, excluding policy parameters."""

        return ()

    def set_training(self, training: bool) -> None:
        method = getattr(self.policy, "train" if training else "eval", None)
        if callable(method):
            method()

    def collection_context(self):
        return nullcontext()

    @abstractmethod
    def select_action(
        self,
        obs: MetaObs,
        *,
        deterministic: bool = False,
        context: Optional[Mapping[str, Any]] = None,
    ) -> PolicyOutput:
        """Produce one ILStudio action and its collection metadata."""

    def select_actions(
        self,
        observations,
        *,
        deterministic: bool = False,
        context: Optional[Mapping[str, Any]] = None,
    ):
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

    def training_forward(self, batch, *, context=None) -> Mapping[str, Any]:
        del batch, context
        raise NotImplementedError(
            "this policy adapter does not implement training_forward"
        )

    def evaluate_actions(self, batch, *, context=None) -> Mapping[str, Any]:
        del batch, context
        raise NotImplementedError("this policy adapter cannot evaluate actions")

    def sample_actions(
        self,
        batch,
        *,
        source="obs",
        deterministic=False,
        policy=None,
        context=None,
    ) -> Mapping[str, Any]:
        del batch, source, deterministic, policy, context
        raise NotImplementedError("this policy adapter cannot sample action batches")

    def batch_actions(self, batch, *, context=None):
        del batch, context
        raise NotImplementedError("this policy adapter cannot tensorize action batches")

    def uniform_actions(self, batch, *, num_samples, context=None):
        del batch, num_samples, context
        raise NotImplementedError("this policy adapter cannot sample its action space")

    def action_scores(self, batch, *, source="obs", policy=None, context=None):
        del batch, source, policy, context
        raise NotImplementedError("this policy adapter cannot compute action scores")

    def clamp_actions(self, actions):
        del actions
        raise NotImplementedError("this policy adapter has no continuous bounds")

    def features(self, observations, *, context=None) -> Mapping[str, Any]:
        return self.critic_features(observations, context=context)

    def feature_forward(self, observations, *, context=None):
        del observations, context
        raise NotImplementedError("this policy adapter has no feature forward path")

    def recompute_traces(self, batch, *, context=None) -> Mapping[str, Any]:
        del batch, context
        raise NotImplementedError("this policy adapter cannot recompute traces")

    def critic_features(self, obs, *, context=None) -> Mapping[str, Any]:
        del obs, context
        raise NotImplementedError(
            "this policy adapter does not expose critic features"
        )

    def set_checkpoint_source(self, checkpoint_path):
        if checkpoint_path is None:
            return self
        checkpoint_path = Path(checkpoint_path)
        current = getattr(self, "checkpoint_path", None)
        if current is not None and Path(current).resolve() != checkpoint_path.resolve():
            raise ValueError(
                "policy adapter checkpoint source does not match model components"
            )
        self.checkpoint_path = checkpoint_path
        return self

    @staticmethod
    def _checkpoint_root(checkpoint_path):
        root = Path(checkpoint_path)
        return root.parent if root.name.startswith("checkpoint-") else root

    def _copy_policy_metadata(self, output_dir):
        if self.checkpoint_path is None:
            return
        source = self._checkpoint_root(self.checkpoint_path) / "policy_metadata.json"
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint metadata was not found: {source}")
        destination = Path(output_dir) / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

    def _copy_checkpoint_assets(self, output_dir):
        self._copy_policy_metadata(output_dir)
        if self.checkpoint_path is None:
            return
        source_root = self._checkpoint_root(self.checkpoint_path)
        output_root = Path(output_dir)
        normalize_path = source_root / "normalize.json"
        if not normalize_path.is_file():
            return
        destination = output_root / normalize_path.name
        if normalize_path.resolve() != destination.resolve():
            shutil.copy2(normalize_path, destination)
        with normalize_path.open("r", encoding="utf-8") as stream:
            normalize_config = json.load(stream)
        for dataset in normalize_config.get("datasets", ()):
            dataset_id = dataset.get("dataset_id")
            if not dataset_id:
                continue
            ctrl_space = dataset.get("ctrl_space", "ee")
            ctrl_type = dataset.get("ctrl_type", "delta")
            filename = f"{dataset_id}_stats_{ctrl_space}_{ctrl_type}.pkl"
            source = source_root / filename
            destination = output_root / filename
            if source.is_file() and source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

    def save_pretrained(self, output_dir):
        save = getattr(self.policy, "save_pretrained", None)
        if not callable(save):
            raise TypeError(
                "policy must provide save_pretrained() or override the adapter hook"
            )
        result = save(output_dir)
        self._copy_checkpoint_assets(output_dir)
        return result

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


class BasePolicyAdapter(MetaPolicyAdapter):
    """Uniform RL facade; policy-specific action semantics live in ActionAdapter."""

    STATE_VERSION = 3

    def __init__(
        self,
        meta_policy: MetaPolicy,
        *,
        action_adapter=None,
        required_capabilities=(),
        checkpoint_path=None,
    ):
        if not isinstance(meta_policy, MetaPolicy):
            raise TypeError("BasePolicyAdapter requires benchmark.MetaPolicy")
        if meta_policy.state_normalizer is None:
            raise ValueError("MetaPolicy must provide state_normalizer")
        if meta_policy.action_normalizer is None:
            raise ValueError("MetaPolicy must provide action_normalizer")
        self.meta_policy = meta_policy
        self.action_adapter = build_action_adapter(
            action_adapter,
            meta_policy=meta_policy,
        )
        super().__init__(
            meta_policy.policy,
            capabilities=self.action_adapter.capabilities,
        )
        self.checkpoint_path = checkpoint_path
        self.require_capabilities(required_capabilities)

    @property
    def device(self):
        return next(self.policy.parameters()).device

    def parameters(self):
        return tuple(self.action_adapter.parameters())

    def actor_parameters(self):
        parameters = []
        seen = set()
        for owner in (self.policy, self.action_adapter):
            for parameter in owner.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return tuple(parameters)

    def collection_context(self):
        return self.action_adapter.collection_context()

    def select_action(self, obs, *, deterministic=False, context=None):
        self._validate_obs(obs)
        output = self.action_adapter.select_action(
            obs,
            deterministic=deterministic,
            context=context,
        )
        return self._finalize_output(output)

    def select_actions(self, observations, *, deterministic=False, context=None):
        observations = tuple(observations)
        if not observations:
            raise ValueError("observations cannot be empty")
        for obs in observations:
            self._validate_obs(obs)
        return tuple(
            self._finalize_output(output)
            for output in self.action_adapter.select_actions(
                observations,
                deterministic=deterministic,
                context=context,
            )
        )

    def training_forward(self, batch, *, context=None):
        return self.action_adapter.training_forward(batch, context=context)

    def evaluate_actions(self, batch, *, context=None):
        return self.action_adapter.evaluate_actions(batch, context=context)

    def sample_actions(
        self,
        batch,
        *,
        source="obs",
        deterministic=False,
        policy=None,
        context=None,
    ):
        return self.action_adapter.sample_actions(
            batch,
            source=source,
            deterministic=deterministic,
            policy=policy,
            context=context,
        )

    def batch_actions(self, batch, *, context=None):
        return self.action_adapter.batch_actions(batch, context=context)

    def uniform_actions(self, batch, *, num_samples, context=None):
        return self.action_adapter.uniform_actions(
            batch,
            num_samples=num_samples,
            context=context,
        )

    def action_scores(self, batch, *, source="obs", policy=None, context=None):
        return self.action_adapter.action_scores(
            batch,
            source=source,
            policy=policy,
            context=context,
        )

    def clamp_actions(self, actions):
        return self.action_adapter.clamp_actions(actions)

    def recompute_traces(self, batch, *, context=None):
        return self.action_adapter.recompute_traces(batch, context=context)

    def feature_forward(self, observations, *, context=None):
        return self.action_adapter.feature_forward(observations, context=context)

    def critic_features(self, observations, *, context=None):
        return self.action_adapter.critic_features(observations, context=context)

    def reset(self):
        self.action_adapter.reset()

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "base": super().state_dict(),
            "action_adapter_type": (
                f"{type(self.action_adapter).__module__}."
                f"{type(self.action_adapter).__qualname__}"
            ),
            "action_adapter": self.action_adapter.state_dict(),
        }

    def load_state_dict(self, state):
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported BasePolicyAdapter state version")
        expected = (
            f"{type(self.action_adapter).__module__}."
            f"{type(self.action_adapter).__qualname__}"
        )
        if state.get("action_adapter_type") != expected:
            raise ValueError("checkpoint action adapter type does not match config")
        super().load_state_dict(state["base"])
        self.action_adapter.load_state_dict(state["action_adapter"])

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = super().save_pretrained(output_dir)
        self.action_adapter.save_assets(
            output_dir,
            checkpoint_path=self.checkpoint_path,
        )
        torch.save(self.state_dict(), output_dir / _STATE_FILENAME)
        return result


def build_rl_adapter(*, model_components, required_capabilities=(), **kwargs):
    if "meta_policy" not in model_components:
        raise KeyError("base policy adapter requires model_components.meta_policy")
    adapter = BasePolicyAdapter(
        model_components["meta_policy"],
        required_capabilities=required_capabilities,
        checkpoint_path=model_components.get("checkpoint_path"),
        **kwargs,
    )
    checkpoint_path = model_components.get("checkpoint_path")
    if checkpoint_path is not None:
        state_path = Path(checkpoint_path) / _STATE_FILENAME
        if state_path.is_file():
            adapter.load_state_dict(torch.load(state_path, map_location=adapter.device))
    return adapter


__all__ = ["MetaPolicyAdapter", "BasePolicyAdapter", "build_rl_adapter"]
