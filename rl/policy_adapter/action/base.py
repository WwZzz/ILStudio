"""Action semantics composed into the default policy adapter."""

from abc import ABC, abstractmethod
from collections.abc import Mapping

from benchmark.base import MetaObs, MetaPolicy


class ActionAdapter(ABC):
    """Translate policy outputs into RL actions, likelihoods and traces."""

    STATE_VERSION = 1

    def __init__(self, meta_policy: MetaPolicy, *, capabilities):
        if not isinstance(meta_policy, MetaPolicy):
            raise TypeError("action adapter requires benchmark.MetaPolicy")
        self.meta_policy = meta_policy
        self.policy = meta_policy.policy
        self.capabilities = frozenset(capabilities)
        if "action" not in self.capabilities:
            raise ValueError("action adapter must provide action capability")

    @abstractmethod
    def select_action(self, obs, *, deterministic=False, context=None):
        """Return a MetaAction, PolicyOutput, or compatible action mapping."""

    def select_actions(self, observations, *, deterministic=False, context=None):
        observations = tuple(observations)
        if not observations:
            raise ValueError("observations cannot be empty")
        return tuple(
            self.select_action(obs, deterministic=deterministic, context=context)
            for obs in observations
        )

    def parameters(self):
        return ()

    def evaluate_actions(self, batch, *, context=None):
        del batch, context
        raise NotImplementedError("action adapter cannot evaluate stored actions")

    def sample_actions(
        self, batch, *, source="obs", deterministic=False, policy=None, context=None
    ):
        del batch, source, deterministic, policy, context
        raise NotImplementedError("action adapter cannot sample differentiable actions")

    def batch_actions(self, batch, *, context=None):
        del batch, context
        raise NotImplementedError("action adapter cannot tensorize stored actions")

    def uniform_actions(self, batch, *, num_samples, context=None):
        del batch, num_samples, context
        raise NotImplementedError("action adapter cannot sample its action space")

    def action_scores(self, batch, *, source="obs", policy=None, context=None):
        del batch, source, policy, context
        raise NotImplementedError("action adapter cannot compute action scores")

    def clamp_actions(self, actions):
        del actions
        raise NotImplementedError("action adapter has no continuous action bounds")

    def recompute_traces(self, batch, *, context=None):
        del batch, context
        raise NotImplementedError("action adapter cannot recompute traces")

    def training_forward(self, batch, *, context=None):
        del batch, context
        raise NotImplementedError("action adapter has no native training forward")

    def feature_forward(self, observations, *, context=None):
        del observations, context
        raise NotImplementedError("action adapter has no feature forward")

    def critic_features(self, observations, *, context=None):
        del observations, context
        raise NotImplementedError("action adapter exposes no critic features")

    def collection_context(self):
        from contextlib import nullcontext

        return nullcontext()

    def reset(self):
        self.meta_policy.reset()

    def save_assets(self, output_dir, *, checkpoint_path=None):
        del output_dir, checkpoint_path

    def state_dict(self):
        return {"version": self.STATE_VERSION}

    def load_state_dict(self, state):
        if not isinstance(state, Mapping) or state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported action adapter state")

    @staticmethod
    def validate_observation(obs):
        if not isinstance(obs, MetaObs):
            raise TypeError("action adapter observation must be MetaObs")


__all__ = ["ActionAdapter"]
