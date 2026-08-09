"""RL operations for the state-only ILStudio MLP policy."""

from __future__ import annotations

import copy
import inspect
import math
import shutil
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from benchmark.base import MetaAction, MetaObs
from rl.base import PolicyOutput
from rl.policy_adapter import BasePolicyAdapter, BasicTrainerAdapter


_ALGORITHMS = {
    "reinforce",
    "actor_critic",
    "ppo",
    "dqn",
    "sarsa",
    "ddpg",
    "sac",
}
_TARGET_ALGORITHMS = {"dqn", "ddpg", "sac"}
_TARGET_POLICY_ALGORITHMS = {"dqn", "ddpg"}
_CONTINUOUS_ALGORITHMS = {"ddpg", "sac"}
_DISCRETE_ALGORITHMS = {"dqn", "sarsa"}
_STATE_FILENAME = "rl_adapter.pt"


class _StateNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        activation: str = "tanh",
    ):
        super().__init__()
        try:
            activation_class = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        except KeyError as exc:
            raise ValueError("critic_activation must be relu or tanh") from exc
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_class(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_class(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value):
        return self.net(value)


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.lerp_(source_parameter, tau)


def _as_bound(value, *, action_dim: int, default: float, name: str):
    if value is None:
        value = [default] * action_dim
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 1:
        array = np.repeat(array, action_dim)
    if array.size != action_dim:
        raise ValueError(f"{name} must contain {action_dim} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


class MLPRLPolicyAdapter(BasePolicyAdapter):
    """Give ``MLPPolicy`` the operations requested by one RL algorithm.

    The supervised MLP remains untouched. Auxiliary value/Q/target networks
    live here and are saved to ``rl_adapter.pt`` beside ``save_pretrained``
    output, so existing MLP checkpoints retain their original parameter keys.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        policy,
        *,
        required_capabilities: Iterable[str],
        action_space: Optional[str] = None,
        action_low=None,
        action_high=None,
        policy_std: float = 0.2,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        critic_hidden_dim: Optional[int] = None,
        critic_activation: str = "tanh",
        actor_learning_rate: Optional[float] = None,
        critic_learning_rate: Optional[float] = None,
        alpha_learning_rate: Optional[float] = None,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
        seed: Optional[int] = None,
        checkpoint_path=None,
        **unused,
    ):
        del unused
        capabilities = set(required_capabilities)
        algorithms = capabilities.intersection(_ALGORITHMS)
        if len(algorithms) != 1:
            raise ValueError(
                "MLP RL adapter requires exactly one algorithm capability, got "
                + ", ".join(sorted(algorithms or capabilities))
            )
        self.algorithm = next(iter(algorithms))
        capabilities.add("action")
        if self.algorithm in _TARGET_ALGORITHMS:
            capabilities.add("target_update")
        super().__init__(policy, capabilities=capabilities)

        config = policy.config
        self.state_dim = int(config.state_dim)
        self.action_dim = int(config.action_dim)
        if int(getattr(config, "chunk_size", 1)) != 1:
            raise ValueError("MLP RL validation currently requires chunk_size=1")

        configured_space = getattr(config, "rl_action_space", None)
        action_space = action_space or configured_space
        if self.algorithm in _DISCRETE_ALGORITHMS:
            action_space = action_space or "discrete"
        elif self.algorithm in _CONTINUOUS_ALGORITHMS:
            action_space = action_space or "continuous"
        else:
            action_space = action_space or "continuous"
        if action_space not in {"discrete", "continuous"}:
            raise ValueError("action_space must be discrete or continuous")
        if self.algorithm in _DISCRETE_ALGORITHMS and action_space != "discrete":
            raise ValueError(f"{self.algorithm} requires discrete MLP actions")
        if self.algorithm in _CONTINUOUS_ALGORITHMS and action_space != "continuous":
            raise ValueError(f"{self.algorithm} requires continuous MLP actions")
        if action_space == "discrete" and self.action_dim < 2:
            raise ValueError("a discrete MLP needs action_dim equal to num_actions")
        self.action_space = action_space

        device = next(policy.parameters()).device
        if critic_activation not in {"relu", "tanh"}:
            raise ValueError("critic_activation must be relu or tanh")
        critic_hidden_dim = int(critic_hidden_dim or config.hidden_dim)
        learning_rates = {}
        for name, value in (
            ("actor", actor_learning_rate),
            ("critic", critic_learning_rate),
            ("alpha", alpha_learning_rate),
        ):
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"{name}_learning_rate must be positive")
            learning_rates[name] = None if value is None else float(value)
        self.actor_learning_rate = learning_rates["actor"]
        self.critic_learning_rate = learning_rates["critic"]
        self.alpha_learning_rate = learning_rates["alpha"]
        self._modules = {}
        self.log_std = None
        self.log_alpha = None

        if self.action_space == "continuous":
            low = action_low
            high = action_high
            if low is None:
                low = getattr(config, "action_low", None)
            if high is None:
                high = getattr(config, "action_high", None)
            low = _as_bound(low, action_dim=self.action_dim, default=-1.0, name="action_low")
            high = _as_bound(high, action_dim=self.action_dim, default=1.0, name="action_high")
            if np.any(high <= low):
                raise ValueError("every action_high must exceed action_low")
            self.action_low = torch.as_tensor(low, device=device)
            self.action_high = torch.as_tensor(high, device=device)
            self.action_scale = (self.action_high - self.action_low) / 2.0
            self.action_mid = (self.action_high + self.action_low) / 2.0
            if policy_std <= 0:
                raise ValueError("policy_std must be positive")
            self.log_std = nn.Parameter(
                torch.full(
                    (self.action_dim,),
                    math.log(float(policy_std)),
                    device=device,
                )
            )
            self.log_std_min = float(log_std_min)
            self.log_std_max = float(log_std_max)
            if self.log_std_max <= self.log_std_min:
                raise ValueError("log_std_max must exceed log_std_min")
        else:
            self.action_low = None
            self.action_high = None
            self.action_scale = None
            self.action_mid = None

        if self.algorithm in {"actor_critic", "ppo"}:
            self._add_module(
                "value",
                _StateNetwork(
                    self.state_dim,
                    critic_hidden_dim,
                    activation=critic_activation,
                ).to(device),
            )
        if self.algorithm in {"ddpg", "sac"}:
            q_input_dim = self.state_dim + self.action_dim
            self._add_module(
                "q1",
                _StateNetwork(
                    q_input_dim,
                    critic_hidden_dim,
                    activation=critic_activation,
                ).to(device),
            )
            self._add_module(
                "target_q1",
                _freeze(copy.deepcopy(self._modules["q1"])),
            )
        if self.algorithm == "sac":
            self._add_module(
                "q2",
                _StateNetwork(
                    self.state_dim + self.action_dim,
                    critic_hidden_dim,
                    activation=critic_activation,
                ).to(device),
            )
            self._add_module(
                "target_q2",
                _freeze(copy.deepcopy(self._modules["q2"])),
            )
            if "temperature" in capabilities:
                self.log_alpha = nn.Parameter(torch.zeros((), device=device))
        if self.algorithm in _TARGET_POLICY_ALGORITHMS:
            self._add_module(
                "target_policy",
                _freeze(copy.deepcopy(policy)),
            )

        for name in ("epsilon_start", "epsilon_end"):
            value = float(locals()[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.action_steps = 0
        self._rng = np.random.default_rng(seed)
        self.checkpoint_path = checkpoint_path

    def _add_module(self, name: str, module: nn.Module) -> None:
        self._modules[name] = module

    def _module(self, name: str) -> nn.Module:
        try:
            return self._modules[name]
        except KeyError as exc:
            raise RuntimeError(
                f"MLP {self.algorithm} adapter has no {name} module"
            ) from exc

    def parameters(self):
        """Return trainable auxiliary parameters, excluding policy parameters."""

        values = []
        for name, module in self._modules.items():
            if not name.startswith("target_"):
                values.extend(module.parameters())
        if self.log_std is not None:
            values.append(self.log_std)
        if self.log_alpha is not None:
            values.append(self.log_alpha)
        return tuple(parameter for parameter in values if parameter.requires_grad)

    def actor_parameters(self):
        values = list(self.policy.parameters())
        if self.log_std is not None and self.algorithm != "ddpg":
            values.append(self.log_std)
        return tuple(parameter for parameter in values if parameter.requires_grad)

    def critic_parameters(self, index: int = 1):
        return tuple(self._module(f"q{index}").parameters())

    def alpha_parameters(self):
        return () if self.log_alpha is None else (self.log_alpha,)

    def set_training(self, training: bool) -> None:
        super().set_training(training)
        for name, module in self._modules.items():
            if name.startswith("target_"):
                module.eval()
            else:
                module.train(training)

    def _state_tensor(self, obs: MetaObs):
        self._validate_obs(obs)
        if obs.state is None:
            raise ValueError("MLP RL adapter requires MetaObs.state")
        device = next(self.policy.parameters()).device
        state = torch.as_tensor(obs.state, dtype=torch.float32, device=device)
        state = state.reshape(1, -1)
        if state.shape[1] != self.state_dim:
            raise ValueError(
                f"MLP expected state_dim={self.state_dim}, got {state.shape[1]}"
            )
        return state

    def _batch_states(self, batch, *, next_state=False):
        attribute = "next_obs" if next_state else "obs"
        states = [np.asarray(getattr(item, attribute).state) for item in batch]
        device = next(self.policy.parameters()).device
        result = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        if result.ndim != 2 or result.shape[1] != self.state_dim:
            raise ValueError("MLP transition states have an incompatible shape")
        return result

    def _batch_actions(self, batch):
        actions = [np.asarray(item.action.action).reshape(-1) for item in batch]
        device = next(self.policy.parameters()).device
        if self.action_space == "discrete":
            values = [int(action[0]) for action in actions]
            return torch.as_tensor(values, dtype=torch.long, device=device)
        result = torch.as_tensor(np.stack(actions), dtype=torch.float32, device=device)
        if result.shape != (len(actions), self.action_dim):
            raise ValueError("MLP transition actions have an incompatible shape")
        return result

    @staticmethod
    def _raw_policy_output(policy, states):
        output = policy(states)["action"]
        if output.ndim != 3 or output.shape[1] != 1:
            raise ValueError("MLP RL policy output must have shape [batch, 1, dim]")
        return output[:, 0]

    def _continuous_sample(self, raw_mean, *, deterministic=False):
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(raw_mean)
        normal = Normal(raw_mean, std)
        if deterministic:
            pre_tanh = raw_mean
        else:
            pre_tanh = normal.rsample()
        unit_action = torch.tanh(pre_tanh)
        action = self.action_mid + self.action_scale * unit_action
        correction = torch.log(
            self.action_scale * (1.0 - unit_action.square()) + 1e-6
        )
        log_prob = (normal.log_prob(pre_tanh) - correction).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def _continuous_log_prob(self, raw_mean, action):
        unit_action = (action - self.action_mid) / self.action_scale
        unit_action = unit_action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(unit_action)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        normal = Normal(raw_mean, log_std.exp().expand_as(raw_mean))
        correction = torch.log(
            self.action_scale * (1.0 - unit_action.square()) + 1e-6
        )
        log_prob = (normal.log_prob(pre_tanh) - correction).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        return log_prob, entropy

    def _deterministic_continuous(self, policy, states):
        raw = self._raw_policy_output(policy, states)
        return self.action_mid + self.action_scale * torch.tanh(raw)

    def _epsilon(self):
        fraction = min(1.0, self.action_steps / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (
            self.epsilon_end - self.epsilon_start
        )

    def select_action(self, obs, *, deterministic=False, context=None):
        del context
        state = self._state_tensor(obs)
        with torch.no_grad():
            raw = self._raw_policy_output(self.policy, state)
            policy_info = {}
            if self.action_space == "discrete":
                if self.algorithm in _DISCRETE_ALGORITHMS:
                    epsilon = 0.0 if deterministic else self._epsilon()
                    if self._rng.random() < epsilon:
                        action_value = int(self._rng.integers(self.action_dim))
                    else:
                        action_value = int(raw.argmax(dim=-1).item())
                    policy_info["epsilon"] = epsilon
                else:
                    distribution = Categorical(logits=raw)
                    action_tensor = (
                        raw.argmax(dim=-1)
                        if deterministic
                        else distribution.sample()
                    )
                    action_value = int(action_tensor.item())
                    policy_info.update(
                        {
                            "log_prob": float(
                                distribution.log_prob(action_tensor).item()
                            ),
                            "entropy": float(distribution.entropy().item()),
                        }
                    )
                action = MetaAction(
                    ctrl_space="gym",
                    ctrl_type="discrete",
                    action=np.asarray([action_value], dtype=np.int64),
                )
            else:
                if self.algorithm == "ddpg":
                    action_tensor = self.action_mid + self.action_scale * torch.tanh(raw)
                    if not deterministic:
                        std = self.log_std.clamp(
                            self.log_std_min, self.log_std_max
                        ).exp()
                        action_tensor = action_tensor + torch.randn_like(
                            action_tensor
                        ) * std
                        action_tensor = torch.maximum(
                            torch.minimum(action_tensor, self.action_high),
                            self.action_low,
                        )
                else:
                    action_tensor, log_prob, entropy = self._continuous_sample(
                        raw,
                        deterministic=deterministic,
                    )
                    policy_info.update(
                        {
                            "log_prob": float(log_prob.item()),
                            "entropy": float(entropy.item()),
                        }
                    )
                action = MetaAction(
                    ctrl_space="gym",
                    ctrl_type="continuous",
                    action=action_tensor.cpu().numpy().reshape(1, -1),
                )
            if self.algorithm in {"actor_critic", "ppo"}:
                policy_info["value"] = float(self._module("value")(state).item())
        self.action_steps += 1
        return self._finalize_output(
            PolicyOutput(action=action, policy_info=policy_info)
        )

    def _policy_terms(self, states, actions):
        raw = self._raw_policy_output(self.policy, states)
        if self.action_space == "discrete":
            distribution = Categorical(logits=raw)
            return {
                "log_prob": distribution.log_prob(actions),
                "entropy": distribution.entropy(),
            }
        log_prob, entropy = self._continuous_log_prob(raw, actions)
        return {"log_prob": log_prob, "entropy": entropy}

    def _q(self, name, states, actions):
        return self._module(name)(torch.cat([states, actions], dim=-1)).squeeze(-1)

    def algorithm_forward(self, operation, batch, *, context=None):
        if operation != self.algorithm:
            raise ValueError(
                f"MLP adapter configured for {self.algorithm}, got {operation}"
            )
        phase = dict(context or {}).get("phase")
        states = self._batch_states(batch)
        actions = self._batch_actions(batch)
        next_states = self._batch_states(batch, next_state=True)

        if operation in {"reinforce", "actor_critic", "ppo"}:
            result = self._policy_terms(states, actions)
            if operation in {"actor_critic", "ppo"}:
                result["value"] = self._module("value")(states).squeeze(-1)
                with torch.no_grad():
                    result["next_value"] = self._module("value")(
                        next_states
                    ).squeeze(-1)
            return result
        if operation == "dqn":
            q_values = self._raw_policy_output(self.policy, states)
            with torch.no_grad():
                target_next = self._raw_policy_output(
                    self._module("target_policy"), next_states
                )
                online_next = self._raw_policy_output(self.policy, next_states)
            return {
                "q_values": q_values,
                "target_next_q_values": target_next,
                "online_next_q_values": online_next,
            }
        if operation == "sarsa":
            q_values = self._raw_policy_output(self.policy, states)
            next_q_values = self._raw_policy_output(self.policy, next_states)
            next_actions = []
            for index, item in enumerate(batch):
                if item.terminated:
                    next_action = 0
                elif not item.episode_done and index + 1 < len(batch):
                    next_action = int(actions[index + 1])
                elif self._rng.random() < self._epsilon():
                    next_action = int(self._rng.integers(self.action_dim))
                else:
                    next_action = int(next_q_values[index].argmax())
                next_actions.append(next_action)
            return {
                "q_values": q_values,
                "next_q_values": next_q_values,
                # Internal rollout steps reuse the action that was actually
                # sampled next. A truncated or incomplete tail has no stored
                # successor action, so sample one from the same epsilon-greedy
                # behaviour policy for the required time-limit bootstrap.
                "next_action": torch.as_tensor(
                    next_actions,
                    dtype=torch.long,
                    device=q_values.device,
                ),
            }
        if operation == "ddpg":
            actor_actions = self._deterministic_continuous(self.policy, states)
            actor_q = self._q("q1", states, actor_actions)
            if phase == "actor":
                return {"actor_q": actor_q}
            with torch.no_grad():
                target_actions = self._deterministic_continuous(
                    self._module("target_policy"), next_states
                )
                target_next_q = self._q("target_q1", next_states, target_actions)
            return {
                "critic_q": self._q("q1", states, actions),
                "target_next_q": target_next_q,
                "actor_q": actor_q,
            }
        if operation == "sac":
            actor_raw = self._raw_policy_output(self.policy, states)
            actor_actions, actor_log_prob, _ = self._continuous_sample(actor_raw)
            actor_result = {
                "actor_q1": self._q("q1", states, actor_actions),
                "actor_q2": self._q("q2", states, actor_actions),
                "actor_log_prob": actor_log_prob,
            }
            if phase == "actor":
                return actor_result
            with torch.no_grad():
                next_raw = self._raw_policy_output(self.policy, next_states)
                next_actions, next_log_prob, _ = self._continuous_sample(next_raw)
                target_next_q1 = self._q("target_q1", next_states, next_actions)
                target_next_q2 = self._q("target_q2", next_states, next_actions)
            result = {
                "q1": self._q("q1", states, actions),
                "q2": self._q("q2", states, actions),
                "target_next_q1": target_next_q1,
                "target_next_q2": target_next_q2,
                "next_log_prob": next_log_prob,
                **actor_result,
            }
            if self.log_alpha is not None:
                result["log_alpha"] = self.log_alpha
            return result
        raise NotImplementedError(operation)

    def algorithm_post_step(self, operation, *, context=None):
        tau = float(dict(context or {}).get("tau", 1.0))
        if not 0.0 < tau <= 1.0:
            raise ValueError("target update tau must be in (0, 1]")
        if operation == "dqn_target":
            _soft_update(self._module("target_policy"), self.policy, tau)
        elif operation == "ddpg_target":
            _soft_update(self._module("target_policy"), self.policy, tau)
            _soft_update(self._module("target_q1"), self._module("q1"), tau)
        elif operation == "sac_target":
            _soft_update(self._module("target_q1"), self._module("q1"), tau)
            _soft_update(self._module("target_q2"), self._module("q2"), tau)
        else:
            raise ValueError(f"unsupported MLP target operation {operation!r}")

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "base": super().state_dict(),
            "algorithm": self.algorithm,
            "action_space": self.action_space,
            "action_steps": self.action_steps,
            "modules": {
                name: module.state_dict() for name, module in self._modules.items()
            },
            "log_std": (
                None if self.log_std is None else self.log_std.detach().cpu()
            ),
            "log_alpha": (
                None if self.log_alpha is None else self.log_alpha.detach().cpu()
            ),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping):
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported MLP RL adapter state version")
        if state.get("algorithm") != self.algorithm:
            raise ValueError("MLP RL adapter algorithm does not match state")
        if state.get("action_space") != self.action_space:
            raise ValueError("MLP RL adapter action space does not match state")
        super().load_state_dict(state["base"])
        module_states = state.get("modules", {})
        if set(module_states) != set(self._modules):
            raise ValueError("MLP RL auxiliary module state does not match adapter")
        for name, module in self._modules.items():
            module.load_state_dict(module_states[name])
        for name in ("log_std", "log_alpha"):
            saved = state.get(name)
            parameter = getattr(self, name)
            if (saved is None) != (parameter is None):
                raise ValueError(f"MLP RL {name} state does not match adapter")
            if parameter is not None:
                parameter.data.copy_(saved.to(parameter.device))
        self.action_steps = int(state.get("action_steps", 0))
        self._rng.bit_generator.state = state["rng_state"]

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = self.policy.save_pretrained(output_dir)
        torch.save(self.state_dict(), output_dir / _STATE_FILENAME)
        if self.checkpoint_path is not None:
            source_root = Path(self.checkpoint_path)
            if source_root.name.startswith("checkpoint-"):
                source_root = source_root.parent
            source_metadata = source_root / "policy_metadata.json"
            if not source_metadata.is_file():
                raise FileNotFoundError(
                    f"checkpoint metadata was not found: {source_metadata}"
                )
            target_metadata = output_dir / source_metadata.name
            if source_metadata.resolve() != target_metadata.resolve():
                shutil.copy2(source_metadata, target_metadata)
        return result


def _load_adapter_state(adapter, checkpoint_path):
    if not checkpoint_path:
        return
    path = Path(checkpoint_path) / _STATE_FILENAME
    if not path.is_file():
        return
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    adapter.load_state_dict(state)


def build_rl_adapter(
    *,
    model_components,
    required_capabilities=(),
    **kwargs,
):
    kwargs.setdefault("checkpoint_path", model_components.get("checkpoint_path"))
    adapter = MLPRLPolicyAdapter(
        model_components["model"],
        required_capabilities=required_capabilities,
        **kwargs,
    )
    _load_adapter_state(adapter, model_components.get("checkpoint_path"))
    return adapter


def _clone_optimizer(template, parameters, *, learning_rate=None):
    parameters = tuple(parameters)
    if not parameters:
        raise ValueError("MLP trainer adapter received an empty parameter group")
    optimizer_class = type(template)
    defaults = dict(template.defaults)
    if learning_rate is not None:
        defaults["lr"] = float(learning_rate)
    try:
        signature = inspect.signature(optimizer_class.__init__)
    except (TypeError, ValueError):
        optimizer_args = defaults
    else:
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        optimizer_args = (
            defaults
            if accepts_kwargs
            else {
                name: value
                for name, value in defaults.items()
                if name in signature.parameters
            }
        )
    return optimizer_class(parameters, **optimizer_args)


def build_trainer_adapter(
    *,
    policy_components,
    policy_adapter=None,
    optimizer=None,
    scheduler=None,
    step_fn=None,
    **kwargs,
):
    del policy_components
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported MLP trainer adapter arguments: {unknown}")
    if not isinstance(policy_adapter, MLPRLPolicyAdapter):
        raise TypeError("MLP trainer adapter requires MLPRLPolicyAdapter")
    if optimizer is None:
        raise ValueError("MLP trainer adapter requires the configured optimizer")
    if step_fn is not None:
        raise ValueError("MLP trainer adapter does not accept a custom step_fn")
    if policy_adapter.algorithm == "ddpg":
        if scheduler is not None:
            raise ValueError("DDPG requires per-optimizer schedulers")
        optimizers = {
            "critic": _clone_optimizer(
                optimizer,
                policy_adapter.critic_parameters(1),
                learning_rate=policy_adapter.critic_learning_rate,
            ),
            "actor": _clone_optimizer(
                optimizer,
                policy_adapter.actor_parameters(),
                learning_rate=policy_adapter.actor_learning_rate,
            ),
        }
        return BasicTrainerAdapter(optimizer=optimizers)
    if policy_adapter.algorithm == "sac":
        if scheduler is not None:
            raise ValueError("SAC requires per-optimizer schedulers")
        optimizers = {
            "critic1": _clone_optimizer(
                optimizer,
                policy_adapter.critic_parameters(1),
                learning_rate=policy_adapter.critic_learning_rate,
            ),
            "critic2": _clone_optimizer(
                optimizer,
                policy_adapter.critic_parameters(2),
                learning_rate=policy_adapter.critic_learning_rate,
            ),
            "actor": _clone_optimizer(
                optimizer,
                policy_adapter.actor_parameters(),
                learning_rate=policy_adapter.actor_learning_rate,
            ),
        }
        if policy_adapter.log_alpha is not None:
            optimizers["alpha"] = _clone_optimizer(
                optimizer,
                policy_adapter.alpha_parameters(),
                learning_rate=policy_adapter.alpha_learning_rate,
            )
        return BasicTrainerAdapter(optimizer=optimizers)
    return BasicTrainerAdapter(optimizer=optimizer, scheduler=scheduler)


RLPolicyAdapter = MLPRLPolicyAdapter

__all__ = [
    "MLPRLPolicyAdapter",
    "RLPolicyAdapter",
    "build_rl_adapter",
    "build_trainer_adapter",
]
