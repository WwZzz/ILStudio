"""In-process executor for trainable reinforcement-learning policies."""

import copy
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Optional

import numpy as np

from benchmark.base import MetaObs
from deploy.action_manager.chunk import (
    AbstractActionChunkManager,
    BasicActionChunkManager,
)
from rl.base import (
    ACTION_POLICY_INFO_KEY,
    PolicyDecision,
    PolicyOutput,
    PolicyTrace,
)
from rl.policy_adapter import MetaPolicyAdapter

from .base import BasePolicyExecutor


class RLPolicyExecutor(BasePolicyExecutor):
    """Call a policy adapter directly and schedule outputs with ILStudio chunks.

    Unlike EvalPolicyExecutor, this executor does not bind to SHM or an
    inference subprocess. It converts one policy PolicyOutput into a list
    of step-level outputs, then delegates all chunk state and dispatch to the
    shared deploy ActionChunkManager.
    """

    def __init__(
        self,
        adapter: MetaPolicyAdapter,
        *,
        chunk_manager: Optional[AbstractActionChunkManager] = None,
        execution_horizon: Optional[int] = None,
        owns_adapter: bool = True,
    ) -> None:
        if not isinstance(adapter, MetaPolicyAdapter):
            raise TypeError("adapter must inherit MetaPolicyAdapter")
        if execution_horizon is not None and (
            not isinstance(execution_horizon, int)
            or isinstance(execution_horizon, bool)
            or execution_horizon <= 0
        ):
            raise ValueError("execution_horizon must be a positive integer or None")
        if chunk_manager is None:
            chunk_manager = BasicActionChunkManager()
        if not isinstance(chunk_manager, AbstractActionChunkManager):
            raise TypeError(
                "chunk_manager must inherit AbstractActionChunkManager"
            )
        if not isinstance(owns_adapter, bool):
            raise TypeError("owns_adapter must be bool")
        self.adapter = adapter
        self.chunk_manager = chunk_manager
        self.execution_horizon = execution_horizon
        self.owns_adapter = owns_adapter
        self._closed = False
        self._inference_count = 0
        self._next_chunk_id = 0
        self._next_decision_id = 0

    @property
    def inference_count(self) -> int:
        return self._inference_count

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def needs_inference(self) -> bool:
        return self.chunk_manager.is_empty()

    @staticmethod
    def _action_size(action) -> int:
        if action is None:
            raise ValueError("MetaAction.action cannot be None")
        if not hasattr(action, "ndim"):
            action = np.asarray(action)
        ndim = int(action.ndim)
        if ndim == 0:
            raise ValueError("MetaAction.action must have at least one dimension")
        return 1 if ndim == 1 else int(action.shape[0])

    def _split_policy_output(self, output: PolicyOutput, obs: MetaObs):
        if not isinstance(output, PolicyOutput):
            raise TypeError("policy adapter must return PolicyOutput")

        action_value = output.action.action
        action_size = self._action_size(action_value)
        if self.execution_horizon is not None:
            action_size = min(action_size, self.execution_horizon)
        if action_size <= 0:
            raise ValueError("action chunk cannot be empty")

        policy_info = dict(output.policy_info)
        action_policy_info = policy_info.pop(ACTION_POLICY_INFO_KEY, {})
        if not isinstance(action_policy_info, Mapping):
            raise TypeError(
                f"policy_info[{ACTION_POLICY_INFO_KEY!r}] must be a mapping"
            )
        for name, values in action_policy_info.items():
            if isinstance(values, (str, bytes, Mapping)) or not hasattr(
                values, "__len__"
            ):
                raise TypeError(
                    f"per-action policy info {name!r} must be a sequence"
                )
            if len(values) < action_size:
                raise ValueError(
                    f"per-action policy info {name!r} has {len(values)} entries "
                    f"for an executed chunk of size {action_size}"
                )

        decision_id = policy_info.setdefault(
            "decision_id",
            self._next_decision_id,
        )
        trace = policy_info.pop("policy_trace", None)
        if trace is not None and not isinstance(trace, PolicyTrace):
            raise TypeError("policy_trace must be PolicyTrace")
        if trace is None and "log_prob" in action_policy_info:
            old_logprobs = np.asarray(action_policy_info["log_prob"][:action_size])
            if old_logprobs.ndim == 1:
                trace = PolicyTrace(
                    kind=str(policy_info.pop("trace_kind", "chunk")),
                    old_logprobs=old_logprobs.copy(),
                    valid_mask=np.ones(action_size, dtype=bool),
                    axis_names=("action",),
                )
        elif trace is None and "log_prob" in policy_info:
            old_logprobs = np.asarray(policy_info["log_prob"])
            if old_logprobs.ndim == 0:
                trace = PolicyTrace(
                    kind=str(policy_info.pop("trace_kind", "decision")),
                    old_logprobs=old_logprobs.reshape(1).copy(),
                    valid_mask=np.ones(1, dtype=bool),
                    axis_names=("decision",),
                )

        decision_obs = policy_info.get("decision_obs", obs)
        if not isinstance(decision_obs, MetaObs):
            raise TypeError("decision_obs must be MetaObs")
        decision_action_ndim = (
            int(action_value.ndim)
            if hasattr(action_value, "ndim")
            else int(np.asarray(action_value).ndim)
        )
        decision_action_value = action_value
        if decision_action_ndim != 1:
            decision_action_value = action_value[:action_size]
        decision_action = replace(output.action, action=decision_action_value)
        decision = PolicyDecision(
            decision_id=decision_id,
            obs=copy.deepcopy(decision_obs),
            action=decision_action,
            trace=trace,
            value=policy_info.get("value"),
            extras={
                **policy_info,
                ACTION_POLICY_INFO_KEY: dict(action_policy_info),
            },
        )

        action_ndim = (
            int(action_value.ndim)
            if hasattr(action_value, "ndim")
            else int(np.asarray(action_value).ndim)
        )
        chunk_id = self._next_chunk_id
        chunk = []
        for index in range(action_size):
            step_action = action_value if action_ndim == 1 else action_value[index]
            action = replace(output.action, action=step_action)
            step_policy_info = dict(policy_info)
            step_policy_info.update(
                {name: values[index] for name, values in action_policy_info.items()}
            )
            step_policy_info.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "chunk_size": action_size,
                }
            )
            chunk.append(
                PolicyOutput(
                    action=action,
                    policy_info=step_policy_info,
                    decision=decision,
                    action_offset=index,
                )
            )
        self._next_decision_id += 1
        return chunk

    def select_action(
        self,
        obs: MetaObs,
        *,
        deterministic: bool = False,
        context: Optional[Mapping[str, Any]] = None,
    ):
        if self._closed:
            raise RuntimeError("policy executor is closed")
        if self.needs_inference:
            output = self.adapter.select_action(
                obs,
                deterministic=deterministic,
                context=context,
            )
            self.enqueue_policy_output(output, obs)
        return self.take_action()

    def enqueue_policy_output(self, output, obs, *, policy_info=None) -> None:
        """Queue one already-computed policy output for this execution lane."""

        if self._closed:
            raise RuntimeError("policy executor is closed")
        if not self.needs_inference:
            raise RuntimeError("cannot enqueue a new chunk before consuming the old one")
        if policy_info is not None:
            if not isinstance(policy_info, Mapping):
                raise TypeError("policy_info overrides must be a mapping")
            merged = dict(output.policy_info)
            merged.update(policy_info)
            output = replace(output, policy_info=merged)
        chunk = self._split_policy_output(output, obs)
        self.chunk_manager.put(chunk)
        self._next_chunk_id += 1
        self._inference_count += 1

    def take_action(self):
        """Pop one step action from the lane's shared chunk manager."""

        if self._closed:
            raise RuntimeError("policy executor is closed")
        output = self.chunk_manager.get()
        if output is None:
            raise RuntimeError("chunk manager returned no action after policy inference")
        return output

    def reset(self) -> None:
        if self._closed:
            return
        self.chunk_manager.reset()
        self._inference_count = 0
        self._next_chunk_id = 0
        if self.owns_adapter:
            reset = getattr(self.adapter, "reset", None)
            if callable(reset):
                reset()

    def policy_updated(self) -> None:
        if self._closed:
            return
        self.chunk_manager.reset()
        if self.owns_adapter:
            hook = getattr(self.adapter, "policy_updated", None)
            if callable(hook):
                hook()

    def close(self) -> None:
        if self._closed:
            return
        self.chunk_manager.reset()
        if self.owns_adapter:
            close = getattr(self.adapter, "close", None)
            if callable(close):
                close()
        self._closed = True
