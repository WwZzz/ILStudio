"""Map decision-level credit onto policy-specific likelihood axes."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch

from rl.base import PolicyObjective, PolicyTrace, Rollout


def _as_tensor(value, *, like=None, dtype=None):
    if torch.is_tensor(value):
        result = value
        if like is not None:
            result = result.to(device=like.device)
        if dtype is not None:
            result = result.to(dtype=dtype)
        return result
    kwargs = {}
    if like is not None:
        kwargs["device"] = like.device
        kwargs["dtype"] = like.dtype if dtype is None else dtype
    elif dtype is not None:
        kwargs["dtype"] = dtype
    return torch.as_tensor(value, **kwargs)


def _trace_mapping(rollout, new_traces):
    if isinstance(new_traces, Mapping):
        mapping = dict(new_traces)
    elif isinstance(new_traces, Sequence):
        if len(new_traces) != len(rollout.decisions):
            raise ValueError("new traces must align with rollout decisions")
        mapping = {
            decision.decision_id: trace
            for decision, trace in zip(rollout.decisions, new_traces)
        }
    else:
        raise TypeError("new traces must be a mapping or sequence")
    missing = {
        decision.decision_id
        for decision in rollout.decisions
        if decision.decision_id not in mapping
    }
    if missing:
        raise KeyError(f"new traces are missing decision ids: {sorted(missing)!r}")
    if not all(isinstance(trace, PolicyTrace) for trace in mapping.values()):
        raise TypeError("new traces must contain PolicyTrace values")
    return mapping


def _trace_tensors(old_trace, new_trace):
    if old_trace.kind != new_trace.kind:
        raise ValueError("old and new policy trace kinds must match")
    if old_trace.axis_names != new_trace.axis_names:
        raise ValueError("old and new policy trace axes must match")
    new = _as_tensor(new_trace.old_logprobs)
    old = _as_tensor(old_trace.old_logprobs, like=new).detach()
    if old.shape != new.shape:
        raise ValueError("old and new policy trace shapes must match")
    old_mask = (
        torch.ones_like(old, dtype=torch.bool)
        if old_trace.valid_mask is None
        else _as_tensor(old_trace.valid_mask, like=old, dtype=torch.bool)
    )
    new_mask = (
        torch.ones_like(new, dtype=torch.bool)
        if new_trace.valid_mask is None
        else _as_tensor(new_trace.valid_mask, like=new, dtype=torch.bool)
    )
    if old_mask.shape != old.shape or new_mask.shape != new.shape:
        raise ValueError("policy trace masks must match likelihood shapes")
    return old, new, old_mask & new_mask


def _apply_executed_action_mask(mask, axis_names, executed_offsets):
    if "action" not in axis_names:
        return mask
    axis = axis_names.index("action")
    action_index = torch.arange(mask.shape[axis], device=mask.device)
    allowed = torch.zeros(mask.shape[axis], dtype=torch.bool, device=mask.device)
    if executed_offsets:
        offsets = torch.as_tensor(
            sorted(executed_offsets), dtype=torch.long, device=mask.device
        )
        if int(offsets.max()) >= mask.shape[axis]:
            raise ValueError("executed action offset exceeds policy trace action axis")
        allowed[offsets] = True
    shape = [1] * mask.ndim
    shape[axis] = mask.shape[axis]
    return mask & allowed.reshape(shape)


class BasePolicyObjectiveBuilder(ABC):
    """Convert rollout decisions and new traces into a PPO-like objective."""

    @abstractmethod
    def build(
        self,
        rollout,
        new_traces,
        advantages,
        *,
        values=None,
        returns=None,
    ) -> PolicyObjective:
        pass

    @staticmethod
    def _inputs(rollout, new_traces, advantages):
        if not isinstance(rollout, Rollout):
            raise TypeError("objective builder rollout must be Rollout")
        groups = rollout.decision_transitions()
        if len(groups) != len(rollout.decisions):
            raise ValueError("every rollout decision must have an executed transition")
        traces = _trace_mapping(rollout, new_traces)
        advantages = _as_tensor(advantages)
        if advantages.ndim != 1 or len(advantages) != len(groups):
            raise ValueError("advantages must contain one value per decision")
        return groups, traces, advantages


class ChunkPolicyObjectiveBuilder(BasePolicyObjectiveBuilder):
    """Use one joint likelihood and one advantage per executed action chunk."""

    def build(self, rollout, new_traces, advantages, *, values=None, returns=None):
        groups, traces, advantages = self._inputs(
            rollout, new_traces, advantages
        )
        old_joint = []
        new_joint = []
        for group in groups:
            old_trace = group.decision.trace
            if old_trace is None:
                raise ValueError("chunk objective requires a stored policy trace")
            new_trace = traces[group.decision.decision_id]
            old, new, mask = _trace_tensors(old_trace, new_trace)
            mask = _apply_executed_action_mask(
                mask,
                old_trace.axis_names,
                {step.action_offset for step in group.steps},
            )
            if not bool(mask.any()):
                raise ValueError("chunk objective has no valid executed likelihood")
            old_joint.append(old.masked_select(mask).sum())
            new_joint.append(new.masked_select(mask).sum())
        new_joint = torch.stack(new_joint)
        old_joint = torch.stack(old_joint).to(new_joint)
        advantage = advantages.to(new_joint)
        return PolicyObjective(
            old_logprobs=old_joint,
            new_logprobs=new_joint,
            advantages=advantage,
            mask=torch.ones_like(new_joint, dtype=torch.bool),
            values=values,
            returns=returns,
            axis_names=("decision",),
        )


class ActionChunkPolicyObjectiveBuilder(BasePolicyObjectiveBuilder):
    """Clip each executed action likelihood while sharing decision credit.

    This keeps chunk-level collection and return estimation, but avoids making
    the PPO ratio's scale proportional to the number of executed actions. A
    one-step decision is the exact ``chunk_size=1`` special case.
    """

    def build(self, rollout, new_traces, advantages, *, values=None, returns=None):
        groups, traces, advantages = self._inputs(
            rollout, new_traces, advantages
        )
        old_parts = []
        new_parts = []
        advantage_parts = []
        for index, group in enumerate(groups):
            old_trace = group.decision.trace
            if old_trace is None or "action" not in old_trace.axis_names:
                raise ValueError("action-chunk objective requires an action trace axis")
            new_trace = traces[group.decision.decision_id]
            old, new, mask = _trace_tensors(old_trace, new_trace)
            mask = _apply_executed_action_mask(
                mask,
                old_trace.axis_names,
                {step.action_offset for step in group.steps},
            )
            if not bool(mask.any()):
                raise ValueError("action-chunk objective has no executed likelihood")
            old_parts.append(old.masked_select(mask))
            new_parts.append(new.masked_select(mask))
            advantage_parts.append(advantages[index].expand(int(mask.sum())))
        new = torch.cat(new_parts)
        return PolicyObjective(
            old_logprobs=torch.cat(old_parts).to(new),
            new_logprobs=new,
            advantages=torch.cat(advantage_parts).to(new),
            mask=torch.ones_like(new, dtype=torch.bool),
            values=values,
            returns=returns,
            axis_names=("objective",),
        )


class DenoisingPolicyObjectiveBuilder(BasePolicyObjectiveBuilder):
    """Repeat each decision advantage over its valid denoising steps."""

    def build(self, rollout, new_traces, advantages, *, values=None, returns=None):
        groups, traces, advantages = self._inputs(
            rollout, new_traces, advantages
        )
        old_parts = []
        new_parts = []
        advantage_parts = []
        for index, group in enumerate(groups):
            old_trace = group.decision.trace
            if old_trace is None or "denoise" not in old_trace.axis_names:
                raise ValueError("denoising objective requires a denoise trace axis")
            new_trace = traces[group.decision.decision_id]
            old, new, mask = _trace_tensors(old_trace, new_trace)
            mask = _apply_executed_action_mask(
                mask,
                old_trace.axis_names,
                {step.action_offset for step in group.steps},
            )
            denoise_axis = old_trace.axis_names.index("denoise")
            old = old.movedim(denoise_axis, 0).reshape(old.shape[denoise_axis], -1)
            new = new.movedim(denoise_axis, 0).reshape(new.shape[denoise_axis], -1)
            mask = mask.movedim(denoise_axis, 0).reshape(mask.shape[denoise_axis], -1)
            valid = mask.any(dim=1)
            if not bool(valid.any()):
                raise ValueError("denoising objective has no valid likelihood")
            old_parts.append((old * mask).sum(dim=1)[valid])
            new_parts.append((new * mask).sum(dim=1)[valid])
            advantage_parts.append(advantages[index].expand(int(valid.sum())))
        new = torch.cat(new_parts)
        return PolicyObjective(
            old_logprobs=torch.cat(old_parts).to(new),
            new_logprobs=new,
            advantages=torch.cat(advantage_parts).to(new),
            mask=torch.ones_like(new, dtype=torch.bool),
            values=values,
            returns=returns,
            axis_names=("objective",),
        )


class TokenPolicyObjectiveBuilder(BasePolicyObjectiveBuilder):
    """Map decision credit onto valid action-token likelihoods."""

    def build(self, rollout, new_traces, advantages, *, values=None, returns=None):
        groups, traces, advantages = self._inputs(
            rollout, new_traces, advantages
        )
        old_parts = []
        new_parts = []
        advantage_parts = []
        for index, group in enumerate(groups):
            old_trace = group.decision.trace
            if old_trace is None or "token" not in old_trace.axis_names:
                raise ValueError("token objective requires a token trace axis")
            new_trace = traces[group.decision.decision_id]
            old, new, mask = _trace_tensors(old_trace, new_trace)
            executed = {step.action_offset for step in group.steps}
            if len(executed) < group.decision.chunk_size:
                action_offsets = old_trace.extras.get("action_offsets")
                if action_offsets is None:
                    raise ValueError(
                        "partial token chunks require trace action_offsets"
                    )
                action_offsets = _as_tensor(
                    action_offsets, like=old, dtype=torch.long
                )
                if action_offsets.shape != old.shape:
                    raise ValueError("token action_offsets must match trace shape")
                executed_mask = torch.zeros_like(mask)
                for offset in executed:
                    executed_mask |= action_offsets == offset
                mask &= executed_mask
            old = old.reshape(-1)
            new = new.reshape(-1)
            mask = mask.reshape(-1)
            if not bool(mask.any()):
                raise ValueError("token objective has no valid action tokens")
            old_parts.append(old[mask])
            new_parts.append(new[mask])
            advantage_parts.append(advantages[index].expand(int(mask.sum())))
        new = torch.cat(new_parts)
        return PolicyObjective(
            old_logprobs=torch.cat(old_parts).to(new),
            new_logprobs=new,
            advantages=torch.cat(advantage_parts).to(new),
            mask=torch.ones_like(new, dtype=torch.bool),
            values=values,
            returns=returns,
            axis_names=("objective",),
        )
