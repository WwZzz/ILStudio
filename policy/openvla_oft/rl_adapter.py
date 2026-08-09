"""Chunk-token reinforcement-learning adapter for OpenVLA-OFT.

SimpleVLA-RL treats an OpenVLA-OFT action chunk as ``chunk_size * action_dim``
parallel discrete action tokens.  This adapter exposes the same likelihood
semantics while keeping ILStudio's ``MetaObs``/``MetaAction`` boundary and
existing action-chunk executor unchanged.
"""

import copy
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch

from benchmark.base import MetaAction, MetaObs
from rl.base import (
    PolicyOutput,
    PolicyTrace,
    RL_LIKELIHOOD_GROUP_KEY,
    RL_LIKELIHOOD_GROUP_SIZE_KEY,
)
from rl.policy_adapter import BasePolicyAdapter
from rl.policy_adapter.trainer import BaseTrainerAdapter, TrainerStepResult

from .modeling import OpenVLAOFTPolicy


_SUPPORTED_ALGORITHMS = frozenset({"ppo", "grpo"})


class OpenVLAOFTPolicyAdapter(BasePolicyAdapter):
    """Sample and train the parallel action-token policy used by SimpleVLA-RL."""

    def __init__(
        self,
        model_components,
        *,
        required_capabilities: Iterable[str] = (),
        temperature: float = 1.6,
        action_vocab_size: int = 256,
    ):
        if not isinstance(model_components, Mapping):
            raise TypeError("model_components must be a mapping")
        policy = model_components.get("model")
        meta_policy = model_components.get("meta_policy")
        if not isinstance(policy, OpenVLAOFTPolicy) or meta_policy is None:
            raise TypeError(
                "OpenVLA-OFT RL requires OpenVLAOFTPolicy and MetaPolicy components"
            )
        if getattr(meta_policy, "policy", None) is not policy:
            raise ValueError("meta_policy and model_components must share the model")

        requested = set(required_capabilities)
        algorithms = requested.intersection(_SUPPORTED_ALGORITHMS)
        unsupported = requested - {"action"} - _SUPPORTED_ALGORITHMS
        if unsupported:
            raise ValueError(
                "OpenVLA-OFT RL adapter does not support capabilities: "
                + ", ".join(sorted(unsupported))
            )
        if len(algorithms) > 1:
            raise ValueError("select exactly one OpenVLA-OFT RL algorithm capability")
        super().__init__(policy, capabilities={"action", *algorithms})

        if not isinstance(temperature, (int, float)) or float(temperature) <= 0:
            raise ValueError("temperature must be positive")
        if (
            isinstance(action_vocab_size, bool)
            or not isinstance(action_vocab_size, int)
            or action_vocab_size <= 0
        ):
            raise ValueError("action_vocab_size must be a positive integer")
        if bool(getattr(policy.config, "use_film", False)):
            raise ValueError("SimpleVLA-RL-compatible token updates do not support FiLM")

        self.temperature = float(temperature)
        self.action_vocab_size = action_vocab_size
        self.meta_policy = meta_policy
        self.vla = policy.vla
        self.tokenizer = policy.tokenizer
        self.action_tokenizer = policy.action_tokenizer
        self.action_dim = int(policy.config.action_dim)
        self.chunk_size = int(policy.config.num_actions_chunk)
        self.num_action_tokens = self.action_dim * self.chunk_size
        self.vocab_size = int(getattr(self.vla, "vocab_size", self.tokenizer.vocab_size))
        self.first_action_token_id = self.vocab_size - self.action_vocab_size
        if self.first_action_token_id < 0:
            raise ValueError("action token range exceeds the OpenVLA vocabulary")

        # SimpleVLA-RL applies LoRA to the complete OFT actor, which freezes the
        # base model and OFT projectors.  ILStudio wraps only ``model.vla``, so
        # explicitly enforce the same trainable-parameter boundary for LoRA RL.
        if getattr(policy.config, "training_mode", None) == "lora":
            for name, parameter in policy.named_parameters():
                parameter.requires_grad_("lora_" in name)
        if not any(parameter.requires_grad for parameter in policy.parameters()):
            raise ValueError("OpenVLA-OFT checkpoint exposes no trainable parameters")

    @property
    def device(self):
        return next(self.policy.parameters()).device

    def _obs_to_sample(self, obs: MetaObs):
        self._validate_obs(obs)
        batched = copy.deepcopy(obs)
        batched.to_batch()
        normalized = self.meta_policy.state_normalizer.normalize_metaobs(
            batched, self.meta_policy.ctrl_space
        )
        samples = self.meta_policy.normed_mobs_to_samples(normalized)
        if len(samples) != 1:
            raise RuntimeError("one MetaObs must produce exactly one OFT sample")
        return samples[0]

    def _encode(self, observations):
        samples = [self._obs_to_sample(obs) for obs in observations]
        batch = self.meta_policy.meta2obs(samples)
        result = {}
        for key, value in batch.items():
            if not torch.is_tensor(value) or key in {"labels", "actions"}:
                continue
            if key == "pixel_values":
                value = value.to(self.device, dtype=torch.bfloat16)
            else:
                value = value.to(self.device)
            result[key] = value
        if "input_ids" not in result or "pixel_values" not in result:
            raise KeyError("OpenVLA-OFT processor must return input_ids and pixel_values")
        if "attention_mask" not in result:
            result["attention_mask"] = result["input_ids"].ne(
                self.tokenizer.pad_token_id
            ).long()
        return result

    @staticmethod
    def _rows_by_prompt_length(encoded):
        groups = {}
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        for index, length in enumerate(lengths):
            groups.setdefault(int(length), []).append(index)
        return groups

    def _parallel_action_logits(self, input_ids, pixel_values, proprio=None):
        """Return ``[B, chunk*dim, 256]`` logits matching SimpleVLA-RL."""
        logits = self.policy.parallel_action_token_logits(
            input_ids,
            pixel_values,
            proprio,
            action_vocab_size=self.action_vocab_size,
        )
        return logits.float() / self.temperature

    def _logits_for_observations(self, observations):
        encoded = self._encode(observations)
        rows = [None] * len(observations)
        for prompt_length, indices in self._rows_by_prompt_length(encoded).items():
            input_ids = torch.stack(
                [encoded["input_ids"][index, :prompt_length] for index in indices]
            )
            pixel_values = encoded["pixel_values"][indices]
            proprio = encoded.get("proprio")
            if proprio is not None:
                proprio = proprio[indices]
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                logits = self._parallel_action_logits(
                    input_ids, pixel_values, proprio=proprio
                )
            for local_index, original_index in enumerate(indices):
                rows[original_index] = logits[local_index]
        return rows

    def _sample_tokens(self, observations, *, deterministic):
        logits_rows = self._logits_for_observations(observations)
        token_rows = []
        logprob_rows = []
        for logits in logits_rows:
            log_distribution = torch.log_softmax(logits, dim=-1)
            if deterministic:
                local_tokens = logits.argmax(dim=-1)
            else:
                local_tokens = torch.multinomial(
                    log_distribution.exp(), num_samples=1
                ).squeeze(1)
            token_rows.append(local_tokens + self.first_action_token_id)
            logprob_rows.append(
                log_distribution.gather(1, local_tokens[:, None]).squeeze(1)
            )
        return token_rows, logprob_rows

    def _decode_meta_action(self, token_ids):
        normalized = self.action_tokenizer.decode_token_ids_to_actions(
            token_ids.detach().cpu().numpy()
        ).astype(np.float32)
        normalized = normalized.reshape(1, self.chunk_size, self.action_dim)
        action = self.meta_policy.act2meta(
            normalized,
            ctrl_space=self.meta_policy.ctrl_space,
            ctrl_type=self.meta_policy.ctrl_type,
        )
        action = self.meta_policy.action_normalizer.denormalize_metaact(action)
        post_process = getattr(self.policy, "post_process_action", None)
        if callable(post_process):
            action.action = post_process(
                action.action,
                None,
                self.meta_policy.action_normalizer,
                self.meta_policy.state_normalizer,
            )
        value = np.asarray(action.action, dtype=np.float32)
        if value.ndim == 3:
            value = value[0]
        if value.ndim == 1:
            value = value[None, :]
        return MetaAction(
            ctrl_space=action.ctrl_space,
            ctrl_type=action.ctrl_type,
            action=value,
            gripper_continuous=action.gripper_continuous,
        )

    def select_actions(self, observations, *, deterministic=False, context=None):
        del context
        observations = tuple(observations)
        if not observations:
            raise ValueError("observations cannot be empty")
        with torch.no_grad():
            token_rows, logprob_rows = self._sample_tokens(
                observations, deterministic=deterministic
            )
        action_offsets = np.repeat(
            np.arange(self.chunk_size, dtype=np.int64), self.action_dim
        )
        outputs = []
        for tokens, logprobs in zip(token_rows, logprob_rows):
            trace = PolicyTrace(
                kind="openvla_oft_parallel_action_tokens",
                old_logprobs=logprobs.float().cpu().numpy(),
                valid_mask=np.ones(self.num_action_tokens, dtype=bool),
                axis_names=("token",),
                extras={
                    "token_ids": tokens.long().cpu().numpy(),
                    "action_offsets": action_offsets.copy(),
                    "temperature": self.temperature,
                    "chunk_size": self.chunk_size,
                    "action_dim": self.action_dim,
                },
            )
            outputs.append(
                self._finalize_output(
                    PolicyOutput(
                        action=self._decode_meta_action(tokens),
                        policy_info={"policy_trace": trace},
                    )
                )
            )
        return tuple(outputs)

    def select_action(self, obs, *, deterministic=False, context=None):
        return self.select_actions(
            (obs,), deterministic=deterministic, context=context
        )[0]

    def _recompute_traces(self, rollout):
        decisions = tuple(rollout.decisions)
        likelihood_groups = {}
        expected_sizes = {}
        for decision in decisions:
            group_id = decision.extras.get(RL_LIKELIHOOD_GROUP_KEY)
            if group_id is None:
                key = ("decision", decision.decision_id)
                expected_size = 1
            else:
                key = ("likelihood", group_id)
                expected_size = decision.extras.get(RL_LIKELIHOOD_GROUP_SIZE_KEY)
                if (
                    isinstance(expected_size, bool)
                    or not isinstance(expected_size, int)
                    or expected_size <= 0
                ):
                    raise ValueError(
                        "OpenVLA-OFT likelihood group size must be positive"
                    )
            previous_size = expected_sizes.setdefault(key, expected_size)
            if previous_size != expected_size:
                raise ValueError(
                    "OpenVLA-OFT likelihood group members disagree on size"
                )
            likelihood_groups.setdefault(key, []).append(decision)

        logits_by_decision = {}
        for key, grouped_decisions in likelihood_groups.items():
            if len(grouped_decisions) != expected_sizes[key]:
                raise ValueError("OpenVLA-OFT likelihood recompute group is incomplete")
            logits_rows = self._logits_for_observations(
                tuple(decision.obs for decision in grouped_decisions)
            )
            logits_by_decision.update(
                (decision.decision_id, logits)
                for decision, logits in zip(grouped_decisions, logits_rows)
            )
        traces = {}
        entropies = []
        for decision in decisions:
            logits = logits_by_decision[decision.decision_id]
            if decision.trace is None:
                raise ValueError("OpenVLA-OFT update requires stored token traces")
            token_ids = decision.trace.extras.get("token_ids")
            if token_ids is None:
                raise KeyError("OpenVLA-OFT token trace is missing token_ids")
            token_ids = torch.as_tensor(
                token_ids, device=self.device, dtype=torch.long
            )
            if token_ids.shape != (self.num_action_tokens,):
                raise ValueError("OpenVLA-OFT token trace has the wrong length")
            local_tokens = token_ids - self.first_action_token_id
            if bool(((local_tokens < 0) | (local_tokens >= self.action_vocab_size)).any()):
                raise ValueError("OpenVLA-OFT trace contains a non-action token")
            log_distribution = torch.log_softmax(logits, dim=-1)
            new_logprobs = log_distribution.gather(
                1, local_tokens[:, None]
            ).squeeze(1)
            distribution = log_distribution.exp()
            entropies.append(-(distribution * log_distribution).sum(dim=-1))
            traces[decision.decision_id] = PolicyTrace(
                kind=decision.trace.kind,
                old_logprobs=new_logprobs,
                valid_mask=torch.ones_like(new_logprobs, dtype=torch.bool),
                axis_names=("token",),
                extras=dict(decision.trace.extras),
            )
        return traces, torch.cat(entropies)

    def algorithm_forward(self, operation, batch, *, context=None):
        del context
        if operation not in {"ppo_trace", "grpo_trace"}:
            raise ValueError(f"unsupported OpenVLA-OFT RL operation {operation!r}")
        rollout = getattr(batch, "rollout", None)
        if rollout is None:
            raise ValueError("OpenVLA-OFT token updates require a rollout-aware batch")
        traces, entropy = self._recompute_traces(rollout)
        return {"traces": traces, "entropy": entropy}

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = self.policy.vla.save_pretrained(output_dir)
        self.policy.config.save_pretrained(output_dir)
        self.policy.processor.save_pretrained(output_dir)

        extra_state_dict = {}
        for name in ("action_head", "proprio_projector", "noisy_action_projector"):
            module = getattr(self.policy, name, None)
            if module is not None:
                extra_state_dict[name] = module.state_dict()
        if extra_state_dict:
            torch.save(extra_state_dict, os.path.join(output_dir, "extra_weights.bin"))
        self._copy_checkpoint_assets(output_dir)
        return result


class OpenVLAOFTTrainerAdapter(BaseTrainerAdapter):
    """Accumulate streamed token-objective micro-batches into one optimizer step."""

    STATE_VERSION = 1

    def __init__(self, optimizer, *, max_grad_norm: float = 1.0):
        if optimizer is None:
            raise TypeError("OpenVLA-OFT trainer adapter requires an optimizer")
        if not isinstance(max_grad_norm, (int, float)) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        self.optimizer = optimizer
        self.max_grad_norm = float(max_grad_norm)
        self.step_count = 0

    def step(self, output, *, policy_adapter, context=None):
        return self.step_many(
            (output,), policy_adapter=policy_adapter, context=context
        )

    def step_many(self, outputs, *, policy_adapter, context=None):
        del policy_adapter, context
        self.optimizer.zero_grad(set_to_none=True)
        updated = False
        total_weight = 0.0
        for output in outputs:
            if output.loss is None:
                continue
            if isinstance(output.loss, Mapping):
                raise TypeError("OpenVLA-OFT trainer adapter expects one combined loss")
            weight = output.payload.get("loss_weight", 1.0)
            if not isinstance(weight, (int, float)) or float(weight) <= 0:
                raise ValueError("OpenVLA-OFT loss_weight must be positive")
            weight = float(weight)
            (output.loss * weight).backward()
            total_weight += weight
            updated = True
        if not updated:
            return TrainerStepResult(updated=False)
        if abs(total_weight - 1.0) > 1e-5:
            raise ValueError("OpenVLA-OFT accumulated loss weights must sum to one")
        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        self.optimizer.step()
        self.step_count += 1
        return TrainerStepResult(
            metrics={"trainer/grad_norm": float(grad_norm.detach().cpu())}
        )

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "step_count": self.step_count,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported OpenVLA-OFT trainer adapter state")
        self.optimizer.load_state_dict(state["optimizer"])
        self.step_count = int(state["step_count"])


RLPolicyAdapter = OpenVLAOFTPolicyAdapter


def build_rl_adapter(*, model_components, required_capabilities=(), **kwargs):
    return OpenVLAOFTPolicyAdapter(
        model_components,
        required_capabilities=required_capabilities,
        **kwargs,
    )


def build_trainer_adapter(
    *, policy_components, optimizer, policy_adapter=None, max_grad_norm=1.0, **kwargs
):
    del policy_components, policy_adapter, kwargs
    return OpenVLAOFTTrainerAdapter(optimizer, max_grad_norm=max_grad_norm)
