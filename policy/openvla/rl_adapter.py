"""Token-policy reinforcement-learning adapter for OpenVLA.

The environment boundary remains ILStudio ``MetaObs``/``MetaAction``.  This
module owns the OpenVLA-specific conversion between those values and
autoregressive action tokens, while algorithms own advantages and losses.
"""

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from benchmark.base import MetaAction, MetaObs
from rl.base import PolicyOutput, PolicyTrace
from rl.policy_adapter import BasePolicyAdapter
from rl.policy_adapter.trainer import BaseTrainerAdapter, TrainerStepResult

from .modeling import OpenPolicy


_SUPPORTED_ALGORITHMS = frozenset({"ppo", "grpo"})


class _ActionTokenLogitsProcessor(LogitsProcessor):
    """Mask language tokens so every sampled token decodes to a valid action."""

    def __init__(self, first_token_id: int, vocab_size: int):
        self.first_token_id = int(first_token_id)
        self.vocab_size = int(vocab_size)
        if not 0 <= self.first_token_id < self.vocab_size:
            raise ValueError("invalid OpenVLA action-token range")

    def __call__(self, input_ids, scores):
        del input_ids
        masked = torch.full_like(scores, -torch.inf)
        masked[:, self.first_token_id : self.vocab_size] = scores[
            :, self.first_token_id : self.vocab_size
        ]
        return masked


def _unwrap_open_policy(policy):
    candidate = policy
    seen = set()
    for _ in range(8):
        if id(candidate) in seen:
            break
        seen.add(id(candidate))
        # PEFT wrappers proxy unknown attributes to the wrapped model, so an
        # attribute-only check mistakes PeftModel for OpenPolicy and makes
        # generation target the wrapper one level too high.
        if isinstance(candidate, OpenPolicy):
            return candidate
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            base = get_base_model()
            if base is not candidate:
                candidate = base
                continue
        base_model = getattr(candidate, "base_model", None)
        if base_model is not None and base_model is not candidate:
            candidate = base_model
            continue
        model = getattr(candidate, "model", None)
        if model is not None and model is not candidate:
            candidate = model
            continue
        break
    raise TypeError("OpenVLA RL adapter could not locate the OpenPolicy base model")


def _model_output_logits(output):
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, Mapping):
        logits = output.get("logits")
    if logits is None:
        raise TypeError("OpenVLA forward output must expose logits for RL")
    return logits


class OpenVLAPolicyAdapter(BasePolicyAdapter):
    """Sample OpenVLA action tokens and recompute their likelihoods."""

    def __init__(
        self,
        model_components,
        *,
        required_capabilities: Iterable[str] = (),
        temperature: float = 1.0,
        action_dim: int | None = None,
        generation_suffix_token_id: int | None = 29871,
        restrict_action_tokens: bool = True,
    ):
        if not isinstance(model_components, Mapping):
            raise TypeError("model_components must be a mapping")
        policy = model_components.get("model")
        meta_policy = model_components.get("meta_policy")
        if policy is None or meta_policy is None:
            raise KeyError("OpenVLA RL requires model and meta_policy components")
        requested = set(required_capabilities)
        algorithms = requested.intersection(_SUPPORTED_ALGORITHMS)
        unsupported = requested - {"action"} - _SUPPORTED_ALGORITHMS
        if unsupported:
            raise ValueError(
                "OpenVLA RL adapter does not support capabilities: "
                + ", ".join(sorted(unsupported))
            )
        if len(algorithms) > 1:
            raise ValueError("select exactly one OpenVLA RL algorithm capability")
        capabilities = {"action", *algorithms}
        super().__init__(policy, capabilities=capabilities)

        if not isinstance(temperature, (int, float)) or float(temperature) <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.meta_policy = meta_policy
        self.open_policy = _unwrap_open_policy(policy)
        self.backbone = self.open_policy.model
        configured_dim = getattr(self.open_policy.config, "action_dim", None)
        self.action_dim = int(action_dim if action_dim is not None else configured_dim)
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if generation_suffix_token_id is not None and not isinstance(
            generation_suffix_token_id, int
        ):
            raise TypeError("generation_suffix_token_id must be int or None")
        self.generation_suffix_token_id = generation_suffix_token_id
        self.restrict_action_tokens = bool(restrict_action_tokens)
        self.action_tokenizer = self.open_policy.action_tokenizer
        self.tokenizer = self.open_policy.tokenizer
        self.first_action_token_id = int(
            self.action_tokenizer.action_token_begin_idx + 1
        )
        self.vocab_size = int(self.tokenizer.vocab_size)
        self._logits_processor = LogitsProcessorList(
            [
                _ActionTokenLogitsProcessor(
                    self.first_action_token_id, self.vocab_size
                )
            ]
            if self.restrict_action_tokens
            else []
        )
        if not any(parameter.requires_grad for parameter in self.policy.parameters()):
            for name, parameter in self.policy.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad_(True)
        if not any(parameter.requires_grad for parameter in self.policy.parameters()):
            raise ValueError("OpenVLA checkpoint exposes no trainable parameters")

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
            raise RuntimeError("one MetaObs must produce exactly one OpenVLA sample")
        return samples[0]

    def _encode(self, observations):
        samples = [self._obs_to_sample(obs) for obs in observations]
        batch = self.meta_policy.meta2obs(samples)
        result = {}
        for key, value in batch.items():
            if not torch.is_tensor(value) or key == "labels":
                continue
            if key == "pixel_values":
                value = value.to(self.device, dtype=torch.bfloat16)
            else:
                value = value.to(self.device)
            result[key] = value
        if "input_ids" not in result:
            raise KeyError("OpenVLA processor did not return input_ids")
        if "attention_mask" not in result:
            pad_id = self.tokenizer.pad_token_id
            result["attention_mask"] = result["input_ids"].ne(pad_id).long()
        return result

    def _prompt_rows(self, encoded):
        rows = []
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        for index, length in enumerate(lengths):
            ids = encoded["input_ids"][index, : int(length)]
            suffix = self.generation_suffix_token_id
            if suffix is not None and (ids.numel() == 0 or int(ids[-1]) != suffix):
                ids = torch.cat([ids, ids.new_tensor([suffix])])
            rows.append(ids)
        return rows

    def _sample_next_token(self, input_ids, logits, *, deterministic):
        scores = self._logits_processor(input_ids, logits.float())
        if not deterministic:
            scores = scores / self.temperature
        log_distribution = torch.log_softmax(scores, dim=-1)
        if deterministic:
            token = scores.argmax(dim=-1)
        else:
            token = torch.multinomial(log_distribution.exp(), num_samples=1).squeeze(1)
        logprob = log_distribution.gather(1, token[:, None]).squeeze(1)
        return token, logprob

    def _generate_batched_tokens(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        *,
        deterministic,
    ):
        """Generate with one visual pass and a batched language-model KV cache.

        Upstream Prismatic rejects batch sizes above one in its GenerationMixin
        and cached ``forward`` branch, even though its underlying causal LM
        supports a batched cache.  Entering those submodules directly preserves
        all injected LoRA layers while avoiding sequential environment rollout.
        """

        if pixel_values is None:
            raise ValueError("OpenVLA action generation requires pixel_values")
        patch_features = self.backbone.vision_backbone(pixel_values)
        projected = self.backbone.projector(patch_features)
        token_embeddings = self.backbone.get_input_embeddings()(input_ids)
        multimodal_embeddings = torch.cat(
            [token_embeddings[:, :1], projected, token_embeddings[:, 1:]],
            dim=1,
        )
        patch_mask = torch.ones(
            (projected.shape[0], projected.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        multimodal_mask = torch.cat(
            [attention_mask[:, :1], patch_mask, attention_mask[:, 1:]],
            dim=1,
        )
        output = self.backbone.language_model(
            inputs_embeds=multimodal_embeddings,
            attention_mask=multimodal_mask,
            use_cache=True,
            return_dict=True,
        )
        generated = []
        logprobs = []
        running_ids = input_ids
        for offset in range(self.action_dim):
            token, logprob = self._sample_next_token(
                running_ids,
                output.logits[:, -1],
                deterministic=deterministic,
            )
            generated.append(token)
            logprobs.append(logprob)
            running_ids = torch.cat([running_ids, token[:, None]], dim=1)
            if offset + 1 == self.action_dim:
                break
            output = self.backbone.language_model(
                input_ids=token[:, None],
                past_key_values=output.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        return torch.stack(generated, dim=1), torch.stack(logprobs, dim=1)

    def _sample_tokens(self, observations, *, deterministic):
        encoded = self._encode(observations)
        rows = self._prompt_rows(encoded)
        token_rows = [None] * len(rows)
        logprob_rows = [None] * len(rows)
        by_length = {}
        for index, row in enumerate(rows):
            by_length.setdefault(int(row.numel()), []).append(index)

        for indices in by_length.values():
            input_ids = torch.stack([rows[index] for index in indices])
            attention_mask = torch.ones_like(input_ids)
            pixel_values = encoded.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values[indices]
            tokens, logprobs = self._generate_batched_tokens(
                input_ids,
                attention_mask,
                pixel_values,
                deterministic=deterministic,
            )
            for local_index, original_index in enumerate(indices):
                token_rows[original_index] = tokens[local_index].detach()
                logprob_rows[original_index] = logprobs[local_index].detach()
        return token_rows, logprob_rows

    def _decode_meta_action(self, token_ids):
        normalized = self.action_tokenizer.decode_token_ids_to_actions(
            token_ids.detach().cpu().numpy()
        ).astype(np.float32)
        action = normalized.reshape(1, 1, -1)
        meta = self.meta_policy.act2meta(
            action,
            ctrl_space=self.meta_policy.ctrl_space,
            ctrl_type=self.meta_policy.ctrl_type,
        )
        meta = self.meta_policy.action_normalizer.denormalize_metaact(meta)
        if hasattr(self.policy, "post_process_action") and callable(
            self.policy.post_process_action
        ):
            meta.action = self.policy.post_process_action(
                meta.action,
                None,
                self.meta_policy.action_normalizer,
                self.meta_policy.state_normalizer,
            )
        value = np.asarray(meta.action, dtype=np.float32)
        if value.ndim == 3:
            value = value[0]
        if value.ndim == 1:
            value = value[None, :]
        return MetaAction(
            ctrl_space=meta.ctrl_space,
            ctrl_type=meta.ctrl_type,
            action=value,
            gripper_continuous=meta.gripper_continuous,
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
        outputs = []
        for tokens, logprobs in zip(token_rows, logprob_rows):
            trace = PolicyTrace(
                kind="openvla_action_tokens",
                old_logprobs=logprobs.float().cpu().numpy(),
                valid_mask=np.ones(self.action_dim, dtype=bool),
                axis_names=("token",),
                extras={
                    "token_ids": tokens.long().cpu().numpy(),
                    "action_offsets": np.zeros(self.action_dim, dtype=np.int64),
                    "temperature": self.temperature,
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
        encoded = self._encode(tuple(decision.obs for decision in decisions))
        prompt_rows = self._prompt_rows(encoded)
        token_rows = []
        for decision in decisions:
            if decision.trace is None:
                raise ValueError("OpenVLA update requires stored token traces")
            tokens = decision.trace.extras.get("token_ids")
            if tokens is None:
                raise KeyError("OpenVLA token trace is missing token_ids")
            tokens = torch.as_tensor(tokens, device=self.device, dtype=torch.long)
            if tokens.shape != (self.action_dim,):
                raise ValueError("OpenVLA token trace has the wrong action length")
            token_rows.append(tokens)

        lengths = [row.numel() for row in prompt_rows]
        full_lengths = [length + self.action_dim for length in lengths]
        pad_id = int(self.tokenizer.pad_token_id)
        full_ids = torch.full(
            (len(decisions), max(full_lengths)),
            pad_id,
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(full_ids)
        for index, (prompt, tokens) in enumerate(zip(prompt_rows, token_rows)):
            joined = torch.cat([prompt, tokens])
            full_ids[index, : joined.numel()] = joined
            attention_mask[index, : joined.numel()] = 1
        kwargs = {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
        }
        if "pixel_values" in encoded:
            kwargs["pixel_values"] = encoded["pixel_values"]
        logits = _model_output_logits(self.policy(**kwargs)).float()
        traces = {}
        entropies = []
        for index, (decision, prompt_len, tokens) in enumerate(
            zip(decisions, lengths, token_rows)
        ):
            positions = torch.arange(
                prompt_len - 1,
                prompt_len + self.action_dim - 1,
                device=self.device,
            )
            token_logits = logits[index, positions]
            if self.restrict_action_tokens:
                token_logits = token_logits[
                    :, self.first_action_token_id : self.vocab_size
                ]
                token_index = tokens - self.first_action_token_id
            else:
                token_index = tokens
            token_logits = token_logits / self.temperature
            log_distribution = torch.log_softmax(token_logits, dim=-1)
            new_logprobs = log_distribution.gather(
                1, token_index[:, None]
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
            raise ValueError(f"unsupported OpenVLA RL operation {operation!r}")
        rollout = getattr(batch, "rollout", None)
        if rollout is None:
            raise ValueError("OpenVLA token updates require a rollout-aware batch")
        traces, entropy = self._recompute_traces(rollout)
        return {"traces": traces, "entropy": entropy}

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = self.policy.save_pretrained(output_dir)
        self.open_policy.config.save_pretrained(output_dir)
        self.open_policy.processor.save_pretrained(output_dir)
        return result


class OpenVLATrainerAdapter(BaseTrainerAdapter):
    """One clipped optimizer step for the large autoregressive policy."""

    STATE_VERSION = 1

    def __init__(self, optimizer, *, max_grad_norm: float = 1.0):
        if optimizer is None:
            raise TypeError("OpenVLA trainer adapter requires an optimizer")
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
                raise TypeError("OpenVLA trainer adapter expects one combined loss")
            weight = output.payload.get("loss_weight", 1.0)
            if not isinstance(weight, (int, float)) or float(weight) <= 0:
                raise ValueError("OpenVLA loss_weight must be positive")
            weight = float(weight)
            (output.loss * weight).backward()
            total_weight += weight
            updated = True
        if not updated:
            return TrainerStepResult(updated=False)
        if abs(total_weight - 1.0) > 1e-5:
            raise ValueError("OpenVLA accumulated loss weights must sum to one")
        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.max_grad_norm
        )
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
            raise ValueError("unsupported OpenVLA trainer adapter state")
        self.optimizer.load_state_dict(state["optimizer"])
        self.step_count = int(state["step_count"])


RLPolicyAdapter = OpenVLAPolicyAdapter


def build_rl_adapter(*, model_components, required_capabilities=(), **kwargs):
    return OpenVLAPolicyAdapter(
        model_components,
        required_capabilities=required_capabilities,
        **kwargs,
    )


def build_trainer_adapter(
    *, policy_components, optimizer, policy_adapter=None, max_grad_norm=1.0, **kwargs
):
    del policy_components, policy_adapter, kwargs
    return OpenVLATrainerAdapter(
        optimizer, max_grad_norm=max_grad_norm
    )
