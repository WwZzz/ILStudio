"""Parameter-update adapters owned by the RL policy-adapter boundary."""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from rl.algorithm import AlgorithmOutput

from .base import BasePolicyAdapter
from .registry import _load_policy_local_adapter


@dataclass(frozen=True)
class TrainerStepResult:
    metrics: Dict[str, Any] = field(default_factory=dict)
    updated: bool = True

    def __post_init__(self):
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "updated", bool(self.updated))


class BaseTrainerAdapter(ABC):
    """Adapt one optimizer step without owning the outer RL lifecycle."""

    @abstractmethod
    def step(
        self,
        output: "AlgorithmOutput",
        *,
        policy_adapter: BasePolicyAdapter,
        context: Optional[Mapping[str, Any]] = None,
    ) -> TrainerStepResult:
        """Apply one parameter update from an algorithm output."""

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer/scheduler adapter state, not model weights."""

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore optimizer/scheduler adapter state."""


def _call_supported(function, *args, **kwargs):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    if not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
    return function(*args, **kwargs)


class BasicTrainerAdapter(BaseTrainerAdapter):
    """Apply normal backward/step mechanics or a policy-specific step hook."""

    STATE_VERSION = 2

    def __init__(
        self,
        *,
        optimizer=None,
        scheduler=None,
        step_fn: Optional[Callable] = None,
        post_step_fn: Optional[Callable] = None,
    ) -> None:
        if step_fn is not None and not callable(step_fn):
            raise TypeError("step_fn must be callable")
        if post_step_fn is not None and not callable(post_step_fn):
            raise TypeError("post_step_fn must be callable")
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_fn = step_fn
        self.post_step_fn = post_step_fn
        self.step_count = 0

    @staticmethod
    def _select_component(component, key, *, name, required):
        if isinstance(component, Mapping):
            if key not in component:
                if required:
                    raise KeyError(f"{name} mapping is missing loss key {key!r}")
                return None
            return component[key]
        if key is not None and required:
            raise TypeError(f"multi-loss updates require a {name} mapping")
        return component

    def _step_loss(self, loss, *, key=None):
        if callable(loss) and not callable(getattr(loss, "backward", None)):
            loss = loss()
        optimizer = self._select_component(
            self.optimizer, key, name="optimizer", required=True
        )
        zero_grad = getattr(optimizer, "zero_grad", None)
        if not callable(zero_grad):
            raise TypeError("optimizer must provide zero_grad")
        _call_supported(zero_grad, set_to_none=True)
        backward = getattr(loss, "backward", None)
        if not callable(backward):
            raise TypeError("algorithm loss must provide backward")
        backward()
        optimizer_step = getattr(optimizer, "step", None)
        if not callable(optimizer_step):
            raise TypeError("optimizer must provide step")
        optimizer_step()
        scheduler = self._select_component(
            self.scheduler, key, name="scheduler", required=False
        )
        if scheduler is not None:
            scheduler_step = getattr(scheduler, "step", None)
            if not callable(scheduler_step):
                raise TypeError("scheduler must provide step")
            scheduler_step()

    def step(
        self,
        output: "AlgorithmOutput",
        *,
        policy_adapter: BasePolicyAdapter,
        context: Optional[Mapping[str, Any]] = None,
    ) -> TrainerStepResult:
        from rl.algorithm import AlgorithmOutput

        if not isinstance(output, AlgorithmOutput):
            raise TypeError("trainer adapter input must be AlgorithmOutput")
        if not isinstance(policy_adapter, BasePolicyAdapter):
            raise TypeError("policy_adapter must inherit BasePolicyAdapter")

        if self.step_fn is not None:
            result = _call_supported(
                self.step_fn,
                output,
                policy_adapter=policy_adapter,
                context=context,
            )
            if isinstance(result, Mapping):
                result = TrainerStepResult(metrics=dict(result), updated=True)
            if not isinstance(result, TrainerStepResult):
                raise TypeError("step_fn must return a mapping or TrainerStepResult")
        elif output.loss is None:
            result = TrainerStepResult(updated=False)
        else:
            if isinstance(output.loss, Mapping):
                if not output.loss:
                    return TrainerStepResult(updated=False)
                order = tuple(output.payload.get("update_order", output.loss))
                if set(order) != set(output.loss) or len(order) != len(output.loss):
                    raise ValueError("update_order must contain every loss key exactly once")
                for key in order:
                    self._step_loss(output.loss[key], key=key)
            else:
                if self.optimizer is None:
                    raise RuntimeError(
                        "an optimizer or step_fn is required for non-empty loss"
                    )
                self._step_loss(output.loss)
            result = TrainerStepResult(updated=True)

        if result.updated:
            if self.post_step_fn is not None:
                _call_supported(
                    self.post_step_fn,
                    output,
                    policy_adapter=policy_adapter,
                    context=context,
                )
            self.step_count += 1
        return result

    def state_dict(self):
        def component_state(component):
            if isinstance(component, Mapping):
                return {key: value.state_dict() for key, value in component.items()}
            return None if component is None else component.state_dict()

        return {
            "version": self.STATE_VERSION,
            "step_count": self.step_count,
            "optimizer": component_state(self.optimizer),
            "scheduler": component_state(self.scheduler),
        }

    def load_state_dict(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("trainer adapter state must be a mapping")
        if state.get("version") not in {1, self.STATE_VERSION}:
            raise ValueError("unsupported trainer adapter state version")
        step_count = state.get("step_count")
        if not isinstance(step_count, int) or step_count < 0:
            raise ValueError("step_count must be a non-negative integer")

        optimizer_state = state.get("optimizer")
        scheduler_state = state.get("scheduler")
        if optimizer_state is not None:
            self._load_component_state(
                self.optimizer, optimizer_state, name="optimizer"
            )
        if scheduler_state is not None:
            self._load_component_state(
                self.scheduler, scheduler_state, name="scheduler"
            )
        self.step_count = step_count

    @staticmethod
    def _load_component_state(component, state, *, name):
        if component is None:
            raise ValueError(f"state contains {name} data but adapter has none")
        if isinstance(component, Mapping):
            if not isinstance(state, Mapping) or set(state) != set(component):
                raise ValueError(f"{name} state keys do not match adapter keys")
            for key, value in component.items():
                value.load_state_dict(state[key])
        else:
            component.load_state_dict(state)


def _build_model_ema_post_step(policy_components):
    if not isinstance(policy_components, Mapping):
        return None
    model = policy_components.get("model")
    ema = getattr(model, "ema", None)
    ema_step = getattr(ema, "step", None)
    parameters = getattr(model, "parameters", None)
    if not callable(ema_step) or not callable(parameters):
        return None
    model_parameters = tuple(parameters())
    if not model_parameters:
        return None
    ema_to = getattr(ema, "to", None)
    model_device = getattr(model_parameters[0], "device", None)
    if callable(ema_to) and model_device is not None:
        _call_supported(ema_to, device=model_device)

    def update_ema(output, *, policy_adapter=None, context=None):
        del output, policy_adapter, context
        ema_step(parameters())

    return update_ema


def _build_basic_trainer_adapter(
    *,
    policy_components=None,
    native_trainer_class=None,
    **kwargs,
):
    del native_trainer_class
    if kwargs.get("post_step_fn") is None:
        post_step_fn = _build_model_ema_post_step(policy_components)
        if post_step_fn is not None:
            kwargs["post_step_fn"] = post_step_fn
    return BasicTrainerAdapter(**kwargs)


_TRAINER_ADAPTER_FACTORIES = {"basic": _build_basic_trainer_adapter}


def register_trainer_adapter(name: str, factory: Callable, *, replace=False):
    if not isinstance(name, str) or not name:
        raise TypeError("trainer adapter name must be a non-empty string")
    if not callable(factory):
        raise TypeError("trainer adapter factory must be callable")
    if name in _TRAINER_ADAPTER_FACTORIES and not replace:
        raise ValueError(f"trainer adapter {name!r} is already registered")
    _TRAINER_ADAPTER_FACTORIES[name] = factory
    return factory


def build_trainer_adapter(
    policy_module,
    policy_components: Mapping,
    *,
    adapter: str = "auto",
    native_trainer_class=None,
    policy_adapter=None,
    **kwargs,
) -> BaseTrainerAdapter:
    """Resolve a policy-local, native-Trainer, or generic update adapter."""

    if not isinstance(policy_components, Mapping):
        raise TypeError("policy_components must be a mapping")
    if "model" not in policy_components:
        raise KeyError("policy_components must contain model")
    if not isinstance(adapter, str) or not adapter:
        raise TypeError("adapter must be a non-empty string")

    result = None
    if adapter in {"auto", "policy"}:
        local_module = _load_policy_local_adapter(policy_module)
        local_factory = None
        if local_module is not None:
            local_factory = getattr(local_module, "build_trainer_adapter", None)
            if not callable(local_factory):
                local_factory = getattr(local_module, "RLTrainerAdapter", None)
        if callable(local_factory):
            local_kwargs = dict(kwargs)
            if policy_adapter is not None:
                local_kwargs["policy_adapter"] = policy_adapter
            result = local_factory(
                policy_components=policy_components,
                **local_kwargs,
            )
        elif adapter == "policy":
            raise AttributeError(
                "policy rl_adapter must define build_trainer_adapter or "
                "RLTrainerAdapter"
            )

    if result is None and adapter in {"auto", "trainer"}:
        native_factory = getattr(
            native_trainer_class,
            "build_trainer_adapter",
            None,
        )
        if callable(native_factory):
            result = native_factory(policy_components=policy_components, **kwargs)
        elif adapter == "trainer":
            raise AttributeError(
                "the native policy Trainer must define the explicit trainer "
                "adapter hook build_trainer_adapter"
            )

    if result is None:
        generic_name = "basic" if adapter == "auto" else adapter
        try:
            factory = _TRAINER_ADAPTER_FACTORIES[generic_name]
        except KeyError as exc:
            available = ", ".join(sorted(_TRAINER_ADAPTER_FACTORIES))
            raise ValueError(
                f"unknown trainer adapter {generic_name!r}; available: {available}"
            ) from exc
        result = factory(
            policy_components=policy_components,
            native_trainer_class=native_trainer_class,
            **kwargs,
        )

    if not isinstance(result, BaseTrainerAdapter):
        raise TypeError("trainer adapter factory must return BaseTrainerAdapter")
    return result
