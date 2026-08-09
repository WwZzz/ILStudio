"""Backward-compatible action manager facade for evaluation pipelines."""

from abc import ABC, abstractmethod

from deploy.action_manager.chunk import (
    AbstractActionChunkManager,
    BasicActionChunkManager,
)
from deploy.executor import EvalPolicyExecutor


class AbstractActionManager(AbstractActionChunkManager, ABC):
    """Legacy evaluation-facing action manager interface.

    Existing evaluation code intentionally keeps using one object. Concrete
    implementations may delegate inference transport to a policy executor while
    inheriting their local buffering behavior from an action chunk manager.
    """

    @abstractmethod
    def select_action(self):
        """Return one action step through the configured inference transport."""


class BasicActionManager(BasicActionChunkManager, AbstractActionManager):
    """Compatibility facade used by ``eval_sim`` and ``eval_real``.

    The object still exposes the original action manager API. Internally, chunk
    behavior is inherited from ``BasicActionChunkManager`` and SHM interaction
    is delegated to ``EvalPolicyExecutor``.
    """

    def __init__(self, debug: bool = False, **kwargs):
        BasicActionChunkManager.__init__(self, debug=debug, **kwargs)
        self._executor = EvalPolicyExecutor(self)

    @property
    def executor(self) -> EvalPolicyExecutor:
        return self._executor

    @property
    def _inference_ctx(self):
        """Compatibility view for code that inspected the old protected field."""
        return self._executor.inference_context

    @_inference_ctx.setter
    def _inference_ctx(self, inference_ctx):
        self._executor.set_inference_context(inference_ctx)

    def set_inference_context(self, inference_ctx):
        self._executor.set_inference_context(inference_ctx)

    def select_action(self):
        return self._executor.select_action()

    def _wait_for_action_chunk(self, timeout: float = 30.0):
        """Compatibility shim for the previous protected helper."""
        return self._executor.wait_for_action_chunk(timeout=timeout)

    def reset(self):
        # Preserve the old ordering: clear local actions, reset the policy worker,
        # then report and clear per-rollout statistics.
        self._reset_buffer_state()
        self._executor.reset()
        self._report_and_reset_stats()
