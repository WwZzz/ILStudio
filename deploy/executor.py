"""Policy execution transports used by ILStudio deployment pipelines."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from deploy.action_manager.chunk import BasicActionChunkManager
    from deploy.inference import InferenceContext


class EvalPolicyExecutor:
    """Drive evaluation-time policy inference through shared memory.

    This class owns only the communication and scheduling flow around
    ``InferenceContext``. Action chunk storage, replacement, aggregation, and
    single-step dispatch remain the responsibility of the injected action
    chunk manager.
    """

    def __init__(self, action_chunk_manager: "BasicActionChunkManager"):
        self.action_chunk_manager = action_chunk_manager
        self._inference_ctx: Optional["InferenceContext"] = None

    @property
    def inference_context(self) -> Optional["InferenceContext"]:
        return self._inference_ctx

    def set_inference_context(self, inference_ctx: "InferenceContext") -> None:
        self._inference_ctx = inference_ctx
        requested_chunk_size = getattr(inference_ctx, "chunk_size", None)
        if requested_chunk_size is not None:
            self.action_chunk_manager.set_chunk_size(requested_chunk_size)
        self.action_chunk_manager._log_debug("bound inference context")

    def select_action(self):
        """Poll, trigger, wait when necessary, and emit one action step."""
        if self._inference_ctx is None:
            raise RuntimeError(
                "InferenceContext not set. Call set_inference_context() first."
            )

        manager = self.action_chunk_manager

        # Poll first so an asynchronously produced chunk can replace or merge
        # the currently executing chunk before this step is emitted.
        mact_list = self._inference_ctx.poll_action_chunks()
        if mact_list:
            mact_list = manager.prepare_chunk(mact_list)
            manager._stats["chunks_received"] += 1
            step_idx, chunk_len = manager._buffer_status()
            manager._log_debug(
                "received chunk_len={} while buffer_step={}/{}",
                len(mact_list),
                step_idx,
                chunk_len,
            )
            manager.put(mact_list, timestamp=time.perf_counter())

        buffer_empty = manager.is_empty()
        need_infer = manager.should_infer()
        if buffer_empty or need_infer:
            self._inference_ctx.send_trigger(manager.t)
            manager._stats["triggers_sent"] += 1
            manager._log_debug(
                "sent trigger at t={} reason={}",
                manager.t,
                "buffer_empty" if buffer_empty else "should_infer",
            )

        if buffer_empty:
            self.wait_for_action_chunk(timeout=3600.0)

        action = manager.get()
        if action is None:
            raise RuntimeError("ActionManager returned None - no action available")
        emit_step, emit_chunk_len = manager._buffer_status()
        manager._log_debug(
            "emit action t={} buffer_step={}/{} {}",
            manager.t,
            emit_step,
            emit_chunk_len,
            manager._summarize_action(action),
        )
        manager.t += 1
        return action

    def wait_for_action_chunk(self, timeout: float = 30.0) -> None:
        """Block until the SHM worker publishes a non-empty action chunk."""
        if self._inference_ctx is None:
            raise RuntimeError(
                "InferenceContext not set. Call set_inference_context() first."
            )

        manager = self.action_chunk_manager
        t_start = time.perf_counter()
        manager._stats["waits"] += 1

        while manager.is_empty():
            mact_list = self._inference_ctx.poll_action_chunks()
            if mact_list:
                mact_list = manager.prepare_chunk(mact_list)
                wait_ms = (time.perf_counter() - t_start) * 1000
                manager._stats["cumulative_wait_ms"] += wait_ms
                manager._stats["chunks_received"] += 1
                manager._log_debug(
                    "waited {:.1f}ms for chunk_len={}", wait_ms, len(mact_list)
                )
                manager.put(mact_list, timestamp=time.perf_counter())
                break

            elapsed = time.perf_counter() - t_start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Timeout ({timeout}s) waiting for inference result. "
                    "Check if inference process is alive."
                )
            if not self._inference_ctx.is_alive():
                raise RuntimeError("Inference process died unexpectedly")
            time.sleep(0.0001)

    def reset(self) -> None:
        """Reset the remote evaluation policy, if one has been bound."""
        if self._inference_ctx is not None:
            self._inference_ctx.send_reset()
