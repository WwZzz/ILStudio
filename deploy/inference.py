"""
Inference Utilities for ILStudio.

Provides the inference worker process and SHM communication helpers for
decoupling policy inference from the environment step loop.

Architecture (Sim):
    ┌──────────────┐   obs_shm    ┌────────────────┐   action_shm   ┌──────────────┐
    │  Environment  │ ──────────> │ Inference Proc  │ ────────────> │ ActionManager │
    │  (main proc)  │  ctrl_shm   │  (subprocess)   │               │  (main proc)  │
    │              │ ──────────> │                  │               │              │
    └──────────────┘             └────────────────┘               └──────────────┘

Architecture (Real):
    ┌──────────────┐             ┌────────────────┐   action_shm   ┌──────────────┐
    │ Device SHMs   │ ──────────>│ Inference Proc  │ ────────────> │ ActionManager │
    │ (device procs)│  (direct)  │  (subprocess)   │               │  (main proc)  │
    └──────────────┘             └────────────────┘               └──────────────┘
                                  ↑ ctrl_shm (trigger)               │
                                  └──────────────────────────────────┘

Key difference: obs source is configurable (obs_shm for sim, device SHMs for real).
Trigger mechanism is unified: ctrl_shm "infer" command from action_manager.

Usage:
    # Sim mode:
    ctx = start_inference_process(model_name_or_path, ...)
    ctx.update_obs(obs, t)        # Main process writes obs
    # action_manager sends trigger internally
    
    # Real mode:
    ctx = start_inference_process(model_name_or_path, ..., 
              device_shm_names=["cam1", "robot1"], robot_module_name="robot.so101")
    # Worker reads obs from device SHMs directly, no obs_shm needed
"""

from __future__ import annotations

import os
import time
import signal
import multiprocessing as mp
from dataclasses import asdict
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
from loguru import logger

from deploy.shm_utils import SharedMemoryChannel, cleanup_all_shm


def _env_flag(name: str) -> bool:
    value = str(os.getenv(name, "")).strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _preview_array(arr: np.ndarray, max_items: int = 4) -> str:
    arr = np.asarray(arr)
    if arr.size == 0:
        return f"shape={arr.shape}, dtype={arr.dtype}, empty"
    flat = arr.reshape(-1)
    preview_items = []
    for item in flat[:max_items]:
        try:
            preview_items.append(f"{float(item):.4f}")
        except (TypeError, ValueError):
            preview_items.append(repr(item))
    return (
        f"shape={arr.shape}, dtype={arr.dtype}, "
        f"preview=[{', '.join(preview_items)}]"
    )


def _summarize_state_array(arr: np.ndarray, edge_items: int = 6) -> str:
    arr = np.asarray(arr)
    if arr.size == 0:
        return f"shape={arr.shape}, dtype={arr.dtype}, empty"
    flat = arr.reshape(-1)
    if flat.size <= edge_items * 2:
        preview = ", ".join(f"{float(x):.4f}" for x in flat)
        return f"shape={arr.shape}, dtype={arr.dtype}, values=[{preview}]"
    head = ", ".join(f"{float(x):.4f}" for x in flat[:edge_items])
    tail = ", ".join(f"{float(x):.4f}" for x in flat[-edge_items:])
    return (
        f"shape={arr.shape}, dtype={arr.dtype}, "
        f"head=[{head}], tail=[{tail}]"
    )


def _delta_summary(curr: Optional[np.ndarray], prev: Optional[np.ndarray], label: str) -> str:
    if curr is None:
        return f"{label}_delta=None"
    curr = np.asarray(curr)
    if prev is None:
        return f"{label}_delta=first_sample"
    prev = np.asarray(prev)
    if curr.shape != prev.shape:
        return f"{label}_delta=shape_change {prev.shape}->{curr.shape}"
    delta = curr - prev
    return (
        f"{label}_delta=max={float(np.max(np.abs(delta))):.4f}, "
        f"l2={float(np.linalg.norm(delta)):.4f}"
    )


def _summarize_synced_data(synced_data: Optional[Dict[str, dict]]) -> str:
    if not synced_data:
        return "no_synced_data"
    parts = []
    for name, data in synced_data.items():
        if not isinstance(data, dict):
            parts.append(f"{name}(type={type(data).__name__})")
            continue
        ts = data.get("__timestamp__", data.get("timestamp", None))
        keys = sorted(k for k in data.keys() if not k.startswith("__"))
        ts_str = "n/a" if ts is None else f"{float(ts):.6f}"
        parts.append(f"{name}(ts={ts_str}, keys={keys})")
    return "; ".join(parts)


def _summarize_mobs(mobs) -> str:
    if mobs is None:
        return "mobs=None"
    state_summary = "state=None"
    if getattr(mobs, "state", None) is not None:
        state_summary = f"state={_summarize_state_array(mobs.state)}"
    image = getattr(mobs, "image", None)
    image_summary = "image=None"
    if image is not None:
        image_summary = f"image_shape={np.asarray(image).shape}"
    timestep = getattr(mobs, "timestep", None)
    timestep_summary = f"timestep={np.asarray(timestep).tolist()}" if timestep is not None else "timestep=None"
    return f"{state_summary}; {image_summary}; {timestep_summary}"


def _summarize_mact_list(mact_list: list) -> str:
    if not mact_list:
        return "chunk_len=0"

    first_action = _extract_step_action(mact_list[0])
    last_action = _extract_step_action(mact_list[-1])
    parts = [f"chunk_len={len(mact_list)}"]
    if first_action is not None:
        parts.append(f"first={_preview_array(first_action)}")
    if last_action is not None:
        parts.append(f"last={_preview_array(last_action)}")
    return "; ".join(parts)


def _extract_step_action(step):
    if isinstance(step, np.ndarray) and step.dtype == object and len(step) > 0:
        first = step[0]
        if isinstance(first, dict):
            return first.get("action", None)
    if isinstance(step, np.ndarray):
        return step
    return None


# ---------------------------------------------------------------------------
# Inference Process (runs in subprocess)
# ---------------------------------------------------------------------------

def inference_worker(
    model_name_or_path: str,
    action_shm_name: str,
    ctrl_shm_name: str,
    # Sim mode: read obs from obs_shm (written by main process)
    obs_shm_name: str = None,
    # Real mode: read obs from device SHMs directly
    device_shm_names: List[str] = None,
    device_configs: List[dict] = None,
    robot_module_name: str = None,
    sync_buffer_maxlen: int = 100,
    sync_max_tolerance_s: float = 0.05,
    # Common params
    action_shm_size_mb: int = 32,
    device: str = "cuda",
    dataset_id: str = "",
    chunk_size: int = -1,
):
    """
    Unified inference worker for both sim and real modes.
    
    Waits for "infer" trigger from ctrl_shm, then:
    - Sim mode: reads latest obs from obs_shm
    - Real mode: reads latest obs from device SHMs via synchronizer + obs2meta
    
    Runs policy inference and writes action chunks to action_shm.
    """
    # Re-apply resource_tracker patch in subprocess
    from deploy.shm_utils import _fix_resource_tracker
    _fix_resource_tracker()
    
    import configs  # noqa: side-effects
    from data_utils.utils import set_seed
    set_seed(0)
    
    is_real_mode = device_shm_names is not None and len(device_shm_names) > 0
    mode_str = "REAL" if is_real_mode else "SIM"
    
    logger.info(f"[InferenceWorker-{mode_str}] Starting (PID={os.getpid()})")
    logger.info(f"[InferenceWorker-{mode_str}] Model: {model_name_or_path}")
    logger.info(f"[InferenceWorker-{mode_str}] Device: {device}")
    
    # Load policy
    from policy.utils import load_policy
    from types import SimpleNamespace
    args = SimpleNamespace(
        model_name_or_path=model_name_or_path,
        device=device,
        dataset_id=dataset_id,
        chunk_size=chunk_size,
        is_training=False,
    )
    policy = load_policy(args)
    logger.info(f"[InferenceWorker-{mode_str}] Policy loaded successfully")
    
    # --- Setup obs source ---
    obs_reader = None
    synchronizer = None
    obs2meta_func = None
    
    if is_real_mode:
        # Real mode: connect to device SHMs + synchronizer
        from deploy.shm_utils import SharedMemoryDataSynchronizer
        from deploy.base import create_obs2meta_func, default_obs2meta
        
        shm_channels = []
        for name in device_shm_names:
            try:
                ch = SharedMemoryChannel(name, is_writer=False, timeout=30.0)
                shm_channels.append((name, ch))
                logger.info(f"[InferenceWorker-REAL] Connected to device SHM: {name}")
            except Exception as e:
                logger.warning(f"[InferenceWorker-REAL] Failed to connect to {name}: {e}")
        
        if not shm_channels:
            logger.error("[InferenceWorker-REAL] No device SHM channels connected. Exiting.")
            return
        
        synchronizer = SharedMemoryDataSynchronizer(
            shm_channels=shm_channels,
            buffer_maxlen=sync_buffer_maxlen,
            max_tolerance_s=sync_max_tolerance_s,
        )
        
        if device_configs:
            obs2meta_func = create_obs2meta_func(device_configs)
        else:
            obs2meta_func = default_obs2meta
            logger.info("[InferenceWorker-REAL] No device_configs, using default obs2meta")
    else:
        # Sim mode: read from obs_shm
        if obs_shm_name is None:
            logger.error("[InferenceWorker-SIM] obs_shm_name is required in sim mode. Exiting.")
            return
        obs_reader = SharedMemoryChannel(name=obs_shm_name, is_writer=False, timeout=30.0)
        logger.info(f"[InferenceWorker-SIM] Connected to obs SHM: {obs_shm_name}")
    
    # --- Setup action and ctrl channels ---
    action_writer = SharedMemoryChannel(name=action_shm_name, max_size_mb=action_shm_size_mb, is_writer=True)
    logger.info(f"[InferenceWorker-{mode_str}] Created action SHM: {action_shm_name}")
    
    ctrl_reader = SharedMemoryChannel(name=ctrl_shm_name, is_writer=False, timeout=30.0)
    logger.info(f"[InferenceWorker-{mode_str}] Connected to ctrl SHM: {ctrl_shm_name}")
    
    running = True
    inference_count = 0
    infer_debug = _env_flag("ILSTUDIO_INFER_DEBUG")
    print_full_state = _env_flag("ILSTUDIO_INFER_PRINT_STATE")
    last_obs_summary = "obs=unavailable"
    prev_state_snapshot = None
    prev_first_action = None
    prev_last_action = None
    
    def _handle_signal(signum, frame):
        nonlocal running
        running = False
    
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    
    logger.info(f"[InferenceWorker-{mode_str}] Ready, waiting for triggers...")
    
    def _get_obs_sim():
        """Read obs from obs_shm (sim mode)."""
        nonlocal last_obs_summary
        from benchmark.base import dict2meta
        obs_data = obs_reader.read(blocking=False)
        if obs_data is None:
            last_obs_summary = "obs=None"
            return None, 0
        t = obs_data.get("t", 0)
        mobs_dict = {k: v for k, v in obs_data.items() if not k.startswith("__") and k != "t"}
        mobs = dict2meta(mobs_dict, mtype="obs")
        last_obs_summary = _summarize_mobs(mobs)
        return mobs, t
    
    def _get_obs_real():
        """Read obs from device SHMs via synchronizer (real mode)."""
        nonlocal last_obs_summary
        # For trigger-based real-world inference, using get_synced_frame_blocking()
        # would return the earliest unread synchronized frame since the previous
        # inference. With chunked execution this can lag by almost one whole chunk.
        # Here we instead wait until every device has at least one fresh sample and
        # then consume the newest unread sample from each device.
        synced_data = None
        while running and synced_data is None:
            synced_data = synchronizer.get_latest_frame_nonblocking(strict_new_only=True)
            if synced_data is None:
                time.sleep(0.001)
        if synced_data is None:
            last_obs_summary = "synced_data=None"
            return None, 0
        mobs = obs2meta_func(synced_data)
        last_obs_summary = (
            f"devices={_summarize_synced_data(synced_data)} | "
            f"mobs={_summarize_mobs(mobs)}"
        )
        return mobs, 0
    
    get_obs = _get_obs_real if is_real_mode else _get_obs_sim
    
    try:
        with torch.inference_mode():
            while running:
                # Real mode: always keep synchronizer buffer updated
                if synchronizer is not None:
                    synchronizer.read_and_buffer()
                
                # Check for control signals (non-blocking)
                ctrl_data = ctrl_reader.read(blocking=False, skip_unchanged=True)
                if ctrl_data is not None:
                    cmd = ctrl_data.get("cmd", "")
                    if cmd == "stop":
                        logger.info(f"[InferenceWorker-{mode_str}] Received stop command")
                        break
                    elif cmd == "reset":
                        policy.reset()
                        logger.info(f"[InferenceWorker-{mode_str}] Policy reset")
                        continue
                    elif cmd == "infer":
                        # Carry trigger metadata through inference.
                        trigger_epoch = ctrl_data.get("epoch", 0)
                        trigger_t = int(ctrl_data.get("t", 0))
                        
                        # Get latest observation
                        mobs, obs_t = get_obs()
                        if mobs is None:
                            if infer_debug:
                                logger.info(
                                    "[InferenceWorker-{}] Skip infer epoch={} t={} because obs unavailable: {}",
                                    mode_str,
                                    trigger_epoch,
                                    trigger_t,
                                    last_obs_summary,
                                )
                            else:
                                logger.debug(f"[InferenceWorker-{mode_str}] No obs available at trigger time")
                            continue
                        t = trigger_t if is_real_mode else obs_t
                        
                        # Set timestep
                        if mobs.state is not None:
                            batch_size = mobs.state.shape[0] if mobs.state.ndim > 1 else 1
                        else:
                            batch_size = 1
                        mobs.timestep = np.array([[t] for _ in range(batch_size)])
                        
                        # Run inference
                        t_start = time.perf_counter()
                        mact_list = policy.inference(mobs)
                        t_end = time.perf_counter()
                        
                        inference_count += 1
                        latency_ms = (t_end - t_start) * 1000
                        chunk_summary = _summarize_mact_list(mact_list)
                        state_snapshot = None if mobs.state is None else np.array(mobs.state, copy=True)
                        first_action = _extract_step_action(mact_list[0]) if mact_list else None
                        last_action = _extract_step_action(mact_list[-1]) if mact_list else None
                        infer_delta_summary = "; ".join([
                            _delta_summary(state_snapshot, prev_state_snapshot, "state"),
                            _delta_summary(first_action, prev_first_action, "first_action"),
                            _delta_summary(last_action, prev_last_action, "last_action"),
                        ])
                        
                        # Serialize and write (include epoch for staleness detection)
                        action_payload = serialize_mact_list(mact_list)
                        action_payload["t"] = t
                        action_payload["epoch"] = trigger_epoch
                        action_payload["latency_ms"] = latency_ms
                        action_writer.write(action_payload)

                        if infer_debug:
                            logger.info(
                                "[InferenceWorker-{}] Infer #{} epoch={} latency={:.1f}ms | {} | {}",
                                mode_str,
                                inference_count,
                                trigger_epoch,
                                latency_ms,
                                last_obs_summary,
                                chunk_summary,
                            )
                            logger.info(
                                "[InferenceWorker-{}] Infer #{} deltas | {}",
                                mode_str,
                                inference_count,
                                infer_delta_summary,
                            )
                            if print_full_state and mobs.state is not None:
                                logger.info(
                                    "[InferenceWorker-{}] Full mobs.state=\n{}",
                                    mode_str,
                                    np.array2string(np.asarray(mobs.state), precision=5, suppress_small=True),
                                )
                        prev_state_snapshot = state_snapshot
                        prev_first_action = None if first_action is None else np.array(first_action, copy=True)
                        prev_last_action = None if last_action is None else np.array(last_action, copy=True)
                        
                        if inference_count % 50 == 0:
                            logger.debug(f"[InferenceWorker-{mode_str}] Inference #{inference_count}, "
                                       f"latency={latency_ms:.1f}ms, chunk_size={len(mact_list)}")
                else:
                    time.sleep(0.0001)  # 0.1ms idle sleep
    
    except Exception as e:
        logger.error(f"[InferenceWorker-{mode_str}] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if obs_reader is not None:
            obs_reader.destroy()
        if synchronizer is not None:
            for _, ch in synchronizer.shm_channels:
                try:
                    ch.destroy()
                except Exception:
                    pass
        action_writer.destroy()
        ctrl_reader.destroy()
        logger.info(f"[InferenceWorker-{mode_str}] Stopped (processed {inference_count} inferences)")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def serialize_mact_list(mact_list: list) -> dict:
    """
    Serialize mact_list for SharedMemory transmission.
    
    mact_list is List[np.ndarray(dtype=object)] where each np.ndarray contains
    dicts like {'action': np.ndarray, 'ctrl_type': str, 'ctrl_space': str}.
    
    We flatten it into a dict of numpy arrays for efficient SHM transfer.
    """
    all_actions = []
    ctrl_type = "delta"
    ctrl_space = "ee"
    
    for step_arr in mact_list:
        step_actions = []
        for item in step_arr:
            if isinstance(item, dict):
                act = item.get("action", None)
                if act is not None:
                    step_actions.append(act)
                ctrl_type = item.get("ctrl_type", ctrl_type)
                ctrl_space = item.get("ctrl_space", ctrl_space)
        if step_actions:
            all_actions.append(np.stack(step_actions))
    
    if all_actions:
        actions_array = np.stack(all_actions)
    else:
        actions_array = np.zeros((0,))
    
    return {
        "actions": actions_array,
        "chunk_len": len(mact_list),
        "ctrl_type": ctrl_type,
        "ctrl_space": ctrl_space,
    }


def deserialize_mact_list(payload: dict) -> list:
    """
    Deserialize SHM payload back to mact_list format.
    
    Returns:
        List[np.ndarray(dtype=object)] matching the format from MetaPolicy.inference()
    """
    actions_array = payload.get("actions", None)
    if actions_array is None or (isinstance(actions_array, np.ndarray) and actions_array.size == 0):
        return []
    
    chunk_len = payload.get("chunk_len", 0)
    ctrl_type = payload.get("ctrl_type", "delta")
    ctrl_space = payload.get("ctrl_space", "ee")
    
    mact_list = []
    for i in range(min(chunk_len, len(actions_array))):
        step_actions = actions_array[i]
        step_dicts = []
        if step_actions.ndim == 1:
            step_dicts.append({
                "action": np.array(step_actions, copy=True),
                "ctrl_type": ctrl_type,
                "ctrl_space": ctrl_space,
            })
        else:
            for j in range(step_actions.shape[0]):
                step_dicts.append({
                    "action": np.array(step_actions[j], copy=True),
                    "ctrl_type": ctrl_type,
                    "ctrl_space": ctrl_space,
                })
        mact_list.append(np.array(step_dicts, dtype=object))
    
    return mact_list


# ---------------------------------------------------------------------------
# Process lifecycle management
# ---------------------------------------------------------------------------

class InferenceContext:
    """
    Context for managing inference process and SHM channels.
    
    Created by start_inference_process(), used by evaluate() and action_manager,
    cleaned up by stop_inference_process().
    
    Obs and trigger are decoupled:
    - update_obs(): main process writes obs to obs_shm (sim mode only)
    - send_trigger(): action_manager requests inference via ctrl_shm
    """
    def __init__(
        self,
        process: mp.Process,
        obs_writer: Optional[SharedMemoryChannel],  # None in real mode
        action_reader: SharedMemoryChannel,
        ctrl_writer: SharedMemoryChannel,
        shm_names: tuple,
    ):
        self.process = process
        self.obs_writer = obs_writer
        self.action_reader = action_reader
        self.ctrl_writer = ctrl_writer
        self.shm_names = shm_names
        self._epoch = 0  # Incremented on reset; used to discard stale chunks
    
    def update_obs(self, mobs, t: int) -> None:
        """
        Write observation to obs_shm (sim mode only).
        
        In real mode (obs_writer is None), this is a no-op because
        the inference worker reads from device SHMs directly.
        """
        if self.obs_writer is None:
            return
        obs_dict = asdict(mobs)
        obs_dict["t"] = t
        self.obs_writer.write(obs_dict)
    
    def send_trigger(self, t: int = 0) -> None:
        """Send inference trigger to worker via ctrl_shm."""
        self.ctrl_writer.write({"cmd": "infer", "t": t, "epoch": self._epoch})
    
    def poll_action_chunks(self) -> Optional[list]:
        """
        Non-blocking check for new action chunks from inference process.
        
        Automatically discards stale chunks from previous epochs (rollouts).
        
        Returns:
            mact_list if new chunk available, None otherwise.
        """
        action_data = self.action_reader.read(blocking=False, skip_unchanged=True)
        if action_data is None:
            return None
        
        # Epoch check: discard stale chunks from previous rollouts
        chunk_epoch = action_data.get("epoch", 0)
        if chunk_epoch != self._epoch:
            logger.debug(f"Discarding stale action chunk (epoch {chunk_epoch} != current {self._epoch})")
            return None
        
        return deserialize_mact_list(action_data)
    
    def send_reset(self) -> None:
        """Send reset command to inference process and increment epoch."""
        self._epoch += 1
        self.ctrl_writer.write({"cmd": "reset", "epoch": self._epoch})
    
    def send_stop(self) -> None:
        """Send stop command to inference process."""
        self.ctrl_writer.write({"cmd": "stop"})
    
    def is_alive(self) -> bool:
        """Check if inference process is still running."""
        return self.process is not None and self.process.is_alive()


def start_inference_process(
    model_name_or_path: str,
    device: str = "cuda",
    dataset_id: str = "",
    chunk_size: int = -1,
    # Sim mode params
    obs_shm_size_mb: int = 64,
    # Real mode params
    device_shm_names: List[str] = None,
    device_configs: List[dict] = None,
    robot_module_name: str = None,
    sync_buffer_maxlen: int = 100,
    sync_max_tolerance_s: float = 0.05,
    # Common params
    action_shm_size_mb: int = 32,
    ctrl_shm_size_mb: int = 1,
    shm_prefix: str = "infer",
) -> InferenceContext:
    """
    Start the inference subprocess and create SHM channels.
    
    Args:
        model_name_or_path: Path to model checkpoint.
        device: Device for inference.
        dataset_id: Dataset ID for normalizer loading.
        chunk_size: Passed through to policy loading / MetaPolicy for compatibility;
                    MetaPolicy no longer truncates chunks (action_manager handles it).
        obs_shm_size_mb: Size of obs SHM in MB (sim mode only).
        device_shm_names: Device SHM names (real mode). If provided, enables real mode.
        device_configs: Device config dicts (real mode). Used to build composite obs2meta.
        robot_module_name: Robot module name for obs2meta (real mode, legacy fallback).
        sync_buffer_maxlen: Synchronizer buffer size (real mode).
        sync_max_tolerance_s: Synchronizer tolerance (real mode).
        action_shm_size_mb: Size of action SHM in MB.
        ctrl_shm_size_mb: Size of control SHM in MB.
        shm_prefix: Prefix for SHM channel names.
    
    Returns:
        InferenceContext for use by action_manager and main process.
    """
    is_real_mode = device_shm_names is not None and len(device_shm_names) > 0
    mode_str = "REAL" if is_real_mode else "SIM"
    
    logger.info(f"🚀 Starting inference process ({mode_str} mode)")
    logger.info(f"   Model: {model_name_or_path}")
    logger.info(f"   Device: {device}")
    
    # Generate SHM names
    pid = os.getpid()
    obs_shm_name = f"{shm_prefix}_{pid}_obs" if not is_real_mode else None
    action_shm_name = f"{shm_prefix}_{pid}_action"
    ctrl_shm_name = f"{shm_prefix}_{pid}_ctrl"
    
    all_shm_names = [action_shm_name, ctrl_shm_name]
    if obs_shm_name:
        all_shm_names.append(obs_shm_name)
    
    # Create SHM channels in main process
    obs_writer = None
    if not is_real_mode:
        obs_writer = SharedMemoryChannel(
            name=obs_shm_name,
            max_size_mb=obs_shm_size_mb,
            is_writer=True,
        )
    
    ctrl_writer = SharedMemoryChannel(
        name=ctrl_shm_name,
        max_size_mb=ctrl_shm_size_mb,
        is_writer=True,
    )
    
    # Start inference subprocess
    worker_kwargs = dict(
        model_name_or_path=model_name_or_path,
        action_shm_name=action_shm_name,
        ctrl_shm_name=ctrl_shm_name,
        action_shm_size_mb=action_shm_size_mb,
        device=device,
        dataset_id=dataset_id,
        chunk_size=chunk_size,
    )
    
    if is_real_mode:
        worker_kwargs.update(
            device_shm_names=device_shm_names,
            device_configs=device_configs,
            robot_module_name=robot_module_name,
            sync_buffer_maxlen=sync_buffer_maxlen,
            sync_max_tolerance_s=sync_max_tolerance_s,
        )
    else:
        worker_kwargs["obs_shm_name"] = obs_shm_name
    
    # Use 'spawn' context for the inference process: CUDA cannot be
    # re-initialized in a forked subprocess (the main process may have
    # already touched CUDA via set_seed / torch.cuda.is_available).
    # Only the inference process needs spawn; device processes stay on fork.
    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=inference_worker,
        kwargs=worker_kwargs,
        daemon=True,
    )
    process.start()
    logger.info(f"   Inference process started (PID={process.pid})")
    
    # Connect to action SHM (created by inference process)
    action_reader = None
    for attempt in range(300):  # Up to 30 seconds
        try:
            action_reader = SharedMemoryChannel(
                name=action_shm_name,
                is_writer=False,
                timeout=0.1,
            )
            break
        except (TimeoutError, FileNotFoundError):
            if not process.is_alive():
                # Cleanup on failure
                if obs_writer is not None:
                    obs_writer.destroy()
                ctrl_writer.destroy()
                cleanup_all_shm(all_shm_names)
                raise RuntimeError("Inference process died during startup")
            time.sleep(0.1)
    else:
        if obs_writer is not None:
            obs_writer.destroy()
        ctrl_writer.destroy()
        cleanup_all_shm(all_shm_names)
        raise RuntimeError("Timeout waiting for inference process to create action SHM")
    
    logger.info(f"✓ Inference process ready ({mode_str} mode)")
    
    return InferenceContext(
        process=process,
        obs_writer=obs_writer,
        action_reader=action_reader,
        ctrl_writer=ctrl_writer,
        shm_names=tuple(all_shm_names),
    )


def stop_inference_process(ctx: InferenceContext) -> None:
    """
    Stop the inference process and cleanup SHM.
    """
    if ctx is None:
        return
    
    logger.info("Stopping inference process...")
    
    # Send stop command
    try:
        ctx.send_stop()
    except Exception:
        pass
    
    # Wait for process to finish
    if ctx.process is not None and ctx.process.is_alive():
        ctx.process.join(timeout=5.0)
        if ctx.process.is_alive():
            logger.warning("Inference process did not stop gracefully, terminating...")
            ctx.process.terminate()
            ctx.process.join(timeout=2.0)
    
    # Cleanup SHM
    if ctx.obs_writer is not None:
        ctx.obs_writer.destroy()
    
    if ctx.ctrl_writer is not None:
        ctx.ctrl_writer.destroy()
    
    if ctx.action_reader is not None:
        ctx.action_reader.destroy()
    
    # Clean up any residual SHM
    cleanup_all_shm(list(ctx.shm_names))
    
    logger.info("✓ Inference process stopped")
