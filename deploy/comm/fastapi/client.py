#!/usr/bin/env python3
"""
FastAPI-based Policy Client for Remote Inference (HTTP/JSON).

This is an HTTP/JSON alternative to the TCP+pickle client in `deploy/comm/client.py`.

Key requirement:
- Client send function accepts `MetaObs`, converts to dict, sends via FastAPI POST
- Server returns actions, client converts back into the same structure used by existing `PolicyClient`
  (a list of numpy object arrays where each element is a dict compatible with MetaAction).

Note:
- SSL certificate verification is disabled by default to support self-signed certificates.
- Each client instance has a unique `client_id` for server-side request deduplication.
"""

from __future__ import annotations

import base64
import os
import uuid
from collections import deque
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import urllib3
from loguru import logger

from benchmark.base import MetaObs

from ..base import BaseClient

# Suppress InsecureRequestWarning globally for this module (self-signed certs are common)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


NDARRAY_TYPE_TAG = "__type__"
NDARRAY_TYPE_NAME = "ndarray"


def _encode_ndarray(arr: np.ndarray) -> Dict[str, Any]:
    return {
        NDARRAY_TYPE_TAG: NDARRAY_TYPE_NAME,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes(order="C")).decode("ascii"),
    }


def _decode_ndarray(obj: Dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(obj["dtype"])
    shape = tuple(int(x) for x in obj["shape"])
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(shape)


def _to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        # object arrays can contain Python dicts; do not base64-encode raw pointers
        if x.dtype == object:
            return [_to_jsonable(v) for v in x.tolist()]
        return _encode_ndarray(x)
    if isinstance(x, (np.generic,)):
        return x.item()
    if hasattr(x, "__dataclass_fields__"):
        return _to_jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _from_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, dict):
        if x.get(NDARRAY_TYPE_TAG) == NDARRAY_TYPE_NAME:
            return _decode_ndarray(x)
        return {k: _from_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_from_jsonable(v) for v in x]
    return x


def _normalize_mact_list(decoded: Any) -> List[np.ndarray]:
    """
    Convert decoded JSON response into the same structure returned by MetaPolicy.inference:
      List[np.ndarray(dtype=object)] where each element holds per-batch dict(s) describing MetaAction.
    """
    if decoded is None:
        return []
    if not isinstance(decoded, list):
        raise ValueError(f"Expected mact_list to be a list, got {type(decoded)}")

    out: List[np.ndarray] = []
    for step in decoded:
        # Each step is typically a list of dicts (batch actions) or already a numpy object array.
        if isinstance(step, np.ndarray):
            out.append(step.astype(object))
        elif isinstance(step, list):
            out.append(np.array(step, dtype=object))
        else:
            # Fallback: single action dict
            out.append(np.array([step], dtype=object))
    return out


class FastAPIPolicyClient(BaseClient):
    """
    HTTP/JSON policy client using FastAPI server (drop-in for TCP PolicyClient).

    `select_action` follows the same chunked-action semantics:
    - when queue is empty OR (t % chunk_size == 0), request a new chunk
    - otherwise pop from queue
    
    Each client instance has a unique `client_id` for server-side request deduplication:
    - When requests arrive faster than the server can process, only the LATEST
      observation from each client is used for inference.
    - Earlier requests receive the same action result as the latest one.
    - This prevents stale observations from being processed.
    """

    def __init__(
        self,
        base_url: str,
        chunk_size: Optional[int] = None,
        timeout_s: float = 30.0,
        ctrl_space: str = "ee",
        ctrl_type: str = "delta",
        client_id: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.chunk_size = chunk_size
        self.timeout_s = timeout_s
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.action_queue = deque(maxlen=chunk_size) if (chunk_size is not None and chunk_size > 0) else deque(maxlen=1000)
        
        # Generate unique client_id for request deduplication
        if client_id is None:
            client_id = f"fastapi_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.client_id = client_id

        self._session = requests.Session()
        # Disable SSL verification to support self-signed certificates
        self._session.verify = False
        logger.info(f"✓ Connected to FastAPI policy server at {self.base_url}")
        logger.info(f"   Client ID: {self.client_id}")

    def health_check(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=self.timeout_s)
            return r.status_code == 200
        except Exception:
            return False

    def send_meta_obs(
        self,
        meta_obs: MetaObs,
        *,
        episode_id: Optional[int] = None,
        index: Optional[int] = None,
        reasoning: Any = None,
        timestamp: Optional[float] = None,
    ) -> List[np.ndarray]:
        payload: Dict[str, Any] = {
            "meta_obs": _to_jsonable(meta_obs),
            "client_id": self.client_id,  # For server-side request deduplication
        }
        if episode_id is not None:
            payload["episode_id"] = episode_id
        if index is not None:
            payload["__index__"] = index
        if reasoning is not None:
            payload["reasoning"] = _to_jsonable(reasoning)
        if timestamp is not None:
            payload["timestamp"] = timestamp

        r = self._session.post(f"{self.base_url}/inference", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        decoded = _from_jsonable(data.get("mact_list"))
        return _normalize_mact_list(decoded)

    def is_action_queue_empty(self) -> bool:
        return len(self.action_queue) == 0

    def select_action(self, mobs: MetaObs, t: int, return_all: bool = False):
        should_infer = len(self.action_queue) == 0
        if self.chunk_size is not None and self.chunk_size > 0:
            should_infer = should_infer or (t % self.chunk_size == 0)

        if should_infer:
            # Match MetaPolicy behavior: timestep is shaped (B, 1) when batching is used.
            if hasattr(mobs, "state") and mobs.state is not None and isinstance(mobs.state, np.ndarray):
                batch_size = mobs.state.shape[0] if (mobs.state.ndim > 1) else 1
            else:
                batch_size = 1
            mobs.timestep = np.array([[t] for _ in range(batch_size)])

            mact_list = self.send_meta_obs(mobs)
            if not mact_list:
                raise RuntimeError("Server returned empty action list")

            self.action_queue.clear()
            actions_to_add = mact_list[: self.chunk_size] if (self.chunk_size is not None and self.chunk_size > 0) else mact_list
            for mact in actions_to_add:
                self.action_queue.append(mact)

        if return_all:
            all_macts = []
            while len(self.action_queue) > 0:
                all_macts.append(self.action_queue.popleft())
            return np.concatenate(all_macts) if all_macts else np.array([])

        if len(self.action_queue) == 0:
            raise RuntimeError("Action queue is empty and server request failed")
        return self.action_queue.popleft()

    def reset(self):
        self.action_queue.clear()

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

