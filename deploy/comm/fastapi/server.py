#!/usr/bin/env python3
"""
FastAPI-based Policy Server for Remote Inference (HTTP/JSON).

Client sends a JSON payload containing a serialized MetaObs (numpy arrays are base64-encoded).
Server decodes payload -> MetaObs -> MetaPolicy.inference(...) -> returns actions as JSON.

This is an HTTP/JSON alternative to the TCP+pickle server in `deploy/comm/server.py`.
"""

from __future__ import annotations

import argparse
import base64
import os
import signal
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from loguru import logger

import configs  # noqa: F401 (kept for side-effects/config registration)
from benchmark.base import MetaObs, dict2meta
from data_utils.normalize import load_normalizers
from data_utils.utils import set_seed

from ..base import BaseServer


NDARRAY_TYPE_TAG = "__type__"
NDARRAY_TYPE_NAME = "ndarray"


def _encode_ndarray(arr: np.ndarray) -> Dict[str, Any]:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(arr)}")
    return {
        NDARRAY_TYPE_TAG: NDARRAY_TYPE_NAME,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes(order="C")).decode("ascii"),
    }


def _decode_ndarray(obj: Dict[str, Any]) -> np.ndarray:
    try:
        dtype = np.dtype(obj["dtype"])
        shape = tuple(int(x) for x in obj["shape"])
        raw = base64.b64decode(obj["data"])
    except Exception as e:
        raise ValueError(f"Invalid ndarray payload: {e}") from e
    arr = np.frombuffer(raw, dtype=dtype)
    try:
        return arr.reshape(shape)
    except Exception as e:
        raise ValueError(f"Invalid ndarray shape {shape} for buffer: {e}") from e


def _to_jsonable(x: Any) -> Any:
    """
    Convert arbitrary Python objects to JSON-serializable payload.
    - numpy arrays -> tagged dict with base64 buffer
    - numpy scalars -> Python scalars
    - dataclasses -> dict
    - lists/tuples/dicts -> recursively converted
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        # IMPORTANT: object arrays (e.g., MetaPolicy outputs) contain Python dicts.
        # Encoding them via `.tobytes()` would serialize memory pointers and is not portable.
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
    # Numpy object arrays often appear in MetaPolicy outputs.
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _from_jsonable(x: Any) -> Any:
    """
    Inverse of `_to_jsonable` for data we emit/accept.
    """
    if x is None:
        return None
    if isinstance(x, dict):
        if x.get(NDARRAY_TYPE_TAG) == NDARRAY_TYPE_NAME:
            return _decode_ndarray(x)
        return {k: _from_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_from_jsonable(v) for v in x]
    return x


def _payload_to_metaobs(meta_obs_payload: Dict[str, Any]) -> MetaObs:
    decoded = _from_jsonable(meta_obs_payload)
    if not isinstance(decoded, dict):
        raise ValueError("meta_obs must decode to a dict")
    return dict2meta(decoded, mtype="obs")


def _infer_via_standard_samples(policy, mobs: MetaObs, payload: Dict[str, Any]):
    """
    Build ILStudio standard-format samples (per `.cursor/rules/data-rule.mdc`)
    and feed them into the model for inference.

    This mirrors `benchmark.base.MetaPolicy.inference`, but ensures the sample dict includes:
    - image/state/raw_lang/timestamp
    - action/is_pad placeholders (None) for inference
    - reasoning/episode_id/__index__ metadata from request payload (best-effort)
    """
    # Normalize state first (same as MetaPolicy.inference)
    normed_mobs = policy.state_normalizer.normalize_metaobs(mobs, policy.ctrl_space)
    samples = policy.normed_mobs_to_samples(normed_mobs)

    # Enrich samples to the "standard dict format" required by ILStudio project rules.
    reasoning = _from_jsonable(payload.get("reasoning", None)) if payload.get("reasoning", None) is not None else None
    episode_id = payload.get("episode_id", -1)
    index = payload.get("__index__", -1)
    ts = payload.get("timestamp", None)

    for s in samples:
        # NOTE: Do NOT set action/is_pad to None; if key exists, data_collator will try to
        # convert it and fail. Leave them absent so `'action' in sample` returns False.
        if reasoning is not None:
            s.setdefault("reasoning", reasoning)
        s.setdefault("episode_id", episode_id)
        s.setdefault("__index__", index)
        # `normed_mobs_to_samples` fills timestamp from mobs.timestep; keep it if already present.
        if "timestamp" not in s and ts is not None:
            s["timestamp"] = ts

    # Convert samples to model input
    policy_obs = policy.meta2obs(samples)

    # Inference action chunk using underlying model
    action_chunk = policy.policy.select_action(policy_obs)

    # Convert to MetaAction and denormalize action (same as MetaPolicy.inference)
    macts = policy.act2meta(action_chunk, ctrl_space=policy.ctrl_space, ctrl_type=policy.ctrl_type)
    action_chunk = macts.action

    is_chunked = (len(action_chunk.shape) == 3)
    bs = action_chunk.shape[0] if is_chunked else 1
    ac_dim = action_chunk.shape[-1]

    if is_chunked:
        macts.action = action_chunk.reshape(-1, ac_dim)

    macts = policy.action_normalizer.denormalize_metaact(macts)

    if is_chunked:
        macts.action = macts.action.reshape(bs, -1, ac_dim).transpose(1, 0, 2)
    else:
        macts.action = macts.action[np.newaxis, :]

    from benchmark.base import MetaAction

    mact_list = [
        np.array(
            [asdict(MetaAction(action=aii, ctrl_type=macts.ctrl_type, ctrl_space=macts.ctrl_space)) for aii in ai],
            dtype=object,
        )
        for ai in macts.action
    ]

    if policy.chunk_size is not None and policy.chunk_size > 0:
        mact_list = mact_list[: policy.chunk_size]
    return mact_list


def create_app(policy) -> FastAPI:
    app = FastAPI(title="ILStudio Policy Server (FastAPI)")
    app.state.policy = policy

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/inference")
    async def inference(payload: Dict[str, Any] = Body(...)):
        """
        Expected payload shape:
        {
          "meta_obs": { ... serialized MetaObs ... },
          "episode_id": optional int,
          "__index__": optional int,
          "reasoning": optional any,
          "timestamp": optional float|int (if you want to override MetaObs.timestep)
        }
        """
        try:
            if "meta_obs" not in payload:
                raise ValueError("Missing field: meta_obs")

            mobs = _payload_to_metaobs(payload["meta_obs"])

            # Optional overrides / metadata (best-effort; model may ignore)
            if "timestamp" in payload and payload["timestamp"] is not None:
                mobs.timestep = _from_jsonable(payload["timestamp"])

            # Run inference
            with torch.no_grad():
                mact_list = _infer_via_standard_samples(app.state.policy, mobs, payload)

            # Return actions (JSON-encoded)
            return {"mact_list": _to_jsonable(mact_list)}
        except Exception as e:
            logger.exception("FastAPI inference failed")
            raise HTTPException(status_code=400, detail=str(e)) from e

    return app


class FastAPIPolicyServer(BaseServer):
    """
    HTTP/JSON policy server using FastAPI + uvicorn.

    Inherits from BaseServer for a unified interface.

    Args:
        policy: The MetaPolicy to serve.
        host: Bind address (e.g., "0.0.0.0").
        port: Port number.
        ssl_keyfile: Path to SSL private key file (for HTTPS).
        ssl_certfile: Path to SSL certificate file (for HTTPS).
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        ssl_keyfile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
    ):
        self.policy = policy
        self.host = host
        self.port = port
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile
        self._app: Optional[FastAPI] = None
        self._server = None

    def start(self) -> None:
        """Start the server (blocking)."""
        self._app = create_app(self.policy)
        scheme = "https" if self.ssl_certfile else "http"
        logger.info(f"🚀 FastAPI Policy Server started on {scheme}://{self.host}:{self.port}")
        uvicorn.run(
            self._app,
            host=self.host,
            port=self.port,
            ssl_keyfile=self.ssl_keyfile,
            ssl_certfile=self.ssl_certfile,
        )

    def stop(self) -> None:
        """Stop the server gracefully (no-op for uvicorn.run blocking mode)."""
        logger.info("✓ FastAPI Policy Server stopped")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Start ILStudio FastAPI policy server",
        epilog=(
            "For HTTPS, set environment variables:\n"
            "  export ILSTD_SSL_KEYFILE=/path/to/key.pem\n"
            "  export ILSTD_SSL_CERTFILE=/path/to/cert.pem"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("-p", "--port", type=int, default=8000)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        default="ckpt/act_sim_transfer_cube_scripted_zscore_example",
    )
    p.add_argument("--dataset_id", type=str, default="")
    p.add_argument("--chunk_size", type=int, default=-1)
    p.add_argument(
        "--https",
        action="store_true",
        help="Enable HTTPS (requires ILSTD_SSL_KEYFILE and ILSTD_SSL_CERTFILE env vars)",
    )
    return p.parse_args()


def _signal_handler(signum, frame):
    logger.info("⏸ Received interrupt signal, shutting down...")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    set_seed(0)
    args = _parse_args()
    args.is_training = False

    logger.info("=" * 60)
    logger.info("🚀 FastAPI Policy Server Startup")
    logger.info("=" * 60)
    logger.info(f"   Model path: {args.model_name_or_path}")
    logger.info(f"   Dataset ID: {args.dataset_id if args.dataset_id else '(first dataset)'}")
    logger.info(f"   Device: {args.device}")

    # Load normalizers
    normalizers, ctrl_space, ctrl_type = load_normalizers(args)
    args.ctrl_space, args.ctrl_type = ctrl_space, ctrl_type

    # Load policy from checkpoint
    from policy.direct_loader import load_model_from_checkpoint
    from benchmark.base import MetaPolicy

    model_components = load_model_from_checkpoint(args.model_name_or_path, args)
    model = model_components["model"]
    model.eval()

    policy = MetaPolicy(
        policy=model,
        chunk_size=args.chunk_size,
        action_normalizer=normalizers["action"],
        state_normalizer=normalizers["state"],
        ctrl_space=ctrl_space,
        ctrl_type=ctrl_type,
    )

    # Get SSL config from environment if HTTPS is requested
    ssl_keyfile = None
    ssl_certfile = None
    if getattr(args, "https", False):
        ssl_keyfile = os.environ.get("ILSTD_SSL_KEYFILE")
        ssl_certfile = os.environ.get("ILSTD_SSL_CERTFILE")
        if not ssl_keyfile or not ssl_certfile:
            logger.error(
                "HTTPS requires SSL certificates. Please set the following environment variables:\n"
                "\n"
                "    export ILSTD_SSL_KEYFILE=/path/to/your/key.pem\n"
                "    export ILSTD_SSL_CERTFILE=/path/to/your/cert.pem\n"
                "\n"
                "To generate a self-signed certificate for testing:\n"
                "\n"
                "    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj \"/CN=localhost\"\n"
                "    export ILSTD_SSL_KEYFILE=key.pem\n"
                "    export ILSTD_SSL_CERTFILE=cert.pem\n"
            )
            sys.exit(1)

    server = FastAPIPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )
    try:
        server.start()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
