#!/usr/bin/env python3
"""
Open a single rollout's Rerun recording from a results run folder.

Usage:
  python scripts/open_rerun.py 0315_asi_overhead+wrist_14w_ck30 1

It will search:
  results/<run_name>/**/rerun/*_roll0001.rrd

If a sibling .rbl exists (same stem), it will be loaded as well to restore the intended layout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _err(msg: str) -> None:
    print(f"[open_rerun] ERROR: {msg}", file=sys.stderr)


def _parse_rollout(s: str) -> int:
    s = s.strip()
    # allow "0003" or "3"
    try:
        v = int(s, 10)
    except ValueError as e:
        raise ValueError(f"Invalid rollout index: {s!r}") from e
    if v < 0:
        raise ValueError("Rollout index must be >= 0")
    return v


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        _err('Usage: python scripts/open_rerun.py <run_name> <rollout_index>')
        _err('Example: python scripts/open_rerun.py 0315_asi_overhead+wrist_14w_ck30 1')
        return 2

    run_name = argv[1].strip()
    rollout = _parse_rollout(argv[2])

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "results" / run_name
    if not run_dir.exists():
        _err(f"Run folder not found: {run_dir}")
        return 2

    pattern = f"**/rerun/*_roll{rollout:04d}.rrd"
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        _err(f"No .rrd found for rollout={rollout} with pattern: {pattern}")
        return 2

    if len(matches) > 1:
        print("[open_rerun] Multiple matches found; picking the first one:")
        for m in matches:
            print(f"  - {m}")

    rrd_path = matches[0].resolve()
    rbl_path = rrd_path.with_suffix(".rbl")

    bind = os.environ.get("ASI_RERUN_BIND", "0.0.0.0")
    port = os.environ.get("ASI_RERUN_WEB_PORT", "9090")
    mem = os.environ.get("ASI_RERUN_MEMORY_LIMIT", "95%")

    # Use the current interpreter so this works with the project's venv:
    #   source .venv/bin/activate && python scripts/open_rerun.py ...
    cmd: list[str] = [
        sys.executable,
        "-m",
        "rerun",
        "--web-viewer",
        "--bind",
        bind,
        "--web-viewer-port",
        str(port),
        "--memory-limit",
        mem,
        "--detach-process",
    ]

    # Load blueprint if present (keeps the dashboard layout consistent).
    if rbl_path.exists():
        cmd.append(str(rbl_path))
    cmd.append(str(rrd_path))

    print(f"[open_rerun] rrd: {rrd_path}")
    if rbl_path.exists():
        print(f"[open_rerun] rbl: {rbl_path}")
    print("[open_rerun] launching:")
    print("  " + " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        _err("Failed to run rerun CLI. Make sure you are using the venv python that has rerun-sdk installed.")
        _err("Try: source .venv/bin/activate")
        return 1
    except subprocess.CalledProcessError as e:
        _err(f"rerun exited with code {e.returncode}")
        return e.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

