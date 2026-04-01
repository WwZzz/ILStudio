"""
Rerun logger used by `eval_sim.py` when `--rerun` is enabled.

This repo's evaluation code expects a `RerunLogger` with:
- __init__(app_id, rrd_save_path, camera_names, enabled)
- new_rollout(rollout_idx)
- log_step(step, obs_images, gen_image_center, gen_image_anti_translated, ee_pose, target_pose,
           relative_traj, absolute_traj, chunk_size, chunk_step, terminate_reason, extra_info)
- finalize()

Key design choices (to match a stable dashboard-like layout):
- Entity paths are stable (no rollout index in the path).
- All samples are timestamped on a single monotonic timeline: `global_step`.
- Rollout/local-step are logged as text/scalars (not as timestamps), to avoid timeline collisions across rollouts.
- A default blueprint is embedded into the `.rrd` and also saved as a `.rbl` next to it.

python -m rerun --web-viewer --bind 0.0.0.0 --web-viewer-port 9090 /home/xudawei/Asi_ILS/results/0313_asi_overhead+wrist_6w_test_rerun1/rlbench/rerun/ReachTarget_env0.rrd
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    # torch.Tensor support (optional dependency)
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception:
        return None


def _chw_to_hwc(img: np.ndarray) -> np.ndarray:
    """Convert CHW images to HWC if needed."""
    if img.ndim != 3:
        return img
    if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        return np.transpose(img, (1, 2, 0))
    return img


def _normalize_u8(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8 where reasonable (best for most viewers)."""
    if img is None:
        return img
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        # Common ranges:
        # - [0, 1]
        # - [0, 255]
        # - [-1, 1] (very common for diffusion/flow image tensors)
        mn = float(np.nanmin(img)) if img.size else 0.0
        mx = float(np.nanmax(img)) if img.size else 0.0
        if mn < -0.01 and mx <= 1.5:
            img = (img + 1.0) * 127.5
        elif mx <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0.0, 255.0)
        return img.astype(np.uint8, copy=False)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8, copy=False)


def _as_points3d(xyz: Any) -> Optional[np.ndarray]:
    arr = _to_numpy(xyz)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1 and arr.size >= 3:
        return arr[:3][None, :]
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr[:, :3]
    return None


def _traj_to_points_world(traj: Any) -> Optional[np.ndarray]:
    """Best-effort (T,D)->(T,3) conversion."""
    arr = _to_numpy(traj)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return None
    return arr[:, :3]


def _relative_traj_to_world_points(rel_traj: Any, ee_pose: Any) -> Optional[np.ndarray]:
    """
    Convert relative trajectory to a 3D polyline in world coordinates (approx):
    - Treat first 3 dims as delta translations (meters) per step
    - Cumulative sum and offset by current ee xyz
    """
    rel = _to_numpy(rel_traj)
    ee = _to_numpy(ee_pose)
    if rel is None or ee is None:
        return None
    rel = np.asarray(rel, dtype=np.float32)
    if rel.ndim == 3:
        rel = rel[0]
    if rel.ndim != 2 or rel.shape[1] < 3:
        return None
    if ee.size < 3:
        return None
    deltas = rel[:, :3]
    pts = np.cumsum(deltas, axis=0)
    pts = pts + np.asarray(ee[:3], dtype=np.float32)[None, :]
    return pts


def _quat_normalize_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if n > 1e-8:
        q = q / n
    return q


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in xyzw order (q = q1 * q2)."""
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float32).reshape(4)
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float32).reshape(4)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float32)


def _quat_inv_xyzw(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion in xyzw order."""
    q = np.asarray(q, dtype=np.float32).reshape(4)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion (xyzw) to 3x3 rotation matrix."""
    x, y, z, w = _quat_normalize_xyzw(q).reshape(4)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )

@dataclass(frozen=True)
class _Paths:
    images: str = "images"
    world: str = "world"
    status: str = "status"


class RerunLogger:
    def __init__(
        self,
        app_id: str = "asi_inference",
        rrd_save_path: str = "/tmp/rerun.rrd",
        camera_names: Optional[Sequence[str]] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.app_id = str(app_id)
        self.rrd_save_path = str(rrd_save_path)
        self.camera_names = list(camera_names) if camera_names is not None else None

        self._paths = _Paths()
        self._rollout_idx: int = 0
        self._global_cursor: int = 0
        self._rollout_global_start: int = 0
        self._last_local_step: int = -1
        # Keep a short executed EE history to visualize the path.
        self._ee_path: list[np.ndarray] = []

        self.blueprint_path: Optional[str] = None
        self._disabled_reason: Optional[str] = None
        self._blueprint_save_error: Optional[str] = None
        # Coordinate/pose configuration
        self._view_coords_name = (os.getenv("ASI_RERUN_VIEW_COORDS", "RFU") or "RFU").strip().upper()
        # EE quaternion order can be 'xyzw' (default) or 'wxyz'
        self._ee_quat_order = (os.getenv("ASI_RERUN_EE_QUAT_ORDER", "xyzw") or "xyzw").strip().lower()
        # Decoded-trajectory quaternion convention (can differ from simulator EE pose).
        self._traj_quat_order = (os.getenv("ASI_RERUN_TRAJ_QUAT_ORDER", "xyzw") or "xyzw").strip().lower()

        if not self.enabled:
            return

        # Import rerun lazily; ensure we imported the SDK, not the unrelated `rerun` file-watcher.
        try:
            import rerun as rr  # type: ignore
        except Exception as e:
            self.enabled = False
            self._disabled_reason = f"[RerunLogger] import rerun failed: {type(e).__name__}: {e}"
            return

        required = ("init", "save", "log", "set_time_sequence", "send_blueprint")
        if not all(hasattr(rr, k) for k in required):
            self.enabled = False
            self._disabled_reason = (
                "[RerunLogger] Detected wrong `rerun` module (missing SDK API). Fix with:\n"
                "  python -m pip uninstall -y rerun\n"
                "  python -m pip install -U rerun-sdk"
            )
            return

        self.rr = rr

        os.makedirs(os.path.dirname(self.rrd_save_path) or ".", exist_ok=True)

        # Init without spawning a viewer; we only write an .rrd.
        try:
            self.rr.init(self.app_id, spawn=False)
        except TypeError:
            self.rr.init(self.app_id)

        try:
            self.rr.save(self.rrd_save_path)
        except Exception as e:
            self.enabled = False
            self._disabled_reason = f"[RerunLogger] rr.save failed: {type(e).__name__}: {e}"
            return

        # Coordinate system + axes (match the "golden" .rrd entity paths)
        try:
            vc = getattr(self.rr.ViewCoordinates, self._view_coords_name, None)
            if vc is None:
                vc = self.rr.ViewCoordinates.RFU
            self.rr.log("/", vc, static=True)
            self.rr.log(
                f"{self._paths.world}/axes/arrows",
                self.rr.Arrows3D(
                    origins=np.zeros((3, 3), dtype=np.float32),
                    vectors=np.eye(3, dtype=np.float32) * 0.3,
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                ),
                static=True,
            )
            self.rr.log(
                f"{self._paths.world}/axes/labels",
                self.rr.Points3D(
                    positions=(np.eye(3, dtype=np.float32) * 0.3),
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    labels=["X", "Y", "Z"],
                    show_labels=True,
                    radii=[0.02, 0.02, 0.02],
                ),
                static=True,
            )
        except Exception:
            pass

        # Embed a default blueprint (layout) + save a .rbl next to the .rrd (best effort).
        try:
            self._setup_blueprint()
        except Exception:
            pass

        # Helpful static metadata
        try:
            self.rr.log("app/id", self.rr.TextDocument(self.app_id), static=True)
            self.rr.log("app/rrd_path", self.rr.TextDocument(self.rrd_save_path), static=True)
            if self.blueprint_path:
                self.rr.log("app/blueprint_path", self.rr.TextDocument(self.blueprint_path), static=True)
        except Exception:
            pass

    def new_rollout(self, rollout_idx: int) -> None:
        if not self.enabled:
            return

        # Move the global cursor forward based on the last seen step of the previous rollout.
        if self._last_local_step >= 0:
            self._global_cursor += int(self._last_local_step) + 1
        self._last_local_step = -1

        self._rollout_idx = int(rollout_idx)
        self._rollout_global_start = int(self._global_cursor)
        self._ee_path = []

        try:
            self.rr.set_time_sequence("global_step", self._rollout_global_start)
        except Exception:
            pass

        # Optional: keep a small events stream (the golden .rrd has /status/events).
        try:
            self.rr.log(
                f"{self._paths.status}/events",
                self.rr.TextDocument(f"Rollout {self._rollout_idx} START"),
            )
        except Exception:
            pass

    def _log_images(self, obs_images: Any) -> None:
        if obs_images is None:
            return
        arr = _to_numpy(obs_images)
        if arr is None:
            return

        # obs_images can be:
        # - (K, C, H, W)
        # - (C, H, W)
        # - (H, W, C)
        if arr.ndim == 4:
            # (K,C,H,W)
            k = arr.shape[0]
            # Determine which index is front/wrist based on camera_names when available.
            idx_front = 0
            idx_wrist = 1 if k > 1 else 0
            if self.camera_names:
                for i, nm in enumerate(self.camera_names[:k]):
                    low = str(nm).lower()
                    if "front" in low or "over" in low:
                        idx_front = i
                    if "wrist" in low:
                        idx_wrist = i
            try:
                front = _normalize_u8(_chw_to_hwc(np.asarray(arr[idx_front])))
                self.rr.log(f"{self._paths.images}/obs_front", self.rr.Image(front))
            except Exception:
                pass
            try:
                wrist = _normalize_u8(_chw_to_hwc(np.asarray(arr[idx_wrist])))
                self.rr.log(f"{self._paths.images}/obs_wrist", self.rr.Image(wrist))
            except Exception:
                pass
        elif arr.ndim == 3:
            img = _normalize_u8(_chw_to_hwc(np.asarray(arr)))
            try:
                # Fallback: log as obs_front
                self.rr.log(f"{self._paths.images}/obs_front", self.rr.Image(img))
            except Exception:
                pass

    def _action_image_to_rgb_views_u8(self, action_image: Any) -> Optional[Dict[str, np.ndarray]]:
        """
        Match `action_process.vis_utils.save_rgb_views` semantics, but return HWC uint8 arrays.
        Returns dict with keys: 'xy', 'yz', 'zx'.
        """
        x = _to_numpy(action_image)
        if x is None:
            return None
        if x.ndim != 3:
            return None
        c = int(x.shape[0])
        if c == 9:
            view_slices = [(i * 3, i * 3 + 3) for i in range(3)]
        elif c == 12:
            view_slices = [(i * 4, i * 4 + 3) for i in range(3)]
        else:
            if c >= 9 and (c % 3 == 0):
                view_slices = [(i * 3, i * 3 + 3) for i in range(3)]
            elif c >= 12 and (c % 4 == 0):
                view_slices = [(i * 4, i * 4 + 3) for i in range(3)]
            else:
                return None

        out: Dict[str, np.ndarray] = {}
        keys = ["xy", "yz", "zx"]
        for key, (c0, c1) in zip(keys, view_slices):
            rgb = x[c0:c1].transpose(1, 2, 0).astype(np.float32, copy=False)
            vmin = float(np.nanmin(rgb))
            vmax = float(np.nanmax(rgb))
            if vmin < -1e-4 or vmax > 1.0001:
                if (vmin >= -1.05) and (vmax <= 1.05):
                    rgb_vis = (np.clip(rgb, -1.0, 1.0) + 1.0) * 0.5
                else:
                    denom = max(vmax - vmin, 1e-6)
                    rgb_vis = (rgb - vmin) / denom
            else:
                rgb_vis = np.clip(rgb, 0.0, 1.0)
            out[key] = (np.clip(rgb_vis * 255.0, 0.0, 255.0)).astype(np.uint8)
        return out

    def _log_generated_images(self, gen_center: Any, gen_inv: Any) -> None:
        # gen_center: centered action image
        if gen_center is not None:
            img = _to_numpy(gen_center)
            if img is not None and img.ndim == 3:
                views = self._action_image_to_rgb_views_u8(img)
                if views:
                    for k, v in views.items():
                        try:
                            self.rr.log(f"{self._paths.images}/gen_center/{k}", self.rr.Image(v))
                        except Exception:
                            pass

        # gen_inv: inverse-centered (anti-translated) action image
        if gen_inv is not None:
            img = _to_numpy(gen_inv)
            if img is not None and img.ndim == 3:
                views = self._action_image_to_rgb_views_u8(img)
                if views:
                    for k, v in views.items():
                        try:
                            self.rr.log(f"{self._paths.images}/gen_anti_translated/{k}", self.rr.Image(v))
                        except Exception:
                            pass

    def _log_poses_and_trajs(
        self,
        ee_pose: Any,
        target_pose: Any,
        relative_traj: Any,
        absolute_traj: Any,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        ee = _to_numpy(ee_pose)
        tgt = _to_numpy(target_pose)

        try:
            if ee is not None and ee.size >= 3:
                ee_xyz = ee[:3].astype(np.float32)
                # Executed EE path (blue line) + current EE (blue sphere).
                self._ee_path.append(ee_xyz.copy())
                if len(self._ee_path) >= 2:
                    self.rr.log(
                        f"{self._paths.world}/ee_path",
                        self.rr.LineStrips3D([np.stack(self._ee_path, axis=0)], colors=[[0, 128, 255]], radii=[0.003]),
                    )
                self.rr.log(
                    f"{self._paths.world}/ee_pos",
                    self.rr.Points3D([ee_xyz], colors=[[0, 128, 255]], radii=[0.03]),
                )
        except Exception:
            pass

        try:
            pts = _as_points3d(tgt)
            if pts is not None:
                self.rr.log(
                    f"{self._paths.world}/target",
                    self.rr.Points3D(
                        pts,
                        colors=[[255, 0, 0]] * int(pts.shape[0]),
                        radii=[0.03] * int(pts.shape[0]),
                    ),
                )
        except Exception:
            pass

        abs_pts = None
        try:
            abs_pts = _traj_to_points_world(absolute_traj)
            if abs_pts is not None and abs_pts.shape[0] >= 2:
                self.rr.log(
                    f"{self._paths.world}/absolute_traj",
                    self.rr.LineStrips3D([abs_pts], colors=[[255, 0, 0]], radii=[0.003]),
                )
                self.rr.log(
                    f"{self._paths.world}/absolute_traj_pts",
                    self.rr.Points3D(
                        abs_pts,
                        colors=[[255, 0, 0]] * int(abs_pts.shape[0]),
                        radii=[0.006] * int(abs_pts.shape[0]),
                    ),
                )
        except Exception:
            abs_pts = None

        rel_pts = None
        try:
            # Relative trajectory is in EE frame; visualize it under a Transform3D at /world/ee_frame.
            # (This matches the "golden" .rrd structure.)
            rel_pts = _traj_to_points_world(relative_traj)
            if rel_pts is not None and rel_pts.shape[0] >= 2 and ee is not None and ee.size >= 7:
                # ee pose: xyz + quat(x,y,z,w)
                ee_xyz = ee[:3].astype(np.float32)
                quat = ee[3:7].astype(np.float32).reshape(4)
                # Some pipelines store wxyz; allow override.
                if self._ee_quat_order == "wxyz":
                    quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
                quat = _quat_normalize_xyzw(quat)
                try:
                    self.rr.log(
                        f"{self._paths.world}/ee_frame",
                        self.rr.Transform3D(translation=ee_xyz, quaternion=quat, axis_length=0.12),
                    )
                except Exception:
                    # Fallback: translation only
                    self.rr.log(
                        f"{self._paths.world}/ee_frame",
                        self.rr.Transform3D(translation=ee_xyz, axis_length=0.12),
                    )

                self.rr.log(
                    f"{self._paths.world}/ee_frame/relative_traj",
                    self.rr.LineStrips3D([rel_pts], colors=[[255, 255, 0]], radii=[0.003]),
                )
                self.rr.log(
                    f"{self._paths.world}/ee_frame/relative_traj_pts",
                    self.rr.Points3D(
                        rel_pts,
                        colors=[[255, 255, 0]] * int(rel_pts.shape[0]),
                        radii=[0.006] * int(rel_pts.shape[0]),
                    ),
                )
        except Exception:
            rel_pts = None

        return abs_pts, rel_pts

    def _log_scalars(self, prefix: str, data: Dict[str, Any]) -> None:
        for k, v in (data or {}).items():
            if v is None:
                continue
            try:
                if isinstance(v, (bool, int, float, np.integer, np.floating)):
                    self.rr.log(f"{prefix}/{k}", self.rr.Scalars(float(v)))
                else:
                    arr = _to_numpy(v)
                    if arr is not None and arr.ndim == 1 and arr.size <= 64:
                        for i, vi in enumerate(arr.tolist()):
                            try:
                                self.rr.log(f"{prefix}/{k}_{i}", self.rr.Scalars(float(vi)))
                            except Exception:
                                pass
            except Exception:
                pass

    def log_step(
        self,
        step: int,
        obs_images: Any = None,
        gen_image_center: Any = None,
        gen_image_anti_translated: Any = None,
        ee_pose: Any = None,
        target_pose: Any = None,
        relative_traj: Any = None,
        absolute_traj: Any = None,
        chunk_size: Any = None,
        chunk_step: Any = None,
        terminate_reason: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        step_i = int(step)
        self._last_local_step = max(self._last_local_step, step_i)
        global_step = int(self._rollout_global_start + step_i)

        try:
            self.rr.set_time_sequence("global_step", global_step)
        except Exception:
            pass

        self._log_images(obs_images)
        self._log_generated_images(gen_image_center, gen_image_anti_translated)
        abs_pts, rel_pts = self._log_poses_and_trajs(ee_pose, target_pose, relative_traj, absolute_traj)

        # ---- Orientation consistency diagnostics (sim EE vs decoded abs traj at chunk_step) ----
        rot_delta_deg = None
        dot_x = dot_y = dot_z = None
        try:
            ee_arr = _to_numpy(ee_pose)
            traj = _to_numpy(absolute_traj)
            if ee_arr is not None and traj is not None:
                ee_arr = np.asarray(ee_arr, dtype=np.float32).reshape(-1)
                traj = np.asarray(traj, dtype=np.float32)
                if traj.ndim == 3:
                    traj = traj[0]
                if ee_arr.size >= 7 and traj.ndim == 2 and traj.shape[0] >= 1 and traj.shape[1] >= 7:
                    idx = int(chunk_step) if chunk_step is not None else 0
                    ii = max(0, min(idx, int(traj.shape[0]) - 1))

                    # sim ee quaternion
                    q_ee = ee_arr[3:7].astype(np.float32)
                    if self._ee_quat_order == "wxyz":
                        q_ee = np.array([q_ee[1], q_ee[2], q_ee[3], q_ee[0]], dtype=np.float32)
                    q_ee = _quat_normalize_xyzw(q_ee)

                    # decoded abs quaternion
                    q_tr = traj[ii, 3:7].astype(np.float32)
                    if self._traj_quat_order == "wxyz":
                        q_tr = np.array([q_tr[1], q_tr[2], q_tr[3], q_tr[0]], dtype=np.float32)
                    q_tr = _quat_normalize_xyzw(q_tr)

                    # relative rotation q_delta = inv(q_ee) * q_tr
                    q_delta = _quat_mul_xyzw(_quat_inv_xyzw(q_ee), q_tr)
                    q_delta = _quat_normalize_xyzw(q_delta)
                    # angle = 2*acos(|w|)
                    w = float(np.clip(abs(q_delta[3]), 0.0, 1.0))
                    rot_delta_deg = float(2.0 * np.arccos(w) * 180.0 / np.pi)

                    R_ee = _quat_to_rotmat_xyzw(q_ee)
                    R_tr = _quat_to_rotmat_xyzw(q_tr)
                    # compare basis vectors (columns are local axes in world)
                    dot_x = float(np.dot(R_ee[:, 0], R_tr[:, 0]))
                    dot_y = float(np.dot(R_ee[:, 1], R_tr[:, 1]))
                    dot_z = float(np.dot(R_ee[:, 2], R_tr[:, 2]))
        except Exception:
            rot_delta_deg = None

        # Abs-traj visualization image (match golden .rrd: /images/traj_vis)
        try:
            from action_process.vis_utils import render_inference_traj_grid  # lazy import

            img = render_inference_traj_grid(
                relative_traj=relative_traj,
                absolute_traj=absolute_traj,
                ee_pose=ee_pose,
                target_pos=target_pose,
                rollout_idx=self._rollout_idx,
                step=step_i,
            )
            if isinstance(img, np.ndarray) and img.ndim == 3:
                self.rr.log(f"{self._paths.images}/traj_vis", self.rr.Image(img))
        except Exception:
            pass

        # Sim-world pose view next to Abs Traj: X→right, Z→up, Y→into screen.
        try:
            every = int(os.getenv("ASI_RERUN_WORLD_VIS_EVERY", "1") or "1")
            if every <= 1 or (step_i % every) == 0:
                from action_process.vis_utils import render_sim_world_pose_view  # lazy import

                sim_traj = np.asarray(self._ee_path, dtype=np.float32) if len(self._ee_path) > 0 else None
                img2 = render_sim_world_pose_view(
                    sim_traj=sim_traj,
                    ee_pose=ee_pose,
                    target_pos=target_pose,
                    rollout_idx=self._rollout_idx,
                    step=step_i,
                )
                if isinstance(img2, np.ndarray) and img2.ndim == 3:
                    self.rr.log(f"{self._paths.images}/world_vis", self.rr.Image(img2))
        except Exception:
            pass

        # Human-readable status panel (match /status/chunk_info)
        try:
            lines = [f"Rollout {self._rollout_idx} | Local step {step_i} | Global step {global_step}"]
            if chunk_size is not None or chunk_step is not None:
                lines.append(f"chunk_size={chunk_size}  chunk_step={chunk_step}")
            if terminate_reason:
                lines.append(f"terminate_reason={terminate_reason}")
            if rot_delta_deg is not None:
                lines.append(f"rot_delta(sim_ee vs decoded_abs)={rot_delta_deg:.3f} deg")
                if dot_x is not None:
                    lines.append(f"axes_dot x={dot_x:.4f} y={dot_y:.4f} z={dot_z:.4f}")
            if extra_info:
                for kk in (
                    "did_infer_new_chunk",
                    "last_infer_t",
                    "queue_len_before",
                    "queue_len_after",
                    "dist_ee_to_target",
                    "dist_pred_end_to_target",
                ):
                    if kk in extra_info and extra_info[kk] is not None:
                        lines.append(f"{kk}={extra_info[kk]}")
                # Allow upstream to append extra human-readable lines into the status panel.
                _more = extra_info.get("chunk_status_lines", None)
                if _more:
                    if isinstance(_more, str):
                        lines += [ln for ln in _more.splitlines() if ln.strip()]
                    elif isinstance(_more, (list, tuple)):
                        lines += [str(ln) for ln in _more if str(ln).strip()]
            self.rr.log(f"{self._paths.status}/chunk_info", self.rr.TextDocument("\n".join(lines)))
        except Exception:
            pass

        # Optional: log a 2D trajectory visualization image if provided by upstream (not currently wired).

    def finalize(self) -> None:
        if not self.enabled:
            return
        try:
            self.rr.flush()
        except Exception:
            pass
        try:
            self.rr.rerun_shutdown()
        except Exception:
            pass

    def _setup_blueprint(self) -> None:
        """Embed + save a default blueprint matching the desired dashboard layout."""
        import rerun.blueprint as rrb  # type: ignore

        obs0 = rrb.Spatial2DView(name="Obs (o)", contents="/images/obs_front")
        obs1 = rrb.Spatial2DView(name="Obs (w)", contents="/images/obs_wrist")
        abs_traj = rrb.Spatial2DView(name="Abs Traj", contents="/images/traj_vis")
        world_vis = rrb.Spatial2DView(name="World (sim)", contents="/images/world_vis")
        # 3D World: executed path (blue) + ee current + decoded relative traj (yellow) + target + axes.
        world = rrb.Spatial3DView(
            name="3D World",
            contents=[
                "/world/ee_path",
                "/world/ee_pos",
                "/world/ee_frame/**",
                "/world/target",
                "/world/axes/**",
                "/world/absolute_traj",
                "/world/absolute_traj_pts",
            ],
        )

        # 3-view RGB slices (xy/yz/zx), matching vis_utils.save_rgb_views semantics.
        gen_c_xy = rrb.Spatial2DView(name="Gen(c) xy", contents="/images/gen_center/xy")
        gen_c_yz = rrb.Spatial2DView(name="Gen(c) yz", contents="/images/gen_center/yz")
        gen_c_zx = rrb.Spatial2DView(name="Gen(c) zx", contents="/images/gen_center/zx")
        gen_i_xy = rrb.Spatial2DView(name="Gen(in) xy", contents="/images/gen_anti_translated/xy")
        gen_i_yz = rrb.Spatial2DView(name="Gen(in) yz", contents="/images/gen_anti_translated/yz")
        gen_i_zx = rrb.Spatial2DView(name="Gen(in) zx", contents="/images/gen_anti_translated/zx")

        chunk_status = rrb.TextDocumentView(name="Chunk Status", contents="/status/chunk_info")

        left_top = rrb.Horizontal(obs0, obs1, abs_traj, world_vis, column_shares=[1.0, 1.0, 1.2, 1.2], name="Top")
        gen_grid = rrb.Grid(
            gen_c_xy,
            gen_c_yz,
            gen_c_zx,
            gen_i_xy,
            gen_i_yz,
            gen_i_zx,
            grid_columns=3,
            name="GenGrid",
        )
        left = rrb.Vertical(left_top, gen_grid, chunk_status, row_shares=[1.1, 1.4, 1.0], name="Left")

        root = rrb.Horizontal(left, world, column_shares=[2.2, 3.0], name="Main")
        bp = rrb.Blueprint(root, auto_layout=False, auto_views=False)

        self.rr.send_blueprint(bp, make_active=True, make_default=True)

        base, _ext = os.path.splitext(self.rrd_save_path)
        self.blueprint_path = base + ".rbl"
        try:
            bp.save(self.app_id, self.blueprint_path)
        except Exception as e:
            self._blueprint_save_error = f"{type(e).__name__}: {e}"
            self.blueprint_path = None

