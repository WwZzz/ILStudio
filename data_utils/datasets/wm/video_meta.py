"""Video batch metadata for reasoning dict – kept separate to avoid circular imports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def build_video_meta(num_views, horizon, is_pad, input_steps, freq):
    """Construct the ``reasoning["video"]`` metadata dict.

    Parameters
    ----------
    num_views : int   – camera views per timestep (K)
    horizon   : int   – temporal frame count (T)
    is_pad    : list[bool] – length-T, True if frame is beyond episode end
    input_steps : int – number of leading conditioning frames (reserved)
    freq      : float – sampling frequency (Hz)
    """
    return {
        "num_views": int(num_views),
        "horizon": int(horizon),
        "is_pad": is_pad,
        "freq": float(freq),
    }


def convert_image_to_video(
    image: torch.Tensor | np.ndarray,
    video_meta: Mapping[str, Any],
    *,
    is_batch: bool = True,
) -> torch.Tensor:
    """Unflatten ILStudio flat video images using ``video_meta``.

    WM datasets stack frames as ``(T * num_views, C, H, W)``. This restores the
    time and view dimensions.

    Parameters
    ----------
    image
        If ``is_batch`` is False: ``(T * K, C, H, W)``. If True: ``(B, T * K, C, H, W)``.
    video_meta
        Must include ``"horizon"`` (T) and ``"num_views"`` (K).
    is_batch
        If True (default), ``image`` includes a leading batch dimension. If False,
        ``image`` is a single sample without that dimension.

    Returns
    -------
    torch.Tensor
        If ``is_batch`` is False: ``(T, K, C, H, W)``. If True: ``(B, T, K, C, H, W)``.
    """
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(np.asarray(image))
    t = int(video_meta["horizon"])
    k = int(video_meta["num_views"])
    flat = t * k
    if is_batch:
        if image.ndim != 5:
            raise ValueError(
                f"with is_batch=True, expected image (B, T*K, C, H, W); got {tuple(image.shape)}"
            )
        if int(image.shape[1]) != flat:
            raise ValueError(
                f"flat image length {image.shape[1]} != horizon * num_views ({flat})"
            )
        b = int(image.shape[0])
        return image.view(b, t, k, *image.shape[2:])
    if image.ndim != 4:
        raise ValueError(
            f"with is_batch=False, expected image (T*K, C, H, W); got {tuple(image.shape)}"
        )
    if int(image.shape[0]) != flat:
        raise ValueError(
            f"flat image length {image.shape[0]} != horizon * num_views ({flat})"
        )
    return image.view(t, k, *image.shape[1:])


def flat_image_to_video_btchw(
    image: torch.Tensor | np.ndarray,
    video_meta: Mapping[str, Any],
    *,
    is_batch: bool = True,
    view_index: int = 0,
) -> torch.Tensor:
    """Convert flat WM images to batched video ``(B, T, C, H, W)`` (one camera).

    Use this when a policy expects standard video layout with a batch dimension.
    For ``num_views == 1``, ``view_index`` is ignored.

    Parameters
    ----------
    image
        Same layout as :func:`convert_image_to_video` for the chosen ``is_batch``.
    video_meta
        Same as :func:`convert_image_to_video`.
    is_batch
        Same as :func:`convert_image_to_video` (whether ``image`` has a leading batch dim).
    view_index
        Which camera to keep when ``num_views > 1`` (0-based).

    Returns
    -------
    torch.Tensor
        ``(1, T, C, H, W)`` if ``is_batch`` is False, else ``(B, T, C, H, W)``.
    """
    tk = convert_image_to_video(image, video_meta, is_batch=is_batch)
    if tk.ndim == 5:
        k_dim = int(tk.shape[1])
        if not 0 <= view_index < k_dim:
            raise IndexError(f"view_index={view_index} out of range for num_views={k_dim}")
        out = tk[:, view_index]
        return out.unsqueeze(0)
    if tk.ndim == 6:
        k_dim = int(tk.shape[2])
        if not 0 <= view_index < k_dim:
            raise IndexError(f"view_index={view_index} out of range for num_views={k_dim}")
        return tk[:, :, view_index]
    raise RuntimeError(f"unexpected tensor rank after unwrap: {tk.ndim}")
