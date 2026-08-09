"""Opt-in PyTorch backend configuration shared by runtime entrypoints."""

import os


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def configure_torch_backends_from_env():
    """Apply explicitly requested backend overrides to the current process."""

    disable_cudnn = os.environ.get("ILSTUDIO_DISABLE_CUDNN", "").strip().lower()
    if disable_cudnn not in _TRUE_VALUES:
        return False

    import torch

    torch.backends.cudnn.enabled = False
    return True
