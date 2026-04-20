"""
DreamZero Trainer for ILStudio.

Extends BaseTrainer with:
- Per logging_steps: raw ``train/dynamics_loss`` (video) and ``train/action_loss``
- Global step synchronization with action head
- DDP static graph for gradient checkpointing compatibility
"""

import torch
from loguru import logger
from policy.trainer import BaseTrainer


class Trainer(BaseTrainer):
    """Custom trainer for DreamZero world-model + policy joint training."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._static_graph_set = False

        # DreamZero (14B+ Wan2.1 backbone) cannot use nn.DataParallel — it
        # replicates all parameters (including frozen VAE / text-enc / image-enc)
        # to every GPU, causing OOM during backward gradient reduction on GPU 0.
        # Force n_gpu=1 so HF Trainer skips the DataParallel wrapper; use
        # torchrun / accelerate for true multi-GPU training (DDP/FSDP).
        if getattr(self.args, "n_gpu", 1) > 1:
            logger.warning(
                "DreamZero: forcing n_gpu=1 to avoid nn.DataParallel OOM. "
                "Use torchrun for multi-GPU training."
            )
            self.args._n_gpu = 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self._static_graph_set and isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model._set_static_graph()
            self._static_graph_set = True
            logger.info("Enabled DDP static graph for gradient checkpointing compatibility")

        outputs = model(inputs)

        if hasattr(outputs, "data"):
            out_dict = outputs.data
        elif isinstance(outputs, dict):
            out_dict = outputs
        else:
            out_dict = {"loss": outputs}

        loss = out_dict["loss"]
        # nn.DataParallel gathers per-replica scalars into a 1-D tensor
        # with n_gpu elements. Reduce to a scalar so backward() works and
        # .item() doesn't fail.
        if loss.dim() > 0:
            loss = loss.mean()

        log_extra = {}
        for key in ("dynamics_loss", "action_loss"):
            if key not in out_dict:
                continue
            val = out_dict[key]
            if isinstance(val, torch.Tensor):
                val = val.detach().mean().item() if val.dim() > 0 else val.detach().item()
            log_extra[f"train/{key}"] = val

        if log_extra and hasattr(self, "state") and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log(log_extra)

        if hasattr(model, "vla") and hasattr(model.vla, "action_head"):
            if hasattr(model.vla.action_head, "global_step"):
                model.vla.action_head.global_step = self.state.global_step

        if return_outputs:
            return loss, outputs
        return loss
