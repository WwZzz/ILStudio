"""Custom trainer for FastWAM within ILStudio.

FastWAM uses its own ``training_loss`` that combines video and action flow-matching
losses, so we override ``compute_loss`` to delegate to the model's ``forward``
(which calls ``training_loss``).  Checkpoint saving uses FastWAM's own format
(MoT state-dict + optional proprio_encoder), not HF ``save_pretrained``.
"""

import os
from pathlib import Path

import torch
from policy.trainer import BaseTrainer
from loguru import logger


class Trainer(BaseTrainer):
    """Trainer for FastWAM models.

    Overrides loss computation to use the FastWAM dual-loss (video + action)
    instead of the default HF Trainer cross-entropy.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, loss_dict = model(inputs)
        if self.state.global_step % max(int(self.args.logging_steps), 1) == 0:
            logger.info(
                f"step={self.state.global_step}  "
                + "  ".join(
                    f"{k}={(v.detach().mean().item() if isinstance(v, torch.Tensor) else float(v)):.5f}"
                    for k, v in loss_dict.items()
                )
            )
        if return_outputs:
            return loss, loss_dict
        return loss

    def create_optimizer(self):
        """Only optimize MoT (video+action DiT) and optional proprio_encoder.

        VAE and text encoder are frozen by design.
        """
        inner = self.model
        while hasattr(inner, "module"):
            inner = inner.module
        fastwam = inner.model if hasattr(inner, "model") else inner

        params = []
        if hasattr(fastwam, "dit"):
            params.extend(
                p for p in fastwam.dit.parameters() if p.requires_grad
            )
        if hasattr(fastwam, "proprio_encoder") and fastwam.proprio_encoder is not None:
            params.extend(
                p for p in fastwam.proprio_encoder.parameters() if p.requires_grad
            )

        if not params:
            logger.warning("No trainable parameters found -- falling back to default optimizer.")
            return super().create_optimizer()

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    def save_model(self, output_dir=None, _internal_call=False):
        """Save FastWAM checkpoint in its native format (MoT + proprio_encoder).

        HF Trainer calls this at every ``save_steps`` and at the end of training.
        Weights use FastWAM's native ``.pt`` format; :class:`~policy.fastwam.modeling.FastWAMPolicyConfig`
        is saved as ``config.json`` for ILStudio / ``direct_loader`` reload (``action_dim``, etc.).
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        inner = self.model
        while hasattr(inner, "module"):
            inner = inner.module

        if hasattr(inner, "config") and inner.config is not None:
            inner.config.save_pretrained(output_dir)

        ckpt_path = os.path.join(output_dir, "fastwam_checkpoint.pt")
        if hasattr(inner, "save_checkpoint"):
            inner.save_checkpoint(
                ckpt_path,
                optimizer=self.optimizer,
                step=self.state.global_step,
            )
        else:
            torch.save(inner.state_dict(), ckpt_path)
        logger.info(f"FastWAM checkpoint saved to {ckpt_path}")
