"""
RynnVLA-002 Trainer for ILStudio.

Extends BaseTrainer with:
- Separate logging of CE loss and continuous action L1 loss
- Gradient clipping friendly with large Chameleon backbone
"""

import torch
from loguru import logger
from policy.trainer import BaseTrainer


def _tensor_to_log_scalar(x: torch.Tensor) -> float:
    """Scalar for logging; ``nn.DataParallel`` may stack per-GPU values to shape ``(K,)``."""
    t = x.detach().float()
    if t.numel() == 0:
        return 0.0
    if t.numel() == 1:
        return float(t.item())
    # 某一卡 CE 为 nan 时 .mean() 会污染整次日志；用 nanmean 更稳
    return float(torch.nanmean(t).item())


def _finite_float(x: float) -> bool:
    return x == x and abs(x) != float("inf")


class Trainer(BaseTrainer):
    """Custom trainer for RynnVLA-002 combined CE + action-head training."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # HF Trainer 在 n_gpu>1 时用 nn.DataParallel：与本 policy 的 list batch、lazy VQ、
        # CE 汇总不兼容，易出现 loss/grad_norm=0、ce_loss=nan、每步重载 VQ。
        if getattr(self.args, "n_gpu", 1) > 1:
            logger.warning(
                "RynnVLA-002: 已强制 n_gpu=1（禁用 DataParallel）。多卡请用 torchrun + DDP / accelerate。"
            )
            self.args._n_gpu = 1
        self._loss_ema = {}
        self._ema_decay = 0.99

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)

        if isinstance(outputs, dict):
            out_dict = outputs
        elif hasattr(outputs, "data"):
            out_dict = outputs.data
        else:
            out_dict = {"loss": outputs}

        loss = out_dict["loss"]
        # DataParallel 可能把各卡标量堆成 shape (n_gpu,)；需压成标量才能正确 backward / 打日志
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()

        for key in ["ce_loss", "ct_loss"]:
            if key in out_dict:
                val = out_dict[key]
                if isinstance(val, torch.Tensor):
                    val = _tensor_to_log_scalar(val)
                if not _finite_float(val):
                    logger.warning(
                        "跳过 {} 的 EMA 更新：本步值为非有限数（常见于 CE 在全 -100 mask 下为 nan，"
                        "或 DataParallel 某卡数值异常）。总 loss 仍照常反传。",
                        key,
                    )
                    continue
                if key not in self._loss_ema:
                    self._loss_ema[key] = val
                else:
                    self._loss_ema[key] = (
                        self._ema_decay * self._loss_ema[key]
                        + (1 - self._ema_decay) * val
                    )

        if hasattr(self, "state") and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            for key, val in self._loss_ema.items():
                self.log({f"train/{key}_ema": val})

        if return_outputs:
            return loss, outputs
        return loss
