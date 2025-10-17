import transformers
import torch 
from policy.trainer import BaseTrainer

class Trainer(BaseTrainer):
    def create_optimizer(self):
        param_groups = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.args.learning_rate*0.1,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            param_groups, 
            lr=self.args.learning_rate, 
            betas=[self.args.adam_beta1, self.args.adam_beta2], 
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay
        )
        return self.optimizer
    
    def training_step(self, *args, **kwargs):
        """扩展训练步骤，更新 EMA 参数"""
        loss = super().training_step(*args, **kwargs)
        if hasattr(self.model, "ema") and self.model.ema is not None:
            self.model.ema.step(self.model.parameters())  # 使用模型的 `ema` 对象更新权重
        return loss
    
    def evaluate(self, *args, **kwargs):
        """评估时使用 EMA 参数"""
        using_ema = hasattr(self.model, "ema") and self.model.ema is not None
        if using_ema:
            # 保存原始模型参数并切换到 EMA 参数
            self.model.ema.store(self.model.parameters())  # 备份当前模型参数
            self.model.ema.copy_to(self.model.parameters())  # 将 EMA 参数加载到模型中
        # 执行默认的评估逻辑
        results = super().evaluate(*args, **kwargs)
        if using_ema:
            # 恢复原始模型参数
            self.model.ema.restore(self.model.parameters())
        return results