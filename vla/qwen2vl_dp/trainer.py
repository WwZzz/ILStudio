import transformers.trainer 
import os
from peft import PeftModel, PeftConfig
import warnings
# 忽略重复的 UserWarning，只让特定警告显示一次
warnings.simplefilter("once", UserWarning)

class Trainer(transformers.trainer.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"]
        logging_steps = self.args.logging_steps
        if (self.state.global_step % logging_steps == 0) and (self.state.global_step != 0):
            log_dict = {}
            if "action_loss" in outputs:
                log_dict["action_loss"] = outputs["action_loss"].detach().cpu().item()
            if "llm_loss" in outputs:
                log_dict["llm_loss"] = outputs["llm_loss"].detach().cpu().item()
            if log_dict:
                self.log(log_dict)
        return (loss, outputs) if return_outputs else loss

    # def save_model(self, output_dir=None, _internal_call=False):
    #     """
    #     修改的保存模型逻辑：
    #     - 如果模型是 PeftModel，LoRA 参数保存到子目录`peft_adapter`。
    #     - 主模型权重保存到主目录。
    #     """
    #     output_dir = output_dir if output_dir is not None else self.args.output_dir
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     if isinstance(self.model, PeftModel):
    #         # 保存 Peft 的权重到子目录 peft_adapter
    #         peft_adapter_dir = os.path.join(output_dir, "peft_adapter")
    #         os.makedirs(peft_adapter_dir, exist_ok=True)
    #         self.model.save_pretrained(peft_adapter_dir)  # 保存 LoRA 参数到子目录
    #         self.model.base_model.save_pretrained(output_dir)
    #         # 调用父类逻辑保存主模型权重到 output_dir
    #         super().save_model(output_dir, _internal_call=_internal_call)#这个只存deepspeed格式的权重
    #     else:
    #         # 如果不是 PeftModel，按默认逻辑保存整个模型
    #         self.model.save_pretrained(output_dir)
    #         super().save_model(output_dir, _internal_call=_internal_call)

    #     # 保存 tokenizer
    #     if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None: 
    #         try:
    #             self.model.tokenizer.save_pretrained(output_dir)
    #         except Exception as e:
    #             warnings.warn(f"Failed to save tokenizer due to {e}")
        
    #     # 保存多模态处理器
    #     if hasattr(self.model, 'multimodal_processor') and self.model.multimodal_processor is not None:
    #         try:
    #             self.model.multimodal_processor.save_pretrained(output_dir)
    #         except Exception as e:
    #             warnings.warn(f"Failed to save processor due to {e}")
                
    # def _load_from_checkpoint(self, resume_from_checkpoint):
    #     """
    #     修改的检查点加载逻辑：
    #     - 如果存在 LoRA 的配置文件，将加载 LoRA 模型及基础权重。
    #     - LoRA 参数从 `peft_adapter` 子目录中加载。
    #     """
    #     if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
    #         # 子目录路径，用于存储 Peft 配置和权重
    #         peft_adapter_dir = os.path.join(resume_from_checkpoint, "peft_adapter")

    #         # 检查是否存在 LoRA 的配置文件
    #         if os.path.exists(os.path.join(peft_adapter_dir, "adapter_config.json")):
    #             # 加载适配器配置
    #             peft_config = PeftConfig.from_pretrained(peft_adapter_dir)
    #             # 加载基础模型权重
    #             base_model = self.model._get_base_model().from_pretrained(peft_config.base_model_name_or_path)
    #             # 加载 Peft 模型
    #             self.model = PeftModel.from_pretrained(base_model, peft_adapter_dir)
    #         else:
    #             # 没有 LoRA 配置，则加载整个主模型权重
    #             self.model = self.model.from_pretrained(resume_from_checkpoint)

    #     # 调用父类逻辑恢复优化器和调度器状态
    #     super()._load_from_checkpoint(resume_from_checkpoint)
    
    # def save_model(self, output_dir=None, _internal_call=False):
    #     """
    #     自定义保存模型逻辑：
    #     - 如果模型是 PeftModel，先保存 LoRA 参数，再调用父类接口保存主模型权重。
    #     - 如果模型不是 PeftModel，则按默认逻辑保存整个模型。
    #     """
    #     output_dir = output_dir if output_dir is not None else self.args.output_dir
    #     os.makedirs(output_dir, exist_ok=True)

    #     if isinstance(self.model, PeftModel):
    #         # 保存 LoRA 参数
    #         self.model.save_pretrained(output_dir)

    #         # 调用父类逻辑保存主模型权重
    #         super().save_model(output_dir, _internal_call=_internal_call)
    #     else:
    #         # 如果不是 PeftModel，按默认逻辑保存整个模型
    #         super().save_model(output_dir, _internal_call=_internal_call)
    #     if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None: 
    #         try:
    #             self.model.tokenizer.save_pretrained(output_dir)
    #         except Exception as e:
    #             warnings.warn(f"Failed to save tokenizer due to {e}")
    #     if hasattr(self.model, 'multimodal_processor') and self.model.multimodal_processor is not None:
    #         try:
    #             self.model.multimodal_processor.save_pretrained(output_dir)
    #         except Exception as e:
    #             warnings.warn(f"Failed to save processor due to {e}")
        
    # def _load_from_checkpoint(self, resume_from_checkpoint):
        # """
        # 自定义检查点加载逻辑：
        # - 检查是否有 LoRA 配置文件来决定是否加载 LoRA 模型。
        # """
        # if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
        #     # 检查是否存在 LoRA 的配置文件
        #     if os.path.exists(os.path.join(resume_from_checkpoint, "adapter_config.json")):
        #         # 加载 LoRA 配置文件和基础模型权重
        #         peft_config = PeftConfig.from_pretrained(resume_from_checkpoint)
        #         base_model = self.model._get_base_model().from_pretrained(peft_config.base_model_name_or_path)
        #         self.model = PeftModel.from_pretrained(base_model, resume_from_checkpoint)
        #     else:
        #         self.model = self.model.from_pretrained(resume_from_checkpoint)

        # # 调用父类逻辑恢复优化器和调度器状态
        # super()._load_from_checkpoint(resume_from_checkpoint)