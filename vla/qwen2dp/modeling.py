import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import numpy as np
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoModelForCausalLM, AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import requests
from typing import Any, Dict, List, Optional, Tuple, Union
from qwen_vl_utils import process_vision_info
from vla.qwen2vl_dp.policy import ConditionalUnet1D
from .data_utils import Qwen2VLAProcess, Qwen2VLADataCollatorForSupervisedDataset
# =============================================================================
# 步骤 1: 创建自定义的 Config 类
# =============================================================================
class QwenVLPolicyConfig(PretrainedConfig):
    """
    这是我们自定义模型的配置类。它继承自 PretrainedConfig，
    使其与 Hugging Face 生态系统兼容。
    参数:
        vlm_model_name_or_path (str, optional):
            基础 VLM 模型的路径或 Hugging Face Hub 名称。
            例如: "Qwen/Qwen2-VL-1.5B-Instruct"。
        policy_input_size (int, optional):
            Policy Head 的输入维度。这通常与 VLM 的 hidden_size 相同。
        policy_hidden_size (int, optional):
            Policy Head 中间隐藏层的维度。
        policy_output_size (int, optional):
            Policy Head 的最终输出维度。
        **kwargs:
            传递给父类 PretrainedConfig 的其他参数。
    """
    # 自定义一个模型类型名称，这对于 `AutoModel` 等自动类很重要
    model_type = "qwen_vl_with_policy_head"

    def __init__(
        self,
        vlm_model_name_or_path: str = "Qwen/Qwen2-VL-1.5B-Instruct",
        policy_action_dim: int = 7,
        policy_state_dim: int = 7,
        policy_cond_dim: int = 1536,
        policy_prediction_horizon: int = 16,
        policy_diffusion_step_embed_dim: int = 256,
        policy_down_dims: List[int] = [256, 512, 1024],
        policy_kernel_size: int = 5,
        policy_n_groups: int = 8,
        policy_noise_samples: int = 1,
        policy_num_inference_timesteps: int = 10,
        policy_num_train_timesteps: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vlm_model_name_or_path = vlm_model_name_or_path
        self.policy_action_dim = policy_action_dim
        self.policy_cond_dim = policy_cond_dim
        self.policy_diffusion_step_embed_dim = policy_diffusion_step_embed_dim
        self.policy_down_dims = policy_down_dims
        self.policy_kernel_size = policy_kernel_size
        self.policy_n_groups = policy_n_groups
        self.policy_state_dim = policy_state_dim
        self.policy_prediction_horizon = policy_prediction_horizon
        self.policy_noise_samples = policy_noise_samples
        self.policy_num_inference_timesteps = policy_num_inference_timesteps
        self.policy_num_train_timesteps = policy_num_train_timesteps



# =============================================================================
# 步骤 2: 创建自定义的 Model 类
# =============================================================================
class QwenVLForPolicy(PreTrainedModel):
    """
    一个包含 Qwen2-VL 模型和 Policy Head 的自定义模型。
    """
    # 将模型类与我们自定义的配置类关联起来
    config_class = QwenVLPolicyConfig

    def __init__(self, config: QwenVLPolicyConfig):
        super().__init__(config)
        self.config = config
        # 1. 加载 VLM 模型
        # 使用 AutoModelForCausalLM 加载，更通用和健壮
        print(f"Initializing VLM from base path: {config.vlm_model_name_or_path}")
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            config.vlm_model_name_or_path,
            torch_dtype=torch.bfloat16, # 自动选择合适的精度
            trust_remote_code=True # Qwen-VL 模型需要此参数
        )
        # 2. 定义 Policy Head
        policy_config_dict = {k[7:]:v for k,v in self.config.to_dict().items() if 'policy_' in k}
        self.policy_head = ConditionalUnet1D(**policy_config_dict)
    
    def set_requires_grad(self, training_args):
        if not training_args.lora_enable:
            # 设置视觉模型的冻结
            self.vlm.visual.requires_grad_(True) # set to true first
            self.config.freeze_vision_tower = training_args.freeze_vision_tower
            if training_args.freeze_vision_tower:
                for n,p in self.vlm.visual.named_parameters():
                    p.requires_grad = False
            if not training_args.freeze_backbone:# 尝试设置微调lm_head
                try:
                    self.vlm.lm_head.requires_grad_(True) 
                except Exception as e:
                    print(e)
        # 设置policy_head要梯度
        self.policy_head.requires_grad_(True)
    
    def forward(
        self,
        # QwenVL Input
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        # Policy Input
        actions: Optional[torch.LongTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        is_pad: bool = False,
        **kwargs,
    ):
        """
        前向传播逻辑。

        返回:
            一个字典，包含 policy_logits。
        """
        # 将输入传递给 VLM
        # 我们需要 VLM 的最后一层 hidden states
        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            output_hidden_states=True,  # 确保 VLM 返回 hidden_states
            # return_dict=True,
        )
        # 提取最后一层的 hidden states
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state.mean(dim=1).unsqueeze(1)
        policy_outputs = self.policy_head(actions, pooled_output, states, is_pad)
        # 计算action loss
        return {
            'llm_loss': outputs['loss'],
            'action_loss': policy_outputs['loss'],
            'loss': outputs['loss'] + policy_outputs['loss'],
        }

    
    @torch.no_grad()
    def generate(self,
        # QwenVL Input
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        # Policy Input
        actions: Optional[torch.LongTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        is_pad: bool = False,
        **kwargs,
    ):
        outputs = self.vlm.generate(
            input_ids=input_ids, 
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            num_beams=1,
            do_sample=False,
            temperature=0.2,
            max_new_tokens=256,
            eos_token_id=self.tokenizer.eos_token_id,  # End of sequence token
            pad_token_id=self.tokenizer.pad_token_id,  # Pad token
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        last_hidden_states = torch.cat([step[-1] for step in outputs.hidden_states], dim=1)
        pooled_output = last_hidden_states.mean(dim=1).unsqueeze(1)
        policy_outputs = self.policy_head(actions, pooled_output, states, is_pad)
        return policy_outputs    
        
    
    def select_action(self, obs):
        if not hasattr(self, 'data_processor'):
            self.data_processor = Qwen2VLAProcess(tokenizer=self.tokenizer, multimodal_processor=self.multimodal_processor)
        if not hasattr(self, 'data_collator'):
            self.data_collator = Qwen2VLADataCollatorForSupervisedDataset(multimodal_processor=self.multimodal_processor, tokenizer=self.tokenizer, computed_type=torch.bfloat16)
        # processor each sample in obs batch
        bs = obs['state'].shape[0]
        all_obs = [{k:torch.from_numpy(v[i]) if isinstance(v, np.ndarray) else v[i] for k,v in obs.items() if v is not None} for i in range(bs)]
        all_obs = [self.data_processor(sample) for sample in all_obs]
        batch_obs = self.data_collator(all_obs)
        batch_obs['states'] = batch_obs['states'].to(dtype=torch.bfloat16)
        for k,v in batch_obs.items():
            if isinstance(v, torch.Tensor):
                batch_obs[k] = v.to(self.device)
        action = self.generate(**batch_obs)
        return action

