import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoModelForCausalLM, AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import requests
from typing import Any, Dict, List, Optional, Tuple, Union
from qwen_vl_utils import process_vision_info
from vla.qwen2vl_dp.policy import ConditionalUnet1D
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
            # 设置视觉模型的冻结
        self.vlm.visual.requires_grad_(True) # set to true first
        self.config.freeze_vision_tower = training_args.freeze_vision_tower
        if training_args.freeze_vision_tower:
            for n,p in self.vlm.visual.named_parameters():
                if not 'lora' in n.lower(): p.requires_grad = False
        else:
            for p in self.vlm.visual.parameters(): p.requires_grad = True
        
        # 这些都是设置哪些要梯度的
        self.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter
        if not training_args.freeze_backbone:# 尝试设置微调lm_head
            try:
                self.vlm.lm_head.requires_grad_(True) 
            except Exception as e:
                print(e)
        # 设置policy_head要梯度
        self.policy_head.requires_grad_(True)
        # if config['model_args'].using_film:
        #     model.input_action_proj.requires_grad_(True)
        #     model.reasoning_action_proj.requires_grad_(True)
        #     model.reasoning_film.requires_grad_(True)
    
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
        if self.training:
            return {
                'llm_loss': outputs['loss'],
                'action_loss': policy_outputs['loss'],
                'loss': outputs['loss'] + policy_outputs['loss'],
            }
        else:
            return {"policy_logits": policy_outputs}
    

# # =============================================================================
# # 步骤 3: 演示如何使用
# # =============================================================================
# if __name__ == '__main__':
#     # --- 设置目录 ---
#     CACHE_DIR = "/inspire/hdd/project/robot-action/public/models/"
#     SAVE_DIR = "./my_qwen_vl_policy_model"
#     os.makedirs(CACHE_DIR, exist_ok=True)
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     # --- 场景 1: 直接初始化模型 ---
#     print("="*50)
#     print("--- 场景 1: 直接初始化模型 ---")
#     print("="*50)

#     # 1. 动态获取 VLM 的 hidden_size 以确保配置正确
#     vlm_base_model = "/inspire/hdd/project/robot-action/public/models/Qwen2-VL-2B-Instruct"
#     print(f"Fetching base VLM config from: {vlm_base_model}")
#     # 使用 AutoConfig 加载配置
#     vlm_base_config = AutoConfig.from_pretrained(
#         vlm_base_model, 
#         cache_dir=CACHE_DIR, 
#         trust_remote_code=True
#     )

#     # 2. 初始化我们的自定义配置
#     my_config = QwenVLPolicyConfig(
#         vlm_model_name_or_path=vlm_base_model,
#     )
#     print("\nCustom config created:")
#     print(my_config)

#     # 3. 使用配置直接初始化我们的模型
#     # 这将会从 Hugging Face Hub 下载 Qwen/Qwen2-VL-1.5B-Instruct
#     print("\nInitializing model directly from config...")
#     model = QwenVLForPolicy(config=my_config).to(torch.bfloat16)
#     print("模型初始化完成。")

#     # --- 准备输入数据 ---
#     print("\nPreparing input data...")
#     processor = AutoProcessor.from_pretrained(vlm_base_model, trust_remote_code=True, cache_dir=CACHE_DIR)

#     # 构造 messages 输入
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": "file://./demo.jpeg",
#                 },
#                 {"type": "text", "text": "图中是什么？"},
#             ],
#         }
#     ]

#     # 1. 使用 apply_chat_template 准备文本
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )

#     # 2. 提取视觉输入
#     image_inputs, video_inputs = process_vision_info(messages)

#     # 3. 调用 processor 生成最终输入张量
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs['actions'] = torch.randn(1, 16, 7).bfloat16()
#     inputs['states'] = torch.zeros(1,7).bfloat16()
#     inputs['is_pad'] = torch.zeros(1, 16).bool()
#     inputs['labels'] = torch.randint(100, 6000, list(inputs['input_ids'].shape))
#     # 将模型和输入移动到合适的设备
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     print(f"Model and inputs moved to {device}.")

#     # --- 进行一次前向传播 ---
#     print("\nPerforming a forward pass with the initial model...")
#     model.eval() # 设置为评估模式
#     with torch.no_grad():
#         # 使用 **inputs 将字典解包为关键字参数传递给 forward 方法
#         outputs = model(**inputs)

#     print("\nPolicy head output (logits):")
#     print(outputs["policy_logits"])
#     print("Output shape:", outputs["policy_logits"].shape)


#     # --- 场景 2: 保存和加载整个模型 ---
#     print("\n\n" + "="*50)
#     print("--- 场景 2: 保存和加载整个模型 ---")
#     print("="*50)

#     # 1. 保存我们的自定义模型
#     # 这会保存 QwenVLPolicyConfig 到 config.json，
#     # 并保存 VLM 和 Policy Head 的权重到 model.safetensors
#     print(f"Saving the entire model to: {SAVE_DIR}")
#     model.save_pretrained(SAVE_DIR)
#     # 顺便保存 processor，方便后续加载
#     processor.save_pretrained(SAVE_DIR)
#     print("Save complete.")

#     # 2. 从保存的路径加载模型
#     # 假设我们现在在一个新的会话中，清空旧模型
#     del model
#     if device == 'cuda':
#         torch.cuda.empty_cache()
#     print(f"\nLoading the entire model from: {SAVE_DIR}")
    
#     # 使用 from_pretrained 方法，它会自动处理所有事情
#     # 注意：因为我们的类不在 transformers 库中，需要 `trust_remote_code=True`
#     loaded_model = QwenVLForPolicy.from_pretrained(SAVE_DIR, trust_remote_code=True)
    
#     print("\nModel loaded successfully from checkpoint.")

#     loaded_model.to(device)
#     loaded_model.eval()

#     # 3. 使用加载后的模型进行推理，验证其工作正常
#     print("\nPerforming a forward pass with the loaded model...")
#     with torch.no_grad():
#         loaded_outputs = loaded_model(**inputs)

#     print("\nLoaded model's policy head output (logits):")
#     print(loaded_outputs["policy_logits"])

#     # 4. 验证两次输出是否一致
#     are_close = torch.allclose(outputs["policy_logits"].cpu(), loaded_outputs["policy_logits"].cpu(), atol=1e-5)
#     print(f"\nVerification: Outputs are consistent? -> {are_close}")
#     assert are_close
#     print("✅ 验证成功：原始模型和从检查点加载的模型输出完全一致！")