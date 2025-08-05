import os
import torch.nn as nn
import torch
import numpy as np
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from .policy import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from collections import OrderedDict
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput

class DiffusionPolicyConfig(PretrainedConfig):
    """
    Configuration class for the DiffusionPolicy model.
    This class manages all hyperparameters related to the model's architecture.
    """
    def __init__(
        self,
        action_dim=7,               # Action dimensionality
        state_dim=7,                # State dimensionality
        camera_names=['primary'],             # List of names for input cameras
        observation_horizon=1,      # Number of timesteps in observation
        prediction_horizon=16,       # Number of timesteps in prediction horizon
        num_inference_timesteps=10,  # Timesteps for inference in noise scheduler
        ema_power=0.75,                # Power for EMA updates
        feature_dimension=64,     # Feature dimensionality for camera embeddings
        num_kp=32,                # Number of keypoints extracted using SpatialSoftmax
        **kwargs
    ):
        super().__init__(**kwargs)
        self.camera_names = camera_names
        self.observation_horizon = observation_horizon
        self.prediction_horizon = prediction_horizon
        self.num_inference_timesteps = num_inference_timesteps
        self.ema_power = ema_power
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.feature_dimension = feature_dimension
        self.num_kp = num_kp

# 转化后的模型
class DiffusionPolicyModel(PreTrainedModel):
    # 默认配置类
    config_class = DiffusionPolicyConfig

    def __init__(self, config):
        super().__init__(config)
        self.camera_names = config.camera_names
        self.observation_horizon = config.observation_horizon
        self.prediction_horizon = config.prediction_horizon
        self.num_kp = config.num_kp
        self.feature_dimension = config.feature_dimension
        self.ac_dim = config.action_dim
        self.obs_dim = self.feature_dimension * len(self.camera_names) + config.state_dim

        # 构建模型结构
        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            backbones.append(ResNet18Conv(input_channel=3, pretrained=False, input_coord_conv=False))
            pools.append(SpatialSoftmax(input_shape=[512, 8, 8], num_kp=self.num_kp, temperature=1.0,
                                        learnable_temperature=False, noise_std=0.0))
            linears.append(nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
        
        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)
        backbones = replace_bn_with_gn(backbones)

        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim * self.observation_horizon,
        )

        self.nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'noise_pred_net': noise_pred_net,
            })
        }).float().cuda()

        # 使用 EMA
        self.ema = EMAModel(model=self.nets, power=config.ema_power)

        # 初始化噪声调度器
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon',
        )

        self.init_weights()

    def init_weights(self):
        """初始化所有模块的权重"""
        for module in self.nets.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        在训练时，输入 `qpos`, `image`, 和 `actions`，返回 loss 。
        在推理时，仅输入 `qpos`, `image`，返回去噪后的动作序列。
        """
        # device = nets['policy']['linears'][0].weight.device
        # qpos = qpos.to(device)
        # image = image.to(device)
        # actions = actions.to(device)
        B = qpos.shape[0]
        nets = self.nets

        # 摄像机特征提取
        all_features = []
        for cam_id in range(len(self.camera_names)):
            cam_image = image[:, cam_id]
            cam_features = nets['policy']['backbones'][cam_id](cam_image)
            pool_features = nets['policy']['pools'][cam_id](cam_features)
            pool_features = torch.flatten(pool_features, start_dim=1)
            out_features = nets['policy']['linears'][cam_id](pool_features)
            all_features.append(out_features)

        obs_cond = torch.cat(all_features + [qpos.squeeze(1)], dim=1)

        if actions is not None:  # 训练模式
            # 添加噪声
            noise = torch.randn(actions.shape, device=obs_cond.device)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=obs_cond.device
            ).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

            # 计算 L2 损失
            all_l2 = nn.functional.mse_loss(noise_pred, noise, reduction='none')
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            return {'loss': loss}
        else:  # 推理模式
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            noisy_action = torch.randn((B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action

            self.noise_scheduler.set_timesteps(self.config.num_inference_timesteps)
            for k in self.noise_scheduler.timesteps:
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
            return naction
        
    def select_action(self, obs):
        # process data
        obs = {k:torch.from_numpy(v).cuda() if isinstance(v, np.ndarray) else v for k,v in obs.items()}
        obs['image'] = obs['image']/255.0
        action = self.forward(obs['state'], obs['image'], None, None)
        return action
        

        
    
    def save_pretrained(self, save_directory, state_dict=None, *args, **kwargs):
        """
        保存模型权重和配置到指定目录。
        
        Args:
            save_directory (str): 模型保存的文件夹路径。
            state_dict (dict, optional): 模型的权重。若为 None，则从 `self.state_dict()` 获取。
            *args, **kwargs: 兼容 `Trainer` 传递的其他参数。
        """
        # 如果保存目录不存在，则创建
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存配置文件到 {save_directory}/config.json
        self.config.save_pretrained(save_directory)

        # 如果未提供 state_dict，则从模型中获取
        if state_dict is None:
            state_dict = self.state_dict()

        # 保存模型权重到 {save_directory}/pytorch_model.bin
        model_save_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(state_dict, model_save_path)

        # 如果 EMA 也在使用，我们保存 EMA 的平均权重
        if self.ema is not None:
            ema_save_path = os.path.join(save_directory, "ema_model.bin")
            torch.save(self.ema.averaged_model.state_dict(), ema_save_path)
            
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        加载预训练模型和它的配置。
        """
        # 1. 加载配置文件
        config = DiffusionPolicyConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # 2. 初始化模型
        model = cls(config)

        # 3. 加载模型权重
        model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        # 4. 加载 EMA 权重（如果存在）
        ema_path = os.path.join(pretrained_model_name_or_path, "ema_model.bin")
        if os.path.isfile(ema_path) and model.ema is not None:
            ema_state_dict = torch.load(ema_path, map_location="cpu")
            model.ema.averaged_model.load_state_dict(ema_state_dict)

        return model

def replace_submodules(
        root_module, 
        predicate, 
        func):
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module

