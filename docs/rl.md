# ILStudio 强化学习集成架构

本文档描述了如何将强化学习 (RL) 集成到 ILStudio 框架中，支持：
- 传统 RL 训练机器人控制策略 (PPO, SAC)
- VLA (Vision-Language-Action) 模型的强化学习微调 (DPO, GRPO)
- 混合训练 (IL + RL)

## 1. 框架现状分析

ILStudio 采用清晰的解耦设计：

| 组件 | 职责 |
|------|------|
| `benchmark/` | 环境抽象层 (`MetaEnv`, `MetaObs`, `MetaAction`) |
| `policy/` | 策略模型定义 (`select_action`, `compute_loss`) |
| `data_utils/` | 数据集处理 |
| `configs/` | 配置管理 (task, policy, training, env) |
| `train.py` | 监督学习训练入口 |
| `eval_sim.py` | 仿真评估入口 |

## 2. 强化学习集成架构

### 2.1 目录结构

```
ILStudio/
├── rl/                          # 新增：强化学习模块
│   ├── __init__.py
│   ├── base.py                  # RL 基类定义
│   ├── algorithms/              # RL 算法实现
│   │   ├── __init__.py
│   │   ├── ppo.py
│   │   ├── sac.py
│   │   ├── dpo.py               # Direct Preference Optimization (VLA)
│   │   ├── reinforce.py
│   │   └── grpo.py              # Group Relative Policy Optimization (VLA)
│   ├── buffer/                  # 经验回放
│   │   ├── __init__.py
│   │   ├── replay_buffer.py
│   │   ├── rollout_buffer.py
│   │   └── priority_buffer.py
│   ├── collectors/              # 数据收集器
│   │   ├── __init__.py
│   │   ├── base_collector.py
│   │   ├── sim_collector.py    # 仿真环境收集
│   │   └── real_collector.py   # 真实机器人收集
│   ├── rewards/                 # 奖励函数
│   │   ├── __init__.py
│   │   ├── sparse_reward.py
│   │   ├── dense_reward.py
│   │   └── learned_reward.py   # 学习的奖励模型
│   └── trainer.py               # RL Trainer
├── configs/
│   ├── rl/                      # 新增：RL 配置
│   │   ├── ppo.yaml
│   │   ├── sac.yaml
│   │   └── dpo.yaml
│   └── ...
├── train_rl.py                  # 新增：RL 训练入口
└── ...
```

### 2.2 核心抽象设计

#### 2.2.1 `rl/base.py` - RL 基类

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch
import numpy as np
from benchmark.base import MetaObs, MetaAction, MetaPolicy

@dataclass
class RLConfig:
    """RL 算法通用配置"""
    gamma: float = 0.99                    # 折扣因子
    gae_lambda: float = 0.95               # GAE lambda
    clip_range: float = 0.2                # PPO clip range
    value_coef: float = 0.5                # Value loss 系数
    entropy_coef: float = 0.01             # 熵正则化系数
    max_grad_norm: float = 0.5             # 梯度裁剪
    n_steps: int = 2048                    # 每次更新的步数
    batch_size: int = 64                   # Mini-batch 大小
    n_epochs: int = 10                     # 每次更新的 epoch 数
    learning_rate: float = 3e-4
    
    # VLA 特定配置
    use_language_reward: bool = False      # 是否使用语言条件奖励
    kl_coef: float = 0.1                   # KL 散度惩罚系数 (用于 VLA fine-tuning)


class RLPolicy(ABC):
    """
    强化学习策略基类
    
    设计原则：
    - 继承/组合 MetaPolicy 以复用推理逻辑
    - 添加 RL 特有的方法 (value estimation, log_prob, etc.)
    """
    
    def __init__(self, 
                 policy_model,           # 基础策略模型 (ACT, DP, VLA 等)
                 value_model=None,       # 价值网络 (可选，Actor-Critic)
                 config: RLConfig = None):
        self.policy = policy_model
        self.value = value_model
        self.config = config or RLConfig()
    
    @abstractmethod
    def select_action(self, obs: MetaObs, deterministic: bool = False) -> MetaAction:
        """选择动作，支持确定性和随机采样"""
        pass
    
    @abstractmethod
    def evaluate_actions(self, obs: MetaObs, actions: MetaAction) -> Dict[str, torch.Tensor]:
        """
        评估动作，返回：
        - log_prob: 动作的对数概率
        - entropy: 策略熵
        - value: 状态价值估计 (如果有 value network)
        """
        pass
    
    @abstractmethod
    def compute_rl_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """计算 RL 损失，返回各项损失用于日志"""
        pass
    
    def get_value(self, obs: MetaObs) -> torch.Tensor:
        """获取状态价值估计"""
        if self.value is not None:
            return self.value(obs)
        return None


class BaseRLAlgorithm(ABC):
    """RL 算法基类"""
    
    def __init__(self, 
                 policy: RLPolicy,
                 env,                     # MetaEnv 或 VectorEnv
                 config: RLConfig,
                 device: str = 'cuda'):
        self.policy = policy
        self.env = env
        self.config = config
        self.device = device
    
    @abstractmethod
    def collect_rollouts(self, n_steps: int) -> Dict:
        """收集交互数据"""
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """执行一次训练更新，返回日志字典"""
        pass
    
    @abstractmethod
    def learn(self, total_timesteps: int) -> None:
        """主训练循环"""
        pass
```

#### 2.2.2 `rl/buffer/rollout_buffer.py` - 经验存储

```python
import numpy as np
import torch
from typing import Dict, Generator, Optional
from benchmark.base import MetaObs, MetaAction

class RolloutBuffer:
    """
    On-policy 算法的 Rollout Buffer
    
    存储格式与 ILStudio 的 MetaObs/MetaAction 兼容
    """
    
    def __init__(self, 
                 buffer_size: int,
                 obs_space: Dict,
                 action_space: Dict,
                 device: str = 'cpu',
                 n_envs: int = 1,
                 gae_lambda: float = 0.95,
                 gamma: float = 0.99):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        
        # 初始化存储
        self.observations = {}  # Dict of arrays matching MetaObs structure
        self.actions = None
        self.rewards = None
        self.dones = None
        self.values = None
        self.log_probs = None
        self.advantages = None
        self.returns = None
        
        self.pos = 0
        self.full = False
        
    def add(self, 
            obs: MetaObs, 
            action: MetaAction, 
            reward: float, 
            done: bool, 
            value: torch.Tensor,
            log_prob: torch.Tensor):
        """添加一条经验"""
        # 存储逻辑...
        pass
    
    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray):
        """计算 GAE 和 returns"""
        pass
    
    def get(self, batch_size: int) -> Generator[Dict, None, None]:
        """生成 mini-batch 用于训练"""
        pass


class ReplayBuffer:
    """
    Off-policy 算法的 Replay Buffer
    
    支持优先级采样 (PER)
    """
    
    def __init__(self, 
                 buffer_size: int,
                 obs_space: Dict,
                 action_space: Dict,
                 device: str = 'cpu',
                 prioritized: bool = False,
                 alpha: float = 0.6):
        pass
    
    def add(self, obs, action, reward, next_obs, done, info=None):
        pass
    
    def sample(self, batch_size: int) -> Dict:
        pass
```

#### 2.2.3 `rl/collectors/sim_collector.py` - 数据收集

```python
from typing import Dict, Optional, Callable
import numpy as np
import torch
from benchmark.base import MetaEnv, MetaObs, MetaAction
from benchmark.utils import SequentialVectorEnv

class SimCollector:
    """
    仿真环境数据收集器
    
    复用 ILStudio 的 SequentialVectorEnv 和 MetaEnv 抽象
    """
    
    def __init__(self,
                 env,                      # MetaEnv 或 VectorEnv
                 policy,                   # RLPolicy
                 buffer,                   # RolloutBuffer 或 ReplayBuffer
                 reward_fn: Optional[Callable] = None,  # 自定义奖励函数
                 n_envs: int = 1):
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.reward_fn = reward_fn
        self.n_envs = n_envs
        
        self._last_obs = None
        self._last_dones = None
    
    def collect(self, n_steps: int) -> Dict[str, float]:
        """
        收集 n_steps 的交互数据
        
        Returns:
            统计信息 (episode_reward, episode_length, etc.)
        """
        if self._last_obs is None:
            self._last_obs = self.env.reset()
        
        stats = {'episode_rewards': [], 'episode_lengths': []}
        
        for step in range(n_steps):
            # 获取动作
            with torch.no_grad():
                action, value, log_prob = self.policy.select_action(
                    self._last_obs, 
                    return_value=True,
                    return_log_prob=True
                )
            
            # 环境交互
            new_obs, reward, done, info = self.env.step(action)
            
            # 自定义奖励
            if self.reward_fn is not None:
                reward = self.reward_fn(self._last_obs, action, new_obs, reward, info)
            
            # 存储经验
            self.buffer.add(
                obs=self._last_obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob
            )
            
            self._last_obs = new_obs
            
            # 统计
            for i, d in enumerate(done):
                if d:
                    if 'episode' in info[i]:
                        stats['episode_rewards'].append(info[i]['episode']['r'])
                        stats['episode_lengths'].append(info[i]['episode']['l'])
        
        return stats
```

### 2.3 VLA 强化学习特殊处理

对于 VLA (Vision-Language-Action) 模型的强化学习，需要特殊考虑：

#### 2.3.1 `rl/algorithms/dpo.py` - Direct Preference Optimization

```python
"""
DPO for VLA fine-tuning
适用于有人类偏好数据的场景
"""

import torch
import torch.nn.functional as F
from .base import BaseRLAlgorithm, RLConfig
from dataclasses import dataclass

@dataclass
class DPOConfig(RLConfig):
    """DPO 特定配置"""
    beta: float = 0.1                      # KL 惩罚系数
    reference_free: bool = False           # 是否使用无参考模型的 DPO
    label_smoothing: float = 0.0


class DPOTrainer(BaseRLAlgorithm):
    """
    Direct Preference Optimization for VLA
    
    适用于：
    - 有偏好数据 (chosen vs rejected trajectories)
    - Fine-tuning 预训练 VLA 模型
    """
    
    def __init__(self, 
                 policy,
                 ref_policy,               # 参考策略 (frozen)
                 config: DPOConfig,
                 device: str = 'cuda'):
        super().__init__(policy, None, config, device)
        self.ref_policy = ref_policy
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False
    
    def compute_dpo_loss(self, 
                         chosen_obs, chosen_actions,
                         rejected_obs, rejected_actions):
        """
        计算 DPO 损失
        
        L_DPO = -log(σ(β * (log π(a_w|s) - log π_ref(a_w|s) 
                         - log π(a_l|s) + log π_ref(a_l|s))))
        """
        # 计算当前策略的 log prob
        chosen_logps = self.policy.get_log_prob(chosen_obs, chosen_actions)
        rejected_logps = self.policy.get_log_prob(rejected_obs, rejected_actions)
        
        # 计算参考策略的 log prob
        with torch.no_grad():
            ref_chosen_logps = self.ref_policy.get_log_prob(chosen_obs, chosen_actions)
            ref_rejected_logps = self.ref_policy.get_log_prob(rejected_obs, rejected_actions)
        
        # DPO 损失
        chosen_rewards = self.config.beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = self.config.beta * (rejected_logps - ref_rejected_logps)
        
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        return {
            'loss': loss,
            'chosen_rewards': chosen_rewards.mean(),
            'rejected_rewards': rejected_rewards.mean(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean()
        }
```

#### 2.3.2 `rl/algorithms/grpo.py` - Group Relative Policy Optimization

```python
"""
GRPO for VLA - 适用于在线 RL fine-tuning
参考: DeepSeek-R1, OpenAI RLHF
"""

from dataclasses import dataclass

@dataclass
class GRPOConfig(RLConfig):
    """GRPO 配置"""
    group_size: int = 4                    # 每个 prompt 采样的 response 数量
    kl_coef: float = 0.1                   # KL 散度系数
    reward_scale: float = 1.0


class GRPOTrainer(BaseRLAlgorithm):
    """
    Group Relative Policy Optimization
    
    适用于 VLA 的在线强化学习：
    1. 对每个任务/指令采样多个轨迹
    2. 使用奖励对轨迹进行排序
    3. 使用组内相对奖励进行策略优化
    """
    
    def collect_group_rollouts(self, obs: MetaObs, n_samples: int):
        """
        对同一观测采样多个动作序列
        """
        trajectories = []
        for _ in range(n_samples):
            traj = self._rollout_episode(obs)
            trajectories.append(traj)
        return trajectories
    
    def compute_grpo_loss(self, trajectories, rewards):
        """
        计算 GRPO 损失
        
        使用组内相对奖励作为 advantage
        """
        # 归一化组内奖励
        rewards = torch.tensor(rewards)
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        loss = 0
        for traj, adv in zip(trajectories, normalized_rewards):
            log_probs = self.policy.get_trajectory_log_prob(traj)
            loss -= (log_probs * adv).mean()
        
        # KL 惩罚
        kl_loss = self._compute_kl_penalty(trajectories)
        
        return loss + self.config.kl_coef * kl_loss
```

### 2.4 配置系统扩展

#### `configs/rl/ppo.yaml`

```yaml
name: ppo
type: rl.algorithms.ppo

# 算法参数
algorithm:
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  n_steps: 2048
  batch_size: 64
  n_epochs: 10

# 策略网络配置 (复用现有 policy 配置)
policy:
  type: policy.act  # 或 policy.diffusion_policy, policy.openvla 等
  # 继承对应 policy 的配置...

# 价值网络配置 (可选)
value_network:
  type: mlp
  hidden_dims: [256, 256]
  activation: relu

# 收集器配置
collector:
  n_envs: 8
  n_steps_per_collect: 2048

# 训练配置
training:
  total_timesteps: 1000000
  learning_rate: 3e-4
  save_freq: 10000
  eval_freq: 5000
```

### 2.5 训练入口 `train_rl.py`

```python
#!/usr/bin/env python3
"""
ILStudio Reinforcement Learning Training Script

支持:
- 传统 RL (PPO, SAC) 训练机器人控制策略
- VLA fine-tuning (DPO, GRPO) 使用强化学习
- 混合训练 (IL + RL)
"""

import configs
import argparse
from loguru import logger
from configs.loader import ConfigLoader
from data_utils.utils import set_seed
from policy.policy_loader import load_policy_model_for_training
from rl.algorithms import get_rl_algorithm
from rl.collectors import get_collector
from rl.buffer import get_buffer

def parse_args():
    parser = argparse.ArgumentParser(description='RL Training for ILStudio')
    
    # 基础配置
    parser.add_argument('-p', '--policy', type=str, required=True,
                       help='Policy config (e.g., act, diffusion_policy, openvla)')
    parser.add_argument('-r', '--rl_config', type=str, default='ppo',
                       help='RL algorithm config (e.g., ppo, sac, dpo)')
    parser.add_argument('-e', '--env', type=str, required=True,
                       help='Environment config')
    parser.add_argument('-o', '--output_dir', type=str, default='ckpt/rl_output')
    
    # 训练模式
    parser.add_argument('--mode', type=str, default='online',
                       choices=['online', 'offline', 'hybrid'],
                       help='Training mode: online (env interaction), offline (dataset), hybrid')
    
    # 预训练模型 (用于 fine-tuning)
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained model checkpoint')
    
    # 其他
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    
    args, unknown = parser.parse_known_args()
    args.unknown_args = unknown
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 加载配置
    cfg_loader = ConfigLoader(args=args, unknown_args=args.unknown_args)
    policy_config, _ = cfg_loader.load_policy(args.policy)
    rl_config, _ = cfg_loader.load_rl(args.rl_config)
    env_config, _ = cfg_loader.load_env(args.env)
    
    # 创建环境
    env = create_env_from_config(env_config)
    
    # 加载/创建策略
    if args.pretrained:
        logger.info(f"Loading pretrained model from {args.pretrained}")
        policy = load_policy_model_for_training(args.pretrained, args)
    else:
        logger.info("Creating policy from scratch")
        policy = create_policy_from_config(policy_config, args)
    
    # 创建 RL 算法
    rl_algorithm = get_rl_algorithm(
        name=rl_config.name,
        policy=policy,
        env=env,
        config=rl_config,
        device=args.device
    )
    
    # 训练
    logger.info("="*60)
    logger.info(f"🚀 Starting RL Training: {rl_config.name}")
    logger.info(f"   Policy: {policy_config.name}")
    logger.info(f"   Environment: {env_config.name}")
    logger.info(f"   Mode: {args.mode}")
    logger.info("="*60)
    
    rl_algorithm.learn(
        total_timesteps=rl_config.training.total_timesteps,
        callback=create_callbacks(args)
    )
    
    logger.info("✓ Training completed!")


if __name__ == '__main__':
    main()
```

## 3. 集成路线图

### Phase 1: 基础设施 (1-2 周)
- [ ] 创建 `rl/` 目录结构
- [ ] 实现 `RLPolicy` 和 `BaseRLAlgorithm` 基类
- [ ] 实现 `RolloutBuffer` 和 `ReplayBuffer`
- [ ] 实现 `SimCollector`

### Phase 2: 经典 RL 算法 (2-3 周)
- [ ] 实现 PPO (on-policy)
- [ ] 实现 SAC (off-policy)
- [ ] 集成 `train_rl.py` 入口
- [ ] 添加 RL 配置系统

### Phase 3: VLA 强化学习 (2-3 周)
- [ ] 实现 DPO trainer
- [ ] 实现 GRPO trainer
- [ ] 添加 KL 散度约束
- [ ] 支持语言条件奖励

### Phase 4: 高级功能 (持续)
- [ ] 混合训练 (IL + RL)
- [ ] 真实机器人数据收集
- [ ] 奖励模型学习
- [ ] 分布式训练支持

## 4. 关键设计原则

1. **复用现有抽象**: 使用 `MetaEnv`, `MetaObs`, `MetaAction` 保持环境接口一致
2. **策略模型无关**: RL 算法应能与任何 ILStudio policy (ACT, DP, VLA) 配合
3. **配置驱动**: 通过 YAML 配置切换算法、环境、策略
4. **模块化**: Buffer、Collector、Algorithm 独立可替换
5. **兼容 HuggingFace**: Trainer 可选继承 `transformers.Trainer` 以复用其生态

## 5. 使用示例

### 5.1 传统 RL 训练 (PPO + ACT)

```bash
python train_rl.py \
    -p act \
    -r ppo \
    -e libero.example \
    -o ckpt/rl_act_ppo
```

### 5.2 VLA Fine-tuning (DPO + OpenVLA)

```bash
python train_rl.py \
    -p openvla \
    -r dpo \
    -e behavior1k.example \
    --pretrained ckpt/openvla_pretrained \
    -o ckpt/openvla_dpo
```

### 5.3 混合训练 (IL + RL)

```bash
python train_rl.py \
    -p diffusion_policy \
    -r ppo \
    -e metaworld.example \
    --mode hybrid \
    --pretrained ckpt/dp_il_pretrained \
    -o ckpt/dp_hybrid
```

## 6. 参考资料

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL 算法实现参考
- [TRL](https://github.com/huggingface/trl) - HuggingFace 的 RLHF 库
- [OpenAI Spinning Up](https://spinningup.openai.com/) - RL 算法教程
- [DeepSeek-R1](https://arxiv.org/abs/2401.02954) - GRPO 算法参考


