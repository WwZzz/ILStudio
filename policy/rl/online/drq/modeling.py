"""
DrQ (Data-regularized Q) Policy Implementation for ILStudio.

DrQ is an image-based reinforcement learning algorithm that uses:
1. SAC (Soft Actor-Critic) as the base algorithm
2. Random shift augmentation for data regularization
3. Shared encoder between actor and critic (with conv weight tying)

Reference: Kostrikov et al., "Image Augmentation Is All You Need: Regularizing 
Deep Reinforcement Learning from Pixels" (2020)
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================================
# Utility Functions
# ============================================================================

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim: int, hidden_dim: int, output_dim: int, hidden_depth: int, 
        output_mod: Optional[nn.Module] = None) -> nn.Sequential:
    """Create MLP with ReLU activations."""
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    return nn.Sequential(*mods)


def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float):
    """Soft update target network parameters."""
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def tie_weights(src: nn.Module, trg: nn.Module):
    """Tie weights between source and target modules."""
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# ============================================================================
# Distributions
# ============================================================================

class TanhTransform(pyd.transforms.Transform):
    """Tanh transform for squashed Gaussian."""
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """Squashed Gaussian distribution for continuous control."""
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


# ============================================================================
# Network Components
# ============================================================================

class Encoder(nn.Module):
    """
    Convolutional encoder for image-based observations.
    
    Architecture: 4 conv layers with 32 filters, followed by LayerNorm.
    Input: (B, C*frame_stack, H, W)
    Output: (B, feature_dim)
    """
    def __init__(self, obs_shape: Tuple[int, ...], feature_dim: int = 50):
        super().__init__()
        assert len(obs_shape) == 3
        
        self.num_layers = 4
        self.num_filters = 32
        self.feature_dim = feature_dim

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        # Compute output size dynamically
        dummy_input = torch.zeros(1, *obs_shape)
        conv_out = self._forward_conv(dummy_input)
        conv_out_size = conv_out.view(1, -1).size(1)
        
        self.head = nn.Sequential(
            nn.Linear(conv_out_size, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

    def _forward_conv(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward through conv layers only."""
        conv = obs / 255.0  # Normalize
        for i in range(self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        return conv

    def forward(self, obs: torch.Tensor, detach: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Image observation (B, C*k, H, W)
            detach: If True, detach conv features (for actor update)
            
        Returns:
            Feature vector (B, feature_dim)
        """
        h = self._forward_conv(obs)
        h = h.view(h.size(0), -1)
        
        if detach:
            h = h.detach()
        
        out = self.head(h)
        out = torch.tanh(out)
        return out

    def copy_conv_weights_from(self, source: 'Encoder'):
        """Tie convolutional layers with source encoder."""
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])


class Actor(nn.Module):
    """
    Actor network that outputs a squashed Gaussian distribution.
    """
    def __init__(
        self, 
        obs_shape: Tuple[int, ...],
        action_dim: int,
        hidden_dim: int = 1024,
        hidden_depth: int = 2,
        feature_dim: int = 50,
        log_std_bounds: Tuple[float, float] = (-10, 2),
    ):
        super().__init__()
        
        self.encoder = Encoder(obs_shape, feature_dim)
        self.log_std_bounds = log_std_bounds
        
        self.trunk = mlp(
            self.encoder.feature_dim, 
            hidden_dim,
            2 * action_dim,  # mu and log_std
            hidden_depth
        )
        
        self.apply(weight_init)

    def forward(self, obs: torch.Tensor, detach_encoder: bool = False) -> SquashedNormal:
        """
        Forward pass.
        
        Args:
            obs: Image observation
            detach_encoder: If True, detach encoder features
            
        Returns:
            SquashedNormal distribution
        """
        h = self.encoder(obs, detach=detach_encoder)
        mu, log_std = self.trunk(h).chunk(2, dim=-1)

        # Constrain log_std
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        return SquashedNormal(mu, std)


class Critic(nn.Module):
    """
    Critic network using double Q-learning.
    """
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        hidden_dim: int = 1024,
        hidden_depth: int = 2,
        feature_dim: int = 50,
    ):
        super().__init__()
        
        self.encoder = Encoder(obs_shape, feature_dim)
        
        self.Q1 = mlp(
            self.encoder.feature_dim + action_dim,
            hidden_dim,
            1,
            hidden_depth
        )
        self.Q2 = mlp(
            self.encoder.feature_dim + action_dim,
            hidden_dim,
            1,
            hidden_depth
        )
        
        self.apply(weight_init)

    def forward(
        self, 
        obs: torch.Tensor, 
        action: torch.Tensor,
        detach_encoder: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Image observation
            action: Action
            detach_encoder: If True, detach encoder features
            
        Returns:
            Tuple of (Q1, Q2) values
        """
        assert obs.size(0) == action.size(0)
        h = self.encoder(obs, detach=detach_encoder)
        
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        
        return q1, q2


# ============================================================================
# DrQ Agent Configuration
# ============================================================================

@dataclass
class DrQConfig:
    """Configuration for DrQ agent."""
    # Observation
    obs_shape: Tuple[int, ...] = (9, 84, 84)  # (C*frame_stack, H, W)
    action_dim: int = 6
    action_range: Tuple[float, float] = (-1.0, 1.0)
    
    # Network architecture
    feature_dim: int = 50
    hidden_dim: int = 1024
    hidden_depth: int = 2
    log_std_bounds: Tuple[float, float] = (-10, 2)
    
    # Training hyperparameters
    discount: float = 0.99
    init_temperature: float = 0.1
    lr: float = 1e-3
    actor_update_frequency: int = 2
    critic_tau: float = 0.01
    critic_target_update_frequency: int = 2
    batch_size: int = 512
    
    # Data augmentation
    image_pad: int = 4
    
    # Device
    device: str = "cuda"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'obs_shape': self.obs_shape,
            'action_dim': self.action_dim,
            'action_range': self.action_range,
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'hidden_depth': self.hidden_depth,
            'log_std_bounds': self.log_std_bounds,
            'discount': self.discount,
            'init_temperature': self.init_temperature,
            'lr': self.lr,
            'actor_update_frequency': self.actor_update_frequency,
            'critic_tau': self.critic_tau,
            'critic_target_update_frequency': self.critic_target_update_frequency,
            'batch_size': self.batch_size,
            'image_pad': self.image_pad,
            'device': self.device,
        }


# ============================================================================
# DrQ Agent
# ============================================================================

class DrQAgent(nn.Module):
    """
    DrQ (Data-regularized Q) Agent.
    
    Key features:
    1. SAC with automatic temperature tuning
    2. Shared encoder between actor and critic (conv weight tying)
    3. Random shift augmentation applied during training
    """
    
    def __init__(self, config: DrQConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.action_range = config.action_range
        self.discount = config.discount
        self.critic_tau = config.critic_tau
        self.actor_update_frequency = config.actor_update_frequency
        self.critic_target_update_frequency = config.critic_target_update_frequency
        self.batch_size = config.batch_size
        
        # Actor
        self.actor = Actor(
            obs_shape=config.obs_shape,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            hidden_depth=config.hidden_depth,
            feature_dim=config.feature_dim,
            log_std_bounds=config.log_std_bounds,
        )
        
        # Critic and target
        self.critic = Critic(
            obs_shape=config.obs_shape,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            hidden_depth=config.hidden_depth,
            feature_dim=config.feature_dim,
        )
        self.critic_target = Critic(
            obs_shape=config.obs_shape,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            hidden_depth=config.hidden_depth,
            feature_dim=config.feature_dim,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Tie conv weights between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        
        # Temperature (alpha) for entropy regularization
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(config.init_temperature), dtype=torch.float32)
        )
        self.target_entropy = -config.action_dim
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.lr
        )
        
        self.to(self.device)
        self.train()
        self.critic_target.train()
        
        self._step = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, sample: bool = True) -> np.ndarray:
        """
        Select action given observation.
        
        Args:
            obs: Image observation (C*k, H, W)
            sample: If True, sample from distribution; else return mean
            
        Returns:
            Action array
        """
        obs = torch.FloatTensor(obs).to(self.device)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            dist = self.actor(obs)
            action = dist.sample() if sample else dist.mean
            action = action.clamp(*self.action_range)
        
        return action.cpu().numpy()[0]

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Alias for act() with deterministic flag."""
        return self.act(obs, sample=not deterministic)

    def update_critic(
        self,
        obs: torch.Tensor,
        obs_aug: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        next_obs_aug: torch.Tensor,
        not_done: torch.Tensor,
    ) -> Dict[str, float]:
        """Update critic with DrQ augmentation."""
        metrics = {}
        
        with torch.no_grad():
            # Compute target Q
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            # Augmented target Q
            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1, keepdim=True)
            target_Q1_aug, target_Q2_aug = self.critic_target(next_obs_aug, next_action_aug)
            target_V_aug = torch.min(target_Q1_aug, target_Q2_aug) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V_aug)

            # Average targets
            target_Q = (target_Q + target_Q_aug) / 2

        # Compute current Q
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Augmented Q
        Q1_aug, Q2_aug = self.critic(obs_aug, action)
        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(Q2_aug, target_Q)

        # Optimize
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics['critic_loss'] = critic_loss.item()
        metrics['q1'] = current_Q1.mean().item()
        metrics['q2'] = current_Q2.mean().item()
        
        return metrics

    def update_actor_and_alpha(self, obs: torch.Tensor) -> Dict[str, float]:
        """Update actor and temperature."""
        metrics = {}
        
        # Actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Temperature loss
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha'] = self.alpha.item()
        metrics['entropy'] = -log_prob.mean().item()
        
        return metrics

    def update(self, batch: Dict[str, torch.Tensor], detailed_timing: Optional[Dict[str, list]] = None) -> Dict[str, float]:
        """
        Update agent from a batch of transitions.
        
        Args:
            batch: Dictionary with keys:
                - obs: (B, C*k, H, W)
                - action: (B, action_dim)
                - reward: (B, 1)
                - next_obs: (B, C*k, H, W)
                - not_done: (B, 1)
                - obs_aug: (B, C*k, H, W) augmented obs
                - next_obs_aug: (B, C*k, H, W) augmented next_obs
            detailed_timing: Optional dict to collect detailed timing stats
                
        Returns:
            Dictionary of training metrics
        """
        import time
        
        self._step += 1
        metrics = {}
        
        # Critic update
        critic_start = time.perf_counter()
        critic_metrics = self.update_critic(
            obs=batch['obs'],
            obs_aug=batch['obs_aug'],
            action=batch['action'],
            reward=batch['reward'],
            next_obs=batch['next_obs'],
            next_obs_aug=batch['next_obs_aug'],
            not_done=batch['not_done'],
        )
        if detailed_timing:
            detailed_timing.setdefault('update_critic', []).append(time.perf_counter() - critic_start)
        metrics.update(critic_metrics)
        
        # Actor update (less frequent)
        if self._step % self.actor_update_frequency == 0:
            actor_start = time.perf_counter()
            actor_metrics = self.update_actor_and_alpha(batch['obs'])
            if detailed_timing:
                detailed_timing.setdefault('update_actor', []).append(time.perf_counter() - actor_start)
            metrics.update(actor_metrics)
        
        # Target update
        if self._step % self.critic_target_update_frequency == 0:
            target_start = time.perf_counter()
            soft_update_params(self.critic, self.critic_target, self.critic_tau)
            if detailed_timing:
                detailed_timing.setdefault('update_target', []).append(time.perf_counter() - target_start)
        
        return metrics

    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
            'config': self.config.to_dict(),
            'step': self._step,
        }, path)

    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha.data = checkpoint['log_alpha'].data
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        self._step = checkpoint.get('step', 0)

