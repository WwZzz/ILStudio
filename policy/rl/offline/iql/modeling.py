"""
IQL (Implicit Q-Learning) Agent Implementation for ILStudio.

IQL is an offline reinforcement learning algorithm that avoids querying
out-of-distribution actions by using expectile regression.

Reference:
    Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning"
    https://arxiv.org/abs/2110.06169
"""

import copy
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class IQLConfig:
    """Configuration for IQL agent."""
    # Observation and action dimensions
    obs_dim: int = 17
    action_dim: int = 6
    
    # Network architecture
    hidden_dim: int = 256
    n_hidden: int = 2
    
    # IQL hyperparameters
    discount: float = 0.99
    tau: float = 0.7  # Expectile for asymmetric loss
    beta: float = 3.0  # Temperature for advantage weighting
    alpha: float = 0.005  # EMA coefficient for target network
    
    # Training
    learning_rate: float = 3e-4
    batch_size: int = 256
    max_steps: int = 1000000
    
    # Policy type
    deterministic_policy: bool = False
    
    # Device
    device: str = "cuda"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'n_hidden': self.n_hidden,
            'discount': self.discount,
            'tau': self.tau,
            'beta': self.beta,
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_steps': self.max_steps,
            'deterministic_policy': self.deterministic_policy,
            'device': self.device,
        }


# ============================================================================
# Utility Functions
# ============================================================================

class Squeeze(nn.Module):
    """Squeeze layer for removing singleton dimensions."""
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


def mlp(
    dims: list,
    activation: type = nn.ReLU,
    output_activation: Optional[type] = None,
    squeeze_output: bool = False
) -> nn.Sequential:
    """
    Create a multi-layer perceptron.
    
    Args:
        dims: List of dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        activation: Activation function class
        output_activation: Output activation function class
        squeeze_output: Whether to squeeze the output dimension
        
    Returns:
        nn.Sequential MLP
    """
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def update_exponential_moving_average(
    target: nn.Module,
    source: nn.Module,
    alpha: float
):
    """Update target network with exponential moving average."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Asymmetric L2 loss for expectile regression.
    
    Args:
        u: Prediction errors (target - prediction)
        tau: Expectile (0.5 for median, higher for upper expectiles)
        
    Returns:
        Asymmetric L2 loss
    """
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


# ============================================================================
# Network Components
# ============================================================================

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
EXP_ADV_MAX = 100.0


class TwinQ(nn.Module):
    """
    Twin Q-network for double Q-learning.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both Q values."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return minimum of both Q values."""
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    """
    State value function V(s).
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy for continuous control.
    Outputs a multivariate normal distribution.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2
    ):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), action_dim])
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        enable_grad: bool = False
    ) -> torch.Tensor:
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()


class DeterministicPolicy(nn.Module):
    """
    Deterministic policy for continuous control.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2
    ):
        super().__init__()
        self.net = mlp(
            [obs_dim, *([hidden_dim] * n_hidden), action_dim],
            output_activation=nn.Tanh
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        enable_grad: bool = False
    ) -> torch.Tensor:
        with torch.set_grad_enabled(enable_grad):
            return self(obs)


# ============================================================================
# IQL Agent
# ============================================================================

class IQLAgent(nn.Module):
    """
    Implicit Q-Learning (IQL) Agent.
    
    IQL avoids querying out-of-distribution actions by:
    1. Learning V(s) via expectile regression on Q(s, a)
    2. Using V(s) as the target for Q-learning
    3. Extracting policy via advantage-weighted regression
    
    This design eliminates the need to query Q-values on unseen actions
    during training, making it suitable for offline RL.
    """
    
    def __init__(self, config: IQLConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Q-networks
        self.qf = TwinQ(
            state_dim=config.obs_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden
        )
        self.q_target = copy.deepcopy(self.qf)
        self.q_target.requires_grad_(False)
        
        # Value function
        self.vf = ValueFunction(
            state_dim=config.obs_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden
        )
        
        # Policy
        if config.deterministic_policy:
            self.policy = DeterministicPolicy(
                obs_dim=config.obs_dim,
                action_dim=config.action_dim,
                hidden_dim=config.hidden_dim,
                n_hidden=config.n_hidden
            )
        else:
            self.policy = GaussianPolicy(
                obs_dim=config.obs_dim,
                action_dim=config.action_dim,
                hidden_dim=config.hidden_dim,
                n_hidden=config.n_hidden
            )
        
        # Optimizers
        self.v_optimizer = torch.optim.Adam(
            self.vf.parameters(), lr=config.learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            self.qf.parameters(), lr=config.learning_rate
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )
        
        # Learning rate scheduler for policy
        self.policy_lr_schedule = CosineAnnealingLR(
            self.policy_optimizer, config.max_steps
        )
        
        # Hyperparameters
        self.tau = config.tau
        self.beta = config.beta
        self.discount = config.discount
        self.alpha = config.alpha
        
        # Move to device
        self.to(self.device)
        
        # Step counter
        self._step = 0

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action given observation.
        
        Args:
            obs: Observation array
            deterministic: If True, return mean action
            
        Returns:
            Action array
        """
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            action = self.policy.act(obs, deterministic=deterministic)
        
        return action.cpu().numpy().squeeze(0)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Alias for act()."""
        return self.act(obs, deterministic=deterministic)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform one IQL update step.
        
        Args:
            observations: Current observations (B, obs_dim)
            actions: Actions taken (B, action_dim)
            next_observations: Next observations (B, obs_dim)
            rewards: Rewards received (B,) or (B, 1)
            terminals: Terminal flags (B,) or (B, 1)
            
        Returns:
            Dictionary of training metrics
        """
        self._step += 1
        metrics = {}
        
        # Ensure correct shapes
        if rewards.dim() == 2:
            rewards = rewards.squeeze(-1)
        if terminals.dim() == 2:
            terminals = terminals.squeeze(-1)
        
        # Compute targets with no gradient
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)
        
        # =====================================================================
        # Update Value Function
        # =====================================================================
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        
        metrics['v_loss'] = v_loss.item()
        metrics['v_mean'] = v.mean().item()
        
        # =====================================================================
        # Update Q Function
        # =====================================================================
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        q1, q2 = self.qf.both(observations, actions)
        q_loss = (F.mse_loss(q1, targets) + F.mse_loss(q2, targets)) / 2
        
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        
        metrics['q_loss'] = q_loss.item()
        metrics['q1_mean'] = q1.mean().item()
        metrics['q2_mean'] = q2.mean().item()
        
        # =====================================================================
        # Update Target Q Network
        # =====================================================================
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)
        
        # =====================================================================
        # Update Policy (Advantage-Weighted Regression)
        # =====================================================================
        # Compute advantage weights
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        
        # Compute policy loss (weighted BC)
        policy_out = self.policy(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            # Gaussian policy: negative log probability
            bc_losses = -policy_out.log_prob(actions).sum(dim=-1)
        elif torch.is_tensor(policy_out):
            # Deterministic policy: MSE
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=-1)
        else:
            raise NotImplementedError(f"Unknown policy output type: {type(policy_out)}")
        
        policy_loss = torch.mean(exp_adv * bc_losses)
        
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()
        
        metrics['policy_loss'] = policy_loss.item()
        metrics['adv_mean'] = adv.mean().item()
        metrics['adv_max'] = adv.max().item()
        metrics['exp_adv_mean'] = exp_adv.mean().item()
        
        return metrics

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'qf': self.qf.state_dict(),
            'q_target': self.q_target.state_dict(),
            'vf': self.vf.state_dict(),
            'policy': self.policy.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'policy_lr_schedule': self.policy_lr_schedule.state_dict(),
            'config': self.config.to_dict(),
            'step': self._step,
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.qf.load_state_dict(checkpoint['qf'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        self.vf.load_state_dict(checkpoint['vf'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        if 'policy_lr_schedule' in checkpoint:
            self.policy_lr_schedule.load_state_dict(checkpoint['policy_lr_schedule'])
        if 'step' in checkpoint:
            self._step = checkpoint['step']

