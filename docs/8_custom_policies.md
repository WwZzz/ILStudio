# 8. Custom Policies

This guide explains how to add a new policy model to the framework, following the standards of the Hugging Face `transformers` library for easy integration.

## The `PreTrainedModel` and `PretrainedConfig` Pattern

To ensure seamless integration with loading, saving, and configuration, all policies should follow two main principles:

1.  **Inherit from `PreTrainedModel`**: Your main policy class should inherit from `transformers.PreTrainedModel`.
2.  **Create a Config Class**: Define a separate configuration class that inherits from `transformers.PretrainedConfig`.

This pattern allows the framework to automatically save and load your model's architecture and weights using `.from_pretrained()` and `.save_pretrained()`.

## Step 1: Define the Configuration Class

Create a class that inherits from `PretrainedConfig`. Its `__init__` method should define all the hyperparameters and architectural choices for your model.

```python
# In policy/my_cool_policy.py
from transformers import PretrainedConfig

class MyCoolPolicyConfig(PretrainedConfig):
    def __init__(
        self,
        image_feature_dim=512,
        state_dim=14,
        action_dim=14,
        hidden_size=1024,
        num_layers=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_feature_dim = image_feature_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
```

## Step 2: Implement the Policy Class

Now, create your policy class inheriting from `PreTrainedModel`.

```python
# In policy/my_cool_policy.py
from transformers import PreTrainedModel
import torch.nn as nn

class MyCoolPolicy(PreTrainedModel):
    config_class = MyCoolPolicyConfig  # Link to your config class

    def __init__(self, config: MyCoolPolicyConfig):
        super().__init__(config)
        # Define your model layers based on the config
        self.image_projector = nn.Linear(config.image_feature_dim, config.hidden_size)
        self.state_projector = nn.Linear(config.state_dim, config.hidden_size)
        
        # A simple MLP backbone
        mlp_layers = [nn.Linear(config.hidden_size * 2, config.hidden_size), nn.ReLU()] * config.num_layers
        self.backbone = nn.Sequential(*mlp_layers)
        
        self.action_head = nn.Linear(config.hidden_size, config.action_dim)

    def forward(self, image_features, state, actions=None, is_pad=None):
        """
        The forward pass is used by the trainer.
        `actions` and `is_pad` are provided during training.
        """
        # Project and concatenate inputs
        img_embed = self.image_projector(image_features)
        state_embed = self.state_projector(state)
        x = torch.cat([img_embed, state_embed], dim=-1)
        
        # Run through backbone
        x = self.backbone(x)
        
        # Get action prediction
        pred_action = self.action_head(x)
        
        if actions is not None:  # Training mode
            loss = nn.functional.l1_loss(pred_action, actions)
            return {'loss': loss}
        else:  # Inference mode
            return pred_action
            
    def select_action(self, obs_dict):
        """
        A separate method for clean inference, used by the MetaPolicy.
        """
        # Preprocess observations, move to device, etc.
        # ...
        # Call the forward pass in inference mode
        return self.forward(obs_dict['image'], obs_dict['state'])
```

## Step 3: Configure the Policy

To use your new policy, specify it in the task configuration file (`configs/task/your_task.yaml`).

```yaml
policy:
  class: "MyCoolPolicy"  # The name of your policy class from `policy.my_cool_policy`
  args:
    # Hyperparameters for your MyCoolPolicyConfig
    hidden_size: 2048
    num_layers: 4
```

The training and evaluation scripts will now be able to load your policy and its configuration correctly.
