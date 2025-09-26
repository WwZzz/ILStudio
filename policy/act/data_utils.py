import torch
from torch.nn.utils.rnn import pad_sequence

def data_collator(instances):
    """
    Collates a list of samples into a batch for training the DiffusionPolicyModel.
    
    Args:
        samples (list): A list of individual samples from the dataset's `__getitem__` method.
                        Each sample is a dictionary containing:
                            - image: tensor of shape [N, C, H, W]
                            - state (qpos): tensor of shape [state_dim]
                            - action: tensor of shape [Ta, action_dim]
                            - is_pad: Boolean or Integer indicating padding
                            - raw_lang: string (not used)
                            - reasoning: tensor/string/any (not used)
    
    Returns:
        A dictionary containing the batched data for the model:
            - image: tensor of shape [B, N, C, H, W]
            - state: tensor of shape [B, state_dim]
            - action: tensor of shape [B, Ta, action_dim]
            - is_pad: tensor of shape [B]
    """
    if not isinstance(instances[0]['action'], torch.Tensor):
        actions = torch.tensor(np.array([instance['action'] for instance in instances]))
        states = torch.tensor(np.array([instance['state'] for instance in instances]))
    else:
        actions = torch.stack([instance['action'] for instance in instances])
        states = torch.stack([instance['state'] for instance in instances])
    is_pad_all = torch.stack([instance['is_pad'] for instance in instances])
    images = torch.stack([instance['image'] for instance in instances])/255.0
    batch = dict(
        image=images,
        actions=actions,
        qpos=states,
        is_pad=is_pad_all
    )
    # Return batched data
    return batch