from .mlp import MLPPolicy, MLPPolicyConfig
from .data_utils import data_collator, MLPDataProcessor
import torch
import numpy as np


def load_model(args):
    """
    Load MLP model according to the framework requirements.
    
    This function provides two functionalities:
    1) Load original model directly
    2) Load checkpoint model trained by the framework
    
    Args:
        args: Arguments containing model configuration
        
    Returns:
        dict: Dictionary containing at least 'model' key with the model instance
    """
    if args.is_pretrained:
        # Load from pretrained checkpoint
        model = MLPPolicy.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model.to(args.device if hasattr(args, 'device') else 'cuda')
    else:
        # Create new model from configuration
        model_args = getattr(args, 'model_args', {})
        
        # Calculate image dimension if using camera
        image_dim = None
        if getattr(args, 'use_camera', False):
            # Calculate flattened image dimension
            # This should be set based on the actual image dimensions from the environment
            image_shapes = getattr(args, 'image_shapes', None)
            if image_shapes:
                # Sum up all camera image dimensions
                image_dim = sum(np.prod(shape) for shape in image_shapes)
            else:
                # Default assumption for debugging (can be overridden)
                image_dim = 3 * 224 * 224  # Default RGB image
        
        # Extract configuration parameters
        config = MLPPolicyConfig(
            state_dim=getattr(args, 'state_dim', 14),
            action_dim=getattr(args, 'action_dim', 14),
            num_layers=getattr(args, 'num_layers', 3),
            hidden_dim=getattr(args, 'hidden_dim', 256),
            activation=getattr(args, 'activation', 'relu'),
            dropout=getattr(args, 'dropout', 0.0),
            use_camera=getattr(args, 'use_camera', False),
            image_dim=image_dim,
            learning_rate=getattr(args, 'learning_rate', 1e-3),
            chunk_size=getattr(args, 'chunk_size', 1),
            **model_args
        )
        
        model = MLPPolicy(config=config)
        
        # Move to device
        device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    
    return {'model': model}


def get_data_collator(args, model_components):
    """
    Get data collator for MLP policy.
    
    This function returns a callable object that can organize multiple samples
    processed by get_data_processor into a batch format that the model can accept.
    This shields the differences between different models and makes it easier
    to integrate into IL-Studio with minimal changes.
    
    Args:
        args: Arguments containing configuration
        model_components: Dictionary containing model components from load_model
        
    Returns:
        Callable: Data collator function
    """
    return data_collator


def get_data_processor(args, model_components):
    """
    Get data processor for MLP policy.
    
    This function returns a callable object that can convert data returned by
    dataset.__getitem__ into a format that the model can accept through format
    conversion, adapting to different input models. It returns a Callable object
    that can perform sample-level transformations.
    
    Args:
        args: Arguments containing configuration
        model_components: Dictionary containing model components from load_model
        
    Returns:
        Callable: Data processor object for sample-level transformation
    """
    # Extract configuration from model if available
    model = model_components.get('model')
    if hasattr(model, 'config'):
        state_dim = model.config.state_dim
    else:
        state_dim = getattr(args, 'state_dim', None)
    
    # Get use_camera setting
    use_camera = getattr(args, 'use_camera', False)
    if hasattr(model, 'config'):
        use_camera = model.config.use_camera
    
    # Create and return the processor
    processor = MLPDataProcessor(
        state_dim=state_dim,
        use_camera=use_camera
    )
    
    return processor


# Export the required interfaces for the framework
__all__ = [
    'load_model',
    'get_data_collator', 
    'get_data_processor',
    'MLPPolicy',
    'MLPPolicyConfig'
]


if __name__ == "__main__":
    # Test the interfaces
    from argparse import Namespace
    
    print("Testing MLP policy interfaces...")
    
    # Create mock args
    args = Namespace(
        is_pretrained=False,
        state_dim=14,
        action_dim=14,
        num_layers=3,
        hidden_dim=256,
        activation='relu',
        dropout=0.1,
        learning_rate=1e-3,
        chunk_size=1,
        device='cpu'
    )
    
    # Test load_model
    model_components = load_model(args)
    print(f"✓ load_model returned: {list(model_components.keys())}")
    
    # Test get_data_processor
    processor = get_data_processor(args, model_components)
    print(f"✓ get_data_processor returned: {type(processor).__name__}")
    
    # Test get_data_collator
    collator = get_data_collator(args, model_components)
    print(f"✓ get_data_collator returned: {collator.__name__}")
    
    # Test a complete pipeline
    from .data_utils import get_dummy_data
    
    # Generate and process dummy data
    dummy_data = get_dummy_data(batch_size=4, state_dim=14, action_dim=14)
    processed_data = [processor(sample) for sample in dummy_data]
    batched_data = collator(processed_data)
    
    # Test model prediction
    model = model_components['model']
    model.eval()
    
    with torch.no_grad():
        output = model(batched_data['state'])
        print(f"✓ Model output shape: {output['action'].shape}")
    
    # Test select_action method
    obs_dict = {'state': batched_data['state'][0].numpy()}
    action = model.select_action(obs_dict)
    print(f"✓ select_action output shape: {action.shape}")
    
    print("All interface tests passed! 🎉")
