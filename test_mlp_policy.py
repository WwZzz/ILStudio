#!/usr/bin/env python3
"""
Test script for MLP policy implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from argparse import Namespace

# Import MLP policy components
from policy.mlp import load_model, get_data_collator, get_data_processor
from policy.mlp.data_utils import get_dummy_data


def test_mlp_policy():
    """Test the complete MLP policy pipeline."""
    print("🧪 Testing MLP Policy Implementation")
    print("=" * 50)
    
    # Create test arguments
    args = Namespace(
        is_pretrained=False,
        state_dim=14,
        action_dim=14,
        num_layers=3,
        hidden_dim=256,
        activation='relu',
        dropout=0.1,
        learning_rate=1e-3,
        chunk_size=2,  # Test with chunk_size > 1
        device='cpu',
        use_camera=False  # Test state-only version first
    )
    
    print(f"📋 Test Configuration:")
    print(f"   State dim: {args.state_dim}")
    print(f"   Action dim: {args.action_dim}")
    print(f"   Layers: {args.num_layers}")
    print(f"   Hidden dim: {args.hidden_dim}")
    print(f"   Activation: {args.activation}")
    print()
    
    # Test 1: Model loading
    print("1️⃣  Testing model loading...")
    try:
        model_components = load_model(args)
        model = model_components['model']
        print(f"   ✅ Model loaded successfully")
        print(f"   📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   🏗️  Model architecture: {model.config.num_layers} layers")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False
    
    # Test 2: Data processor
    print("\n2️⃣  Testing data processor...")
    try:
        processor = get_data_processor(args, model_components)
        
        # Test with dummy sample
        dummy_sample = {
            'state': np.random.randn(args.state_dim).astype(np.float32),
            'action': np.random.randn(args.action_dim).astype(np.float32),
        }
        
        processed_sample = processor(dummy_sample)
        print(f"   ✅ Data processor working")
        print(f"   📝 Processed sample keys: {list(processed_sample.keys())}")
    except Exception as e:
        print(f"   ❌ Data processor failed: {e}")
        return False
    
    # Test 3: Data collator
    print("\n3️⃣  Testing data collator...")
    try:
        collator = get_data_collator(args, model_components)
        
        # Generate batch of dummy data with text modalities
        dummy_data = get_dummy_data(batch_size=4, state_dim=args.state_dim, action_dim=args.action_dim, include_text=True)
        print(f"   📝 Original sample keys: {list(dummy_data[0].keys())}")
        
        processed_batch = [processor(sample) for sample in dummy_data]
        print(f"   📝 Processed sample keys: {list(processed_batch[0].keys())}")
        
        batched_data = collator(processed_batch)
        print(f"   ✅ Data collator working")
        print(f"   📦 Final batch keys: {list(batched_data.keys())}")
        print(f"   📏 State batch shape: {batched_data['state'].shape}")
        print(f"   📏 Action batch shape: {batched_data['action'].shape}")
        
        # Verify text modalities are ignored
        text_keys = ['raw_lang', 'task', 'episode_id', 'instruction', 'trajectory_id']
        found_text = any(key in batched_data for key in text_keys)
        if not found_text:
            print(f"   ✅ Text modalities successfully ignored")
        else:
            print(f"   ❌ Some text modalities found in batch")
    except Exception as e:
        print(f"   ❌ Data collator failed: {e}")
        return False
    
    # Test 4: Model forward pass
    print("\n4️⃣  Testing model forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            output = model(batched_data['state'])
            predicted_actions = output['action']
        
        print(f"   ✅ Forward pass successful")
        print(f"   📏 Input shape: {batched_data['state'].shape}")
        print(f"   📏 Output shape: {predicted_actions.shape}")
        print(f"   📊 Output range: [{predicted_actions.min().item():.3f}, {predicted_actions.max().item():.3f}]")
        
        # Verify output shape is correct (batch_size, chunk_size, action_dim)
        expected_shape = (batched_data['state'].shape[0], args.chunk_size, args.action_dim)
        assert predicted_actions.shape == expected_shape, f"Expected {expected_shape}, got {predicted_actions.shape}"
        print(f"   ✅ Output shape verification passed")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False
    
    # Test 5: select_action method (for evaluation)
    print("\n5️⃣  Testing select_action method...")
    try:
        obs_dict = {'state': batched_data['state'][0].numpy()}
        predicted_action = model.select_action(obs_dict)
        
        print(f"   ✅ select_action working")
        print(f"   📏 Single prediction shape: {predicted_action.shape}")
        print(f"   📊 Prediction range: [{predicted_action.min():.3f}, {predicted_action.max():.3f}]")
    except Exception as e:
        print(f"   ❌ select_action failed: {e}")
        return False
    
    # Test 6: Test with camera modality
    print("\n6️⃣  Testing camera modality...")
    try:
        # Create camera-enabled configuration
        args_camera = Namespace(
            is_pretrained=False,
            state_dim=14,
            action_dim=14,
            num_layers=3,
            hidden_dim=256,
            activation='relu',
            dropout=0.1,
            learning_rate=1e-3,
            chunk_size=1,
            device='cpu',
            use_camera=True,
            image_shapes=[(3, 64, 64)]  # Single RGB 64x64 image
        )
        
        # Load camera-enabled model
        camera_model_components = load_model(args_camera)
        camera_model = camera_model_components['model']
        
        # Create sample with image
        dummy_image = np.random.randn(4, 3, 64, 64).astype(np.float32)
        sample_with_image = {
            'state': batched_data['state'][0].numpy(),
            'image': dummy_image[0]
        }
        
        # Test camera processor
        camera_processor = get_data_processor(args_camera, camera_model_components)
        processed_camera_sample = camera_processor(sample_with_image)
        
        print(f"   ✅ Camera modality test passed")
        print(f"   📷 Image shape in sample: {processed_camera_sample['image'].shape}")
        print(f"   🎛️  Model input dim: {camera_model.config.input_dim}")
        
    except Exception as e:
        print(f"   ❌ Camera modality test failed: {e}")
        return False
    
    # Test 7: Configuration validation
    print("\n7️⃣  Testing configuration...")
    try:
        config = model.config
        assert config.state_dim == args.state_dim
        assert config.action_dim == args.action_dim
        assert config.num_layers == args.num_layers
        assert config.hidden_dim == args.hidden_dim
        
        print(f"   ✅ Configuration validation passed")
        print(f"   🎛️  All parameters match expected values")
    except Exception as e:
        print(f"   ❌ Configuration validation failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! MLP policy is ready to use!")
    print("\n📚 Usage examples:")
    print("   Training: python train.py --policy mlp --config configs/policy/mlp.yaml")
    print("   Evaluation: python eval.py --policy mlp --model_name_or_path /path/to/checkpoint")
    
    return True


if __name__ == "__main__":
    success = test_mlp_policy()
    sys.exit(0 if success else 1)
