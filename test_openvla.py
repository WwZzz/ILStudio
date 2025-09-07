#!/usr/bin/env python3
"""
Test script for OpenVLA policy integration.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from policy.policy_loader import policy_loader
from configs.task.loader import load_task_config


def test_openvla_config_loading():
    """Test loading OpenVLA configuration."""
    print("Testing OpenVLA configuration loading...")
    
    try:
        # Load policy config
        policy_config = policy_loader.load_policy_config("configs/policy/openvla.yaml")
        print(f"✅ Policy config loaded: {policy_config.name}")
        print(f"   Module path: {policy_config.module_path}")
        print(f"   Config class: {policy_config.config_class}")
        print(f"   Model class: {policy_config.model_class}")
        print(f"   Training mode: {policy_config.config_params.get('training_mode', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load OpenVLA config: {e}")
        return False


def test_openvla_model_creation():
    """Test creating OpenVLA model instance."""
    print("\nTesting OpenVLA model creation...")
    
    try:
        # Create mock args
        class MockArgs:
            def __init__(self):
                self.model_name_or_path = "prismatic-vlms/openvla-7b-instruct"
                self.training_mode = "lora"
                self.lora_r = 16
                self.lora_alpha = 32
                self.lora_dropout = 0.1
                self.use_quantization = False
                self.max_length = 2048
                self.state_dim = 14
                self.action_dim = 14
                self.camera_names = ["primary"]
        
        args = MockArgs()
        
        # Load model components
        model_components = policy_loader.load_model("configs/policy/openvla.yaml", args)
        print(f"✅ Model components loaded: {list(model_components.keys())}")
        
        # Check if all required components are present
        required_components = ['model', 'processor', 'action_tokenizer', 'tokenizer']
        for component in required_components:
            if component in model_components:
                print(f"   ✅ {component}: {type(model_components[component])}")
            else:
                print(f"   ❌ Missing {component}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Failed to create OpenVLA model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openvla_data_processing():
    """Test OpenVLA data processing."""
    print("\nTesting OpenVLA data processing...")
    
    try:
        # Create mock args
        class MockArgs:
            def __init__(self):
                self.model_name_or_path = "prismatic-vlms/openvla-7b-instruct"
                self.training_mode = "lora"
                self.lora_r = 16
                self.lora_alpha = 32
                self.lora_dropout = 0.1
                self.use_quantization = False
                self.max_length = 2048
                self.state_dim = 14
                self.action_dim = 14
                self.camera_names = ["primary"]
        
        args = MockArgs()
        
        # Load model components
        model_components = policy_loader.load_model("configs/policy/openvla.yaml", args)
        
        # Get data processor
        data_processor = policy_loader.get_data_processor("configs/policy/openvla.yaml", None, args, model_components)
        
        if data_processor is None:
            print("❌ No data processor found")
            return False
        
        print(f"✅ Data processor loaded: {type(data_processor)}")
        
        # Test data processing with mock data
        mock_sample = {
            'image': torch.randn(1, 3, 224, 224),  # Batch of 1, 3 channels, 224x224
            'action': torch.randn(1, 14),  # Single action of 14 dimensions
            'raw_lang': "pick up the red cube"
        }
        
        processed = data_processor(mock_sample)
        print(f"✅ Data processing successful")
        print(f"   Processed keys: {list(processed.keys())}")
        print(f"   Pixel values shape: {processed['pixel_values'].shape}")
        print(f"   Input IDs shape: {processed['input_ids'].shape}")
        print(f"   Labels shape: {processed['labels'].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test data processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openvla_data_collator():
    """Test OpenVLA data collator."""
    print("\nTesting OpenVLA data collator...")
    
    try:
        # Create mock args
        class MockArgs:
            def __init__(self):
                self.model_name_or_path = "prismatic-vlms/openvla-7b-instruct"
                self.training_mode = "lora"
                self.lora_r = 16
                self.lora_alpha = 32
                self.lora_dropout = 0.1
                self.use_quantization = False
                self.max_length = 2048
                self.state_dim = 14
                self.action_dim = 14
                self.camera_names = ["primary"]
        
        args = MockArgs()
        
        # Load model components
        model_components = policy_loader.load_model("configs/policy/openvla.yaml", args)
        
        # Get data collator
        data_collator = policy_loader.get_data_collator("configs/policy/openvla.yaml", args, model_components)
        
        if data_collator is None:
            print("❌ No data collator found")
            return False
        
        print(f"✅ Data collator loaded: {type(data_collator)}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test data collator: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openvla_trainer_class():
    """Test OpenVLA trainer class."""
    print("\nTesting OpenVLA trainer class...")
    
    try:
        # Get trainer class
        trainer_class = policy_loader.get_trainer_class("configs/policy/openvla.yaml")
        
        if trainer_class is None:
            print("❌ No trainer class found")
            return False
        
        print(f"✅ Trainer class loaded: {trainer_class}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test trainer class: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Testing OpenVLA Policy Integration")
    print("=" * 50)
    
    tests = [
        test_openvla_config_loading,
        test_openvla_model_creation,
        test_openvla_data_processing,
        test_openvla_data_collator,
        test_openvla_trainer_class,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! OpenVLA integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
