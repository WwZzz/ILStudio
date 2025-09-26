#!/usr/bin/env python3
"""
Test script to validate MLP policy configuration files.
"""

import yaml
import os
from pathlib import Path

def test_config_format(config_path):
    """Test if the configuration file has the correct format."""
    print(f"🔍 Testing config: {config_path}")
    
    # Required top-level fields
    required_fields = ['name', 'module_path', 'pretrained_config', 'model_args']
    optional_fields = ['config_class', 'model_class', 'data_processor', 'data_collator', 'trainer_class']
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                print(f"   ❌ Missing required field: {field}")
                return False
            else:
                print(f"   ✅ Found required field: {field}")
        
        # Check optional fields
        for field in optional_fields:
            if field in config:
                print(f"   ✅ Found optional field: {field}")
        
        # Check pretrained_config structure
        if 'model_name_or_path' not in config['pretrained_config']:
            print(f"   ❌ Missing 'model_name_or_path' in pretrained_config")
            return False
        
        if 'is_pretrained' not in config['pretrained_config']:
            print(f"   ❌ Missing 'is_pretrained' in pretrained_config")
            return False
        
        # Check model_args structure for MLP-specific fields
        model_args = config['model_args']
        mlp_required_fields = ['state_dim', 'action_dim', 'num_layers', 'hidden_dim']
        
        for field in mlp_required_fields:
            if field not in model_args:
                print(f"   ❌ Missing MLP field in model_args: {field}")
                return False
            else:
                print(f"   ✅ Found MLP field: {field}")
        
        print(f"   🎉 Config format validation passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading config: {e}")
        return False

def main():
    """Test all MLP configuration files."""
    print("🧪 Testing MLP Configuration Files")
    print("=" * 50)
    
    config_dir = Path("configs/policy")
    mlp_configs = [
        "mlp.yaml",
        "mlp_large.yaml", 
        "mlp_small.yaml",
        "mlp_camera.yaml"
    ]
    
    all_passed = True
    
    for config_file in mlp_configs:
        config_path = config_dir / config_file
        if config_path.exists():
            success = test_config_format(config_path)
            all_passed = all_passed and success
            print()
        else:
            print(f"⚠️  Config file not found: {config_path}")
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All MLP configuration files passed validation!")
        print()
        print("📚 Usage examples:")
        print("   Basic MLP:     python train.py --policy configs/policy/mlp.yaml")
        print("   Large MLP:     python train.py --policy configs/policy/mlp_large.yaml") 
        print("   Small MLP:     python train.py --policy configs/policy/mlp_small.yaml")
        print("   Camera MLP:    python train.py --policy configs/policy/mlp_camera.yaml")
    else:
        print("❌ Some configuration files failed validation!")
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
