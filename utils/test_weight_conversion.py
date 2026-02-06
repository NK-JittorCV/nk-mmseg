#!/usr/bin/env python3
"""
Test script to verify the fixed weight conversion
"""

import os
import sys
import torch
import numpy as np

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dformer_conversion():
    """Test DFormer weight conversion."""
    print("=== Testing DFormer Weight Conversion ===")
    
    try:
        from importlib import import_module
        from models import build_model
        from utils.jt_utils import load_pytorch_weights_basic
        
        # Load Jittor model
        config = getattr(import_module('local_configs.NYUDepthv2.DFormer_Large'), 'C')
        model = build_model(config)
        
        print(f"Jittor model created with {len(model.state_dict())} parameters")
        
        # Test weight loading
        pytorch_checkpoint_path = 'checkpoints/trained/NYUv2_DFormer_Large.pth'
        if os.path.exists(pytorch_checkpoint_path):
            print(f"Loading PyTorch weights from {pytorch_checkpoint_path}")
            model = load_pytorch_weights_basic(model, pytorch_checkpoint_path)
            print("✓ DFormer weight conversion successful!")
        else:
            print(f"⚠ PyTorch checkpoint not found: {pytorch_checkpoint_path}")
            
    except Exception as e:
        print(f"✗ DFormer weight conversion failed: {e}")
        import traceback
        traceback.print_exc()

def test_dformerv2_conversion():
    """Test DFormerv2 weight conversion."""
    print("\n=== Testing DFormerv2 Weight Conversion ===")
    
    try:
        from importlib import import_module
        from models import build_model
        from utils.jt_utils import load_pytorch_weights_basic
        
        # Load Jittor model
        config = getattr(import_module('local_configs.NYUDepthv2.DFormerv2_L'), 'C')
        model = build_model(config)
        
        print(f"Jittor model created with {len(model.state_dict())} parameters")
        
        # Test weight loading
        pytorch_checkpoint_path = 'checkpoints/trained/DFormerv2_Large_NYU.pth'
        if os.path.exists(pytorch_checkpoint_path):
            print(f"Loading PyTorch weights from {pytorch_checkpoint_path}")
            model = load_pytorch_weights_basic(model, pytorch_checkpoint_path)
            print("✓ DFormerv2 weight conversion successful!")
        else:
            print(f"⚠ PyTorch checkpoint not found: {pytorch_checkpoint_path}")
            
    except Exception as e:
        print(f"✗ DFormerv2 weight conversion failed: {e}")
        import traceback
        traceback.print_exc()

def test_weight_precision():
    """Test weight loading precision for DFormerv2."""
    print("\n=== Testing Weight Loading Precision ===")
    
    try:
        from importlib import import_module
        from models import build_model
        import jittor as jt
        
        # Load models
        config = getattr(import_module('local_configs.NYUDepthv2.DFormerv2_L'), 'C')
        model = build_model(config)
        
        # Load PyTorch weights
        pytorch_checkpoint_path = 'checkpoints/trained/DFormerv2_Large_NYU.pth'
        if not os.path.exists(pytorch_checkpoint_path):
            print(f"⚠ PyTorch checkpoint not found: {pytorch_checkpoint_path}")
            return
        
        pytorch_checkpoint = torch.load(pytorch_checkpoint_path, map_location='cpu')
        if 'model' in pytorch_checkpoint:
            pytorch_state_dict = pytorch_checkpoint['model']
        else:
            pytorch_state_dict = pytorch_checkpoint
        
        # Get Jittor model state dict before loading
        jt_state_dict_before = model.state_dict()
        
        # Load weights into Jittor model
        from utils.jt_utils import load_pytorch_weights_basic
        model = load_pytorch_weights_basic(model, pytorch_checkpoint_path)
        jt_state_dict_after = model.state_dict()
        
        # Compare weights for LayerScale parameters
        layerscale_keys = [k for k in jt_state_dict_after.keys() if 'gamma1' in k or 'gamma2' in k]
        
        print(f"Checking precision for {len(layerscale_keys)} LayerScale parameters...")
        
        max_diff = 0.0
        max_diff_key = ""
        precision_issues = []
        
        for jt_key in layerscale_keys[:10]:  # Check first 10 LayerScale parameters
            # Find corresponding PyTorch key
            pytorch_key = jt_key.replace('gamma1', 'gamma_1').replace('gamma2', 'gamma_2')
            
            if pytorch_key in pytorch_state_dict:
                pytorch_param = pytorch_state_dict[pytorch_key].detach().cpu().numpy()
                jittor_param = jt_state_dict_after[jt_key].numpy()
                
                if pytorch_param.shape == jittor_param.shape:
                    diff = np.abs(pytorch_param - jittor_param)
                    max_param_diff = diff.max()
                    mean_param_diff = diff.mean()
                    
                    if max_param_diff > max_diff:
                        max_diff = max_param_diff
                        max_diff_key = jt_key
                    
                    if max_param_diff > 1e-5:
                        precision_issues.append((jt_key, max_param_diff, mean_param_diff))
                    
                    print(f"  {jt_key}: max_diff={max_param_diff:.2e}, mean_diff={mean_param_diff:.2e}")
                else:
                    print(f"  Shape mismatch for {jt_key}: PyTorch {pytorch_param.shape} vs Jittor {jittor_param.shape}")
            else:
                print(f"  PyTorch key not found: {pytorch_key}")
        
        print(f"\nOverall precision:")
        print(f"  Maximum difference: {max_diff:.2e} (at {max_diff_key})")
        print(f"  Precision issues (>1e-5): {len(precision_issues)}")
        
        if max_diff < 1e-4:
            print("✓ Weight loading precision is good!")
        else:
            print("⚠ Weight loading precision may have issues")
            
    except Exception as e:
        print(f"✗ Weight precision test failed: {e}")
        import traceback
        traceback.print_exc()

def test_model_forward():
    """Test model forward pass after weight loading."""
    print("\n=== Testing Model Forward Pass ===")
    
    try:
        from importlib import import_module
        from models import build_model
        from utils.jt_utils import load_pytorch_weights_basic
        import jittor as jt
        
        # Load model
        config = getattr(import_module('local_configs.NYUDepthv2.DFormerv2_L'), 'C')
        model = build_model(config)
        
        # Load weights
        pytorch_checkpoint_path = 'checkpoints/trained/DFormerv2_Large_NYU.pth'
        if os.path.exists(pytorch_checkpoint_path):
            model = load_pytorch_weights_basic(model, pytorch_checkpoint_path)
        
        # Test forward pass
        model.eval()
        
        # Create dummy input
        batch_size = 1
        height, width = 480, 640
        rgb_input = jt.randn(batch_size, 3, height, width)
        depth_input = jt.randn(batch_size, 1, height, width)
        
        print(f"Testing forward pass with input shapes: RGB {rgb_input.shape}, Depth {depth_input.shape}")
        
        with jt.no_grad():
            output = model(rgb_input, depth_input)

        # Handle tuple/list output (some models return tuple or list)
        if isinstance(output, (tuple, list)):
            output = output[0]  # Take the first element (main output)

        # Handle nested list case
        if isinstance(output, list):
            output = output[0]

        print(f"✓ Forward pass successful! Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"  Output mean: {output.mean():.6f}, std: {output.std():.6f}")

        # Check if output is reasonable
        if np.abs(output.mean().numpy()) > 10 or output.std().numpy() > 100:
            print("⚠ Warning: Output statistics seem unreasonable!")
        else:
            print("✓ Output statistics look reasonable")
            
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_dformer_conversion()
    test_dformerv2_conversion()
    test_weight_precision()
    test_model_forward()
