#!/usr/bin/env python3
"""
Layer-by-layer comparison between PyTorch and Jittor versions
"""

import os
import sys
import numpy as np
import jittor as jt
from jittor import nn

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def compare_models():
    """Compare PyTorch and Jittor models layer by layer."""
    print("=== Layer-by-Layer Model Comparison ===")
    
    # Load Jittor config and model
    from importlib import import_module
    config = getattr(import_module('local_configs.NYUDepthv2.DFormer_Large'), "C")
    
    # Set device
    jt.flags.use_cuda = 1
    
    # Create Jittor model
    from models import build_model
    from utils.jt_utils import load_model
    
    jt_model = build_model(config)
    jt_model = load_model(jt_model, "checkpoints/trained/NYUv2_DFormer_Large.pth")
    jt_model.eval()
    
    # Load PyTorch model for comparison
    sys.path.insert(0, '../DFormer')
    try:
        import torch
        torch_config = getattr(import_module('local_configs.NYUDepthv2.DFormer_Large'), "C")
        
        # Import PyTorch model
        from models import build_model as torch_build_model
        torch_model = torch_build_model(torch_config)
        
        # Load PyTorch weights
        checkpoint = torch.load("checkpoints/trained/NYUv2_DFormer_Large.pth", map_location='cpu')
        torch_model.load_state_dict(checkpoint['model'])
        torch_model.eval()
        torch_model.cuda()
        
        print("✓ Both models loaded successfully")
        
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        print("Will proceed with Jittor-only analysis")
        torch_model = None
    
    # Create test inputs
    print("\n=== Creating Test Inputs ===")
    np.random.seed(42)  # For reproducibility
    
    # Use fixed test data
    test_rgb_np = np.random.randn(1, 3, 480, 640).astype(np.float32)
    test_depth_np = np.random.randn(1, 3, 480, 640).astype(np.float32)
    
    # Jittor inputs
    jt_rgb = jt.array(test_rgb_np)
    jt_depth = jt.array(test_depth_np)
    
    print(f"Test input shapes: RGB {jt_rgb.shape}, Depth {jt_depth.shape}")
    
    # PyTorch inputs (if available)
    if torch_model is not None:
        torch_rgb = torch.from_numpy(test_rgb_np).cuda()
        torch_depth = torch.from_numpy(test_depth_np).cuda()
    
    # Forward pass comparison
    print("\n=== Forward Pass Comparison ===")
    
    with jt.no_grad():
        # Jittor forward pass
        jt_output = jt_model(jt_rgb, jt_depth)
        if isinstance(jt_output, (list, tuple)) and len(jt_output) == 2:
            jt_output = jt_output[0]
            if isinstance(jt_output, list):
                jt_output = jt_output[0]
        
        jt_output_np = jt_output.numpy()
        
        print(f"Jittor output shape: {jt_output.shape}")
        print(f"Jittor output range: [{jt_output_np.min():.6f}, {jt_output_np.max():.6f}]")
        print(f"Jittor output mean: {jt_output_np.mean():.6f}, std: {jt_output_np.std():.6f}")
        
        if torch_model is not None:
            with torch.no_grad():
                torch_output = torch_model(torch_rgb, torch_depth)
                if isinstance(torch_output, (list, tuple)) and len(torch_output) == 2:
                    torch_output = torch_output[0]
                    if isinstance(torch_output, list):
                        torch_output = torch_output[0]
                
                torch_output_np = torch_output.cpu().numpy()
                
                print(f"PyTorch output shape: {torch_output.shape}")
                print(f"PyTorch output range: [{torch_output_np.min():.6f}, {torch_output_np.max():.6f}]")
                print(f"PyTorch output mean: {torch_output_np.mean():.6f}, std: {torch_output_np.std():.6f}")
                
                # Compare outputs
                if jt_output_np.shape == torch_output_np.shape:
                    diff = np.abs(jt_output_np - torch_output_np)
                    max_diff = diff.max()
                    mean_diff = diff.mean()
                    
                    print(f"\nOutput Comparison:")
                    print(f"  Max difference: {max_diff:.2e}")
                    print(f"  Mean difference: {mean_diff:.2e}")
                    print(f"  Relative max diff: {max_diff / (np.abs(torch_output_np).max() + 1e-8):.2e}")
                    
                    if max_diff > 1e-3:
                        print("  ⚠️  Large output difference detected!")
                    elif max_diff > 1e-6:
                        print("  ⚠️  Small but noticeable difference")
                    else:
                        print("  ✓ Outputs are very close")
                else:
                    print(f"  ✗ Shape mismatch: Jittor {jt_output_np.shape} vs PyTorch {torch_output_np.shape}")
    
    # Detailed parameter comparison
    print("\n=== Parameter Comparison ===")
    
    jt_state_dict = jt_model.state_dict()
    
    if torch_model is not None:
        torch_state_dict = torch_model.state_dict()
        
        # Compare common parameters
        common_keys = set(jt_state_dict.keys()) & set(torch_state_dict.keys())
        mapped_keys = {
            'decode_head.conv_seg.weight': 'decode_head.cls_seg.weight',
            'decode_head.conv_seg.bias': 'decode_head.cls_seg.bias'
        }
        
        print(f"Common parameters: {len(common_keys)}")
        
        large_diffs = []
        
        for key in sorted(common_keys):
            if 'num_batches_tracked' in key:
                continue
                
            jt_param = jt_state_dict[key].numpy()
            torch_param = torch_state_dict[key].detach().cpu().numpy()
            
            if jt_param.shape != torch_param.shape:
                print(f"  ✗ {key}: Shape mismatch {jt_param.shape} vs {torch_param.shape}")
                continue
            
            diff = np.abs(jt_param - torch_param)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            if max_diff > 1e-6:
                large_diffs.append((key, max_diff, mean_diff))
        
        # Check mapped parameters
        for torch_key, jt_key in mapped_keys.items():
            if torch_key in torch_state_dict and jt_key in jt_state_dict:
                jt_param = jt_state_dict[jt_key].numpy()
                torch_param = torch_state_dict[torch_key].detach().cpu().numpy()
                
                diff = np.abs(jt_param - torch_param)
                max_diff = diff.max()
                mean_diff = diff.mean()
                
                if max_diff > 1e-6:
                    large_diffs.append((f"{torch_key}->{jt_key}", max_diff, mean_diff))
        
        print(f"\nParameters with differences > 1e-6: {len(large_diffs)}")
        for key, max_diff, mean_diff in sorted(large_diffs, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: max={max_diff:.2e}, mean={mean_diff:.2e}")
    
    return jt_model, torch_model if 'torch_model' in locals() else None

if __name__ == '__main__':
    compare_models()
