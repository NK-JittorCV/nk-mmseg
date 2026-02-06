#!/usr/bin/env python3
"""
Verify weight loading precision between PyTorch and Jittor
"""

import os
import sys
import numpy as np
import jittor as jt

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module
from models import build_model

def verify_weight_precision():
    """Verify weight loading precision."""
    print("=== Verifying Weight Loading Precision ===")
    
    # Load config
    config = getattr(import_module('local_configs.NYUDepthv2.DFormer_Large'), "C")
    
    # Create Jittor model
    jt_model = build_model(config)
    
    # Load PyTorch weights
    import torch
    checkpoint_path = "checkpoints/trained/NYUv2_DFormer_Large.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    pytorch_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in pytorch_checkpoint:
        pytorch_state_dict = pytorch_checkpoint['model']
    else:
        pytorch_state_dict = pytorch_checkpoint
    
    # Get Jittor model state dict before loading
    jt_state_dict_before = jt_model.state_dict()
    
    # Load weights into Jittor model
    from utils.jt_utils import load_model
    jt_model = load_model(jt_model, checkpoint_path)
    jt_state_dict_after = jt_model.state_dict()
    
    print(f"\nWeight comparison:")
    print(f"PyTorch parameters: {len(pytorch_state_dict)}")
    print(f"Jittor parameters: {len(jt_state_dict_after)}")
    
    # Compare weights for common parameters
    common_keys = set(pytorch_state_dict.keys()) & set(jt_state_dict_after.keys())
    print(f"Common parameters: {len(common_keys)}")
    
    max_diff = 0.0
    max_diff_key = ""
    precision_issues = []
    
    for key in sorted(common_keys):
        if 'num_batches_tracked' in key:
            continue
            
        pytorch_param = pytorch_state_dict[key].detach().cpu().numpy()
        jittor_param = jt_state_dict_after[key].numpy()
        
        if pytorch_param.shape != jittor_param.shape:
            print(f"Shape mismatch for {key}: PyTorch {pytorch_param.shape} vs Jittor {jittor_param.shape}")
            continue
        
        diff = np.abs(pytorch_param - jittor_param)
        max_param_diff = diff.max()
        mean_param_diff = diff.mean()
        
        if max_param_diff > max_diff:
            max_diff = max_param_diff
            max_diff_key = key
        
        if max_param_diff > 1e-5:
            precision_issues.append((key, max_param_diff, mean_param_diff))
        
        if len(precision_issues) < 10:  # Only print first 10
            print(f"  {key}: max_diff={max_param_diff:.2e}, mean_diff={mean_param_diff:.2e}")
    
    print(f"\nOverall precision analysis:")
    print(f"Maximum difference: {max_diff:.2e} (in {max_diff_key})")
    print(f"Parameters with precision issues (>1e-5): {len(precision_issues)}")
    
    if precision_issues:
        print(f"\nTop precision issues:")
        for key, max_diff, mean_diff in sorted(precision_issues, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {key}: max={max_diff:.2e}, mean={mean_diff:.2e}")
    
    # Check specific critical layers
    critical_layers = [
        'decode_head.linear_fuse.weight',
        'decode_head.linear_fuse.bias',
        'decode_head.classifier.weight',
        'decode_head.classifier.bias'
    ]
    
    print(f"\nCritical layer analysis:")
    for layer in critical_layers:
        if layer in common_keys:
            pytorch_param = pytorch_state_dict[layer].detach().cpu().numpy()
            jittor_param = jt_state_dict_after[layer].numpy()
            diff = np.abs(pytorch_param - jittor_param).max()
            print(f"  {layer}: max_diff={diff:.2e}")
        else:
            print(f"  {layer}: NOT FOUND")
    
    # Test forward pass precision
    print(f"\n=== Forward Pass Precision Test ===")
    
    # Create test input
    test_rgb = jt.randn(1, 3, 480, 640)
    test_depth = jt.randn(1, 3, 480, 640)
    
    # Jittor forward pass
    jt_model.eval()
    with jt.no_grad():
        jt_output = jt_model(test_rgb, test_depth)
        if isinstance(jt_output, (list, tuple)) and len(jt_output) == 2:
            jt_output = jt_output[0]
            if isinstance(jt_output, list):
                jt_output = jt_output[0]
    
    print(f"Jittor output shape: {jt_output.shape}")
    print(f"Jittor output range: [{jt_output.min():.6f}, {jt_output.max():.6f}]")
    
    # Check output distribution
    jt_output_np = jt_output.numpy()
    print(f"Output statistics:")
    print(f"  Mean: {jt_output_np.mean():.6f}")
    print(f"  Std: {jt_output_np.std():.6f}")
    print(f"  Min: {jt_output_np.min():.6f}")
    print(f"  Max: {jt_output_np.max():.6f}")
    
    # Check if output is reasonable
    if np.abs(jt_output_np.mean()) > 10 or jt_output_np.std() > 100:
        print("WARNING: Output statistics seem unreasonable!")
    
    return max_diff < 1e-4  # Return True if precision is good

if __name__ == '__main__':
    verify_weight_precision()
