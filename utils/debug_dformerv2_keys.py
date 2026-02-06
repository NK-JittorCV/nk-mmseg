#!/usr/bin/env python3
"""
Debug script specifically for DFormerv2 LayerScale parameters
"""

import os
import sys
import torch

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dformerv2_pytorch():
    """Check DFormerv2 PyTorch weights."""
    print("=== Checking DFormerv2 PyTorch Weights ===")
    
    pytorch_files = [
        'checkpoints/trained/DFormerv2_Large_NYU.pth',
        'checkpoints/pretrained/DFormerv2_Large_pretrained.pth'
    ]
    
    for file_path in pytorch_files:
        if os.path.exists(file_path):
            print(f"\n--- Checking {file_path} ---")
            try:
                checkpoint = torch.load(file_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                print(f"Total parameters: {len(state_dict)}")
                
                # Check for LayerScale parameters (gamma_1, gamma_2)
                layerscale_params = []
                for key in sorted(state_dict.keys()):
                    if 'gamma_1' in key or 'gamma_2' in key:
                        layerscale_params.append(key)
                
                if layerscale_params:
                    print(f"LayerScale parameters found ({len(layerscale_params)}):")
                    for param in layerscale_params:
                        print(f"  {param}: {state_dict[param].shape}")
                else:
                    print("No LayerScale parameters found")
                
                # Check for any gamma parameters
                gamma_params = []
                for key in sorted(state_dict.keys()):
                    if 'gamma' in key:
                        gamma_params.append(key)
                
                if gamma_params:
                    print(f"All gamma parameters found ({len(gamma_params)}):")
                    for param in gamma_params[:20]:  # Show first 20
                        print(f"  {param}: {state_dict[param].shape}")
                    if len(gamma_params) > 20:
                        print(f"  ... and {len(gamma_params) - 20} more")
                
                return state_dict
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    return None

def check_dformerv2_jittor():
    """Check DFormerv2 Jittor model."""
    print("\n=== Checking DFormerv2 Jittor Model ===")
    
    try:
        from importlib import import_module
        from models import build_model
        
        # Check DFormerv2 Large
        try:
            config = getattr(import_module('local_configs.NYUDepthv2.DFormerv2_L'), 'C')
            model = build_model(config)
            state_dict = model.state_dict()
            
            print(f"DFormerv2 Large - Total parameters: {len(state_dict)}")
            
            # Check for LayerScale parameters (gamma1, gamma2)
            layerscale_params = []
            for key in sorted(state_dict.keys()):
                if 'gamma1' in key or 'gamma2' in key:
                    layerscale_params.append(key)
            
            if layerscale_params:
                print(f"LayerScale parameters found ({len(layerscale_params)}):")
                for param in layerscale_params:
                    print(f"  {param}: {state_dict[param].shape}")
            else:
                print("No LayerScale parameters found")
            
            # Check for any gamma parameters
            gamma_params = []
            for key in sorted(state_dict.keys()):
                if 'gamma' in key:
                    gamma_params.append(key)
            
            if gamma_params:
                print(f"All gamma parameters found ({len(gamma_params)}):")
                for param in gamma_params[:20]:  # Show first 20
                    print(f"  {param}: {state_dict[param].shape}")
                if len(gamma_params) > 20:
                    print(f"  ... and {len(gamma_params) - 20} more")
            
            return state_dict
            
        except Exception as e:
            print(f"Error with DFormerv2 Large: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error importing Jittor modules: {e}")
    
    return None

def analyze_layerscale_mapping(pytorch_dict, jittor_dict):
    """Analyze LayerScale parameter mapping."""
    if pytorch_dict is None or jittor_dict is None:
        print("Cannot analyze - missing model dictionaries")
        return
    
    print("\n=== LayerScale Parameter Mapping Analysis ===")
    
    # Find all gamma parameters in both models
    pytorch_gamma = {}
    jittor_gamma = {}
    
    for key in pytorch_dict.keys():
        if 'gamma' in key:
            pytorch_gamma[key] = pytorch_dict[key].shape
    
    for key in jittor_dict.keys():
        if 'gamma' in key:
            jittor_gamma[key] = jittor_dict[key].shape
    
    print(f"PyTorch gamma parameters: {len(pytorch_gamma)}")
    print(f"Jittor gamma parameters: {len(jittor_gamma)}")
    
    if pytorch_gamma:
        print("\nPyTorch gamma parameters:")
        for key, shape in sorted(pytorch_gamma.items())[:10]:
            print(f"  {key}: {shape}")
        if len(pytorch_gamma) > 10:
            print(f"  ... and {len(pytorch_gamma) - 10} more")
    
    if jittor_gamma:
        print("\nJittor gamma parameters:")
        for key, shape in sorted(jittor_gamma.items())[:10]:
            print(f"  {key}: {shape}")
        if len(jittor_gamma) > 10:
            print(f"  ... and {len(jittor_gamma) - 10} more")
    
    # Try to find mapping patterns
    print("\n=== Potential Mapping Patterns ===")
    
    # Check gamma_1 -> gamma1 mapping
    gamma_1_keys = [k for k in pytorch_gamma.keys() if 'gamma_1' in k]
    gamma1_keys = [k for k in jittor_gamma.keys() if 'gamma1' in k]
    
    if gamma_1_keys and gamma1_keys:
        print(f"gamma_1 -> gamma1 mapping:")
        print(f"  PyTorch gamma_1 keys: {len(gamma_1_keys)}")
        print(f"  Jittor gamma1 keys: {len(gamma1_keys)}")
        
        # Show some examples
        for i in range(min(5, len(gamma_1_keys), len(gamma1_keys))):
            pytorch_key = gamma_1_keys[i]
            expected_jittor_key = pytorch_key.replace('gamma_1', 'gamma1')
            if expected_jittor_key in jittor_gamma:
                print(f"  {pytorch_key} -> {expected_jittor_key} ✓")
            else:
                print(f"  {pytorch_key} -> {expected_jittor_key} ✗ (not found)")
    
    # Check gamma_2 -> gamma2 mapping
    gamma_2_keys = [k for k in pytorch_gamma.keys() if 'gamma_2' in k]
    gamma2_keys = [k for k in jittor_gamma.keys() if 'gamma2' in k]
    
    if gamma_2_keys and gamma2_keys:
        print(f"\ngamma_2 -> gamma2 mapping:")
        print(f"  PyTorch gamma_2 keys: {len(gamma_2_keys)}")
        print(f"  Jittor gamma2 keys: {len(gamma2_keys)}")
        
        # Show some examples
        for i in range(min(5, len(gamma_2_keys), len(gamma2_keys))):
            pytorch_key = gamma_2_keys[i]
            expected_jittor_key = pytorch_key.replace('gamma_2', 'gamma2')
            if expected_jittor_key in jittor_gamma:
                print(f"  {pytorch_key} -> {expected_jittor_key} ✓")
            else:
                print(f"  {pytorch_key} -> {expected_jittor_key} ✗ (not found)")

if __name__ == '__main__':
    pytorch_dict = check_dformerv2_pytorch()
    jittor_dict = check_dformerv2_jittor()
    analyze_layerscale_mapping(pytorch_dict, jittor_dict)
