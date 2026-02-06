#!/usr/bin/env python3
"""
Debug script to analyze PyTorch and Jittor model key differences
"""

import os
import sys
import torch
import pickle

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_pytorch_weights():
    """Check PyTorch model weights."""
    print("=== Checking PyTorch Weights ===")
    
    # Check different model files
    pytorch_files = [
        'checkpoints/trained/NYUv2_DFormer_Large.pth',
        'checkpoints/trained/DFormerv2_Large_NYU.pth',
        'checkpoints/pretrained/DFormer_Large.pth.tar',
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
                    for param in layerscale_params[:10]:  # Show first 10
                        print(f"  {param}: {state_dict[param].shape}")
                    if len(layerscale_params) > 10:
                        print(f"  ... and {len(layerscale_params) - 10} more")
                
                # Check decoder head parameters
                decoder_params = []
                for key in sorted(state_dict.keys()):
                    if any(word in key.lower() for word in ['decode_head', 'head', 'cls_seg', 'conv_seg']):
                        decoder_params.append(key)
                
                if decoder_params:
                    print(f"Decoder head parameters found ({len(decoder_params)}):")
                    for param in decoder_params:
                        print(f"  {param}: {state_dict[param].shape}")
                
                # Check backbone norm parameters
                norm_params = []
                for key in sorted(state_dict.keys()):
                    if 'backbone.norm' in key:
                        norm_params.append(key)
                
                if norm_params:
                    print(f"Backbone norm parameters found ({len(norm_params)}):")
                    for param in norm_params:
                        print(f"  {param}: {state_dict[param].shape}")
                
                return state_dict  # Return the first valid state dict
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    return None

def check_jittor_model():
    """Check Jittor model structure."""
    print("\n=== Checking Jittor Model ===")
    
    try:
        from importlib import import_module
        from models import build_model
        
        # Check DFormer Large
        try:
            config = getattr(import_module('local_configs.NYUDepthv2.DFormer_Large'), 'C')
            model = build_model(config)
            state_dict = model.state_dict()
            
            print(f"DFormer Large - Total parameters: {len(state_dict)}")
            
            # Check for LayerScale parameters (gamma1, gamma2)
            layerscale_params = []
            for key in sorted(state_dict.keys()):
                if 'gamma1' in key or 'gamma2' in key:
                    layerscale_params.append(key)
            
            if layerscale_params:
                print(f"LayerScale parameters found ({len(layerscale_params)}):")
                for param in layerscale_params[:10]:  # Show first 10
                    print(f"  {param}: {state_dict[param].shape}")
                if len(layerscale_params) > 10:
                    print(f"  ... and {len(layerscale_params) - 10} more")
            
            # Check decoder head parameters
            decoder_params = []
            for key in sorted(state_dict.keys()):
                if any(word in key.lower() for word in ['decode_head', 'head', 'cls_seg', 'conv_seg']):
                    decoder_params.append(key)
            
            if decoder_params:
                print(f"Decoder head parameters found ({len(decoder_params)}):")
                for param in decoder_params:
                    print(f"  {param}: {state_dict[param].shape}")
            
            return state_dict
            
        except Exception as e:
            print(f"Error with DFormer Large: {e}")
        
        # Check DFormerv2 Large
        try:
            config = getattr(import_module('local_configs.NYUDepthv2.DFormerv2_L'), 'C')
            model = build_model(config)
            state_dict = model.state_dict()
            
            print(f"\nDFormerv2 Large - Total parameters: {len(state_dict)}")
            
            # Check for LayerScale parameters (gamma1, gamma2)
            layerscale_params = []
            for key in sorted(state_dict.keys()):
                if 'gamma1' in key or 'gamma2' in key:
                    layerscale_params.append(key)
            
            if layerscale_params:
                print(f"LayerScale parameters found ({len(layerscale_params)}):")
                for param in layerscale_params[:10]:  # Show first 10
                    print(f"  {param}: {state_dict[param].shape}")
                if len(layerscale_params) > 10:
                    print(f"  ... and {len(layerscale_params) - 10} more")
            
            return state_dict
            
        except Exception as e:
            print(f"Error with DFormerv2 Large: {e}")
    
    except Exception as e:
        print(f"Error importing Jittor modules: {e}")
    
    return None

def compare_keys(pytorch_dict, jittor_dict):
    """Compare keys between PyTorch and Jittor models."""
    if pytorch_dict is None or jittor_dict is None:
        print("Cannot compare - missing model dictionaries")
        return
    
    print("\n=== Key Comparison ===")
    
    pytorch_keys = set(pytorch_dict.keys())
    jittor_keys = set(jittor_dict.keys())
    
    print(f"PyTorch keys: {len(pytorch_keys)}")
    print(f"Jittor keys: {len(jittor_keys)}")
    
    # Keys only in PyTorch
    pytorch_only = pytorch_keys - jittor_keys
    if pytorch_only:
        print(f"\nKeys only in PyTorch ({len(pytorch_only)}):")
        for key in sorted(pytorch_only)[:20]:  # Show first 20
            print(f"  {key}")
        if len(pytorch_only) > 20:
            print(f"  ... and {len(pytorch_only) - 20} more")
    
    # Keys only in Jittor
    jittor_only = jittor_keys - pytorch_keys
    if jittor_only:
        print(f"\nKeys only in Jittor ({len(jittor_only)}):")
        for key in sorted(jittor_only)[:20]:  # Show first 20
            print(f"  {key}")
        if len(jittor_only) > 20:
            print(f"  ... and {len(jittor_only) - 20} more")
    
    # Common keys
    common_keys = pytorch_keys & jittor_keys
    print(f"\nCommon keys: {len(common_keys)}")
    
    # Check for potential mapping issues
    print("\n=== Potential Key Mapping Issues ===")
    
    # Check gamma_1/gamma_2 vs gamma1/gamma2
    gamma_pytorch = [k for k in pytorch_keys if 'gamma_1' in k or 'gamma_2' in k]
    gamma_jittor = [k for k in jittor_keys if 'gamma1' in k or 'gamma2' in k]
    
    if gamma_pytorch or gamma_jittor:
        print(f"LayerScale gamma parameters:")
        print(f"  PyTorch (gamma_1/gamma_2): {len(gamma_pytorch)}")
        print(f"  Jittor (gamma1/gamma2): {len(gamma_jittor)}")
        
        if gamma_pytorch:
            print("  PyTorch examples:")
            for key in gamma_pytorch[:5]:
                print(f"    {key}")
        
        if gamma_jittor:
            print("  Jittor examples:")
            for key in gamma_jittor[:5]:
                print(f"    {key}")
    
    # Check conv_seg vs cls_seg
    conv_seg_pytorch = [k for k in pytorch_keys if 'conv_seg' in k]
    cls_seg_jittor = [k for k in jittor_keys if 'cls_seg' in k]
    
    if conv_seg_pytorch or cls_seg_jittor:
        print(f"\nDecoder head parameters:")
        print(f"  PyTorch (conv_seg): {len(conv_seg_pytorch)}")
        print(f"  Jittor (cls_seg): {len(cls_seg_jittor)}")
        
        if conv_seg_pytorch:
            print("  PyTorch examples:")
            for key in conv_seg_pytorch:
                print(f"    {key}")
        
        if cls_seg_jittor:
            print("  Jittor examples:")
            for key in cls_seg_jittor:
                print(f"    {key}")

if __name__ == '__main__':
    pytorch_dict = check_pytorch_weights()
    jittor_dict = check_jittor_model()
    compare_keys(pytorch_dict, jittor_dict)
