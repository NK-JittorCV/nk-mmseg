#!/usr/bin/env python3
"""
Debug script to check num_classes configuration and model output channels
"""

import os
import sys
import traceback

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_model_classes(config_path, model_name):
    """Debug model class configuration."""
    print(f"\n{'='*60}")
    print(f"Debugging {model_name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")
    
    try:
        # Import required modules
        from importlib import import_module
        from models import build_model
        
        # Load config
        config = getattr(import_module(config_path), 'C')
        
        print(f"Config num_classes: {config.num_classes}")
        print(f"Config dataset_name: {config.dataset_name}")
        print(f"Config backbone: {config.backbone}")
        print(f"Config decoder: {config.decoder}")
        print(f"Config decoder_embed_dim: {config.decoder_embed_dim}")
        
        # Build model
        print("\n🔄 Building model...")
        model = build_model(config)
        
        # Check decode_head cls_seg layer
        if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'cls_seg'):
            cls_seg_layer = model.decode_head.cls_seg
            print(f"decode_head.cls_seg: {cls_seg_layer}")
            print(f"decode_head.cls_seg.out_channels: {cls_seg_layer.out_channels}")
            print(f"decode_head.cls_seg.in_channels: {cls_seg_layer.in_channels}")
        else:
            print("❌ No decode_head.cls_seg found")
        
        # Check if there's a num_classes attribute
        if hasattr(model.decode_head, 'num_classes'):
            print(f"decode_head.num_classes: {model.decode_head.num_classes}")
        
        # Check model state dict for cls_seg parameters
        state_dict = model.state_dict()
        cls_seg_keys = [k for k in state_dict.keys() if 'cls_seg' in k]
        print(f"\nCls_seg parameters in state_dict:")
        for key in cls_seg_keys:
            param = state_dict[key]
            print(f"  {key}: {param.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Debug all model configurations."""
    print("Debugging num_classes configuration for all models...")
    
    # Model configurations to debug
    models_to_debug = [
        ('DFormerv2-L-NYU', 'local_configs.NYUDepthv2.DFormerv2_L'),
        ('DFormerv2-B-NYU', 'local_configs.NYUDepthv2.DFormerv2_B'),
        ('DFormerv2-S-NYU', 'local_configs.NYUDepthv2.DFormerv2_S'),
        ('DFormer-L-NYU', 'local_configs.NYUDepthv2.DFormer_Large'),
        ('DFormer-B-NYU', 'local_configs.NYUDepthv2.DFormer_Base'),
        ('DFormer-S-NYU', 'local_configs.NYUDepthv2.DFormer_Small'),
        ('DFormer-T-NYU', 'local_configs.NYUDepthv2.DFormer_Tiny'),
        ('DFormerv2-B-SUNRGBD', 'local_configs.SUNRGBD.DFormerv2_B'),
        ('DFormerv2-S-SUNRGBD', 'local_configs.SUNRGBD.DFormerv2_S'),
        ('DFormer-L-SUNRGBD', 'local_configs.SUNRGBD.DFormer_Large'),
    ]
    
    results = {}
    
    for model_name, config_path in models_to_debug:
        success = debug_model_classes(config_path, model_name)
        results[model_name] = success
    
    # Summary
    print(f"\n{'='*80}")
    print("DEBUG SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Model':<25} {'Status':<10}")
    print("-" * 40)
    
    for model_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_name:<25} {status:<10}")
    
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
