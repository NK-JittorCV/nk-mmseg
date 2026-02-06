#!/usr/bin/env python3
"""
Debug script to check data format and model behavior
"""

import os
import sys
import numpy as np
import jittor as jt
from jittor import nn

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module
from utils.dataloader.dataloader import get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from models import build_model

def debug_data_and_model():
    """Debug data format and model behavior."""
    print("=== Debugging Data Format and Model Behavior ===")
    
    # Load config
    config = getattr(import_module('local_configs.NYUDepthv2.DFormer_Large'), "C")
    
    # Set device
    jt.flags.use_cuda = 1
    
    # Create engine
    engine = Engine()
    
    # Create data loader
    val_loader, val_sampler = get_val_loader(
        engine=engine,
        dataset_cls=RGBXDataset,
        config=config,
        val_batch_size=1
    )
    
    print(f"Dataset config:")
    print(f"  x_is_single_channel: {config.x_is_single_channel}")
    print(f"  backbone: {config.backbone}")
    print(f"  norm_mean: {config.norm_mean}")
    print(f"  norm_std: {config.norm_std}")
    
    # Get one sample
    sample = next(iter(val_loader))
    rgb = sample['data']
    depth = sample['modal_x']
    label = sample['label']
    
    print(f"\nData shapes:")
    print(f"  RGB: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"  Depth: {depth.shape}, dtype: {depth.dtype}")
    print(f"  Label: {label.shape}, dtype: {label.dtype}")
    
    print(f"\nData ranges:")
    print(f"  RGB: [{rgb.min():.3f}, {rgb.max():.3f}]")
    print(f"  Depth: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"  Label: [{label.min()}, {label.max()}]")
    
    # Check if depth channels are identical
    if depth.shape[1] == 3:
        ch0 = depth[0, 0, :, :]
        ch1 = depth[0, 1, :, :]
        ch2 = depth[0, 2, :, :]
        
        print(f"\nDepth channel analysis:")
        print(f"  Channel 0 range: [{ch0.min():.3f}, {ch0.max():.3f}]")
        print(f"  Channel 1 range: [{ch1.min():.3f}, {ch1.max():.3f}]")
        print(f"  Channel 2 range: [{ch2.min():.3f}, {ch2.max():.3f}]")
        
        # Check if channels are identical
        diff_01 = jt.abs(ch0 - ch1).max()
        diff_12 = jt.abs(ch1 - ch2).max()
        print(f"  Max difference: ch0-ch1: {diff_01:.6f}, ch1-ch2: {diff_12:.6f}")
        print(f"  Channels identical: ch0==ch1: {diff_01 < 1e-6}, ch1==ch2: {diff_12 < 1e-6}")
    
    # Create model and test forward pass
    print(f"\n=== Model Forward Pass Debug ===")
    model = build_model(config)
    model.eval()
    
    print(f"Model backbone: {config.backbone}")
    
    # Test forward pass step by step
    with jt.no_grad():
        print(f"\nInput to model:")
        print(f"  RGB shape: {rgb.shape}")
        print(f"  Depth shape: {depth.shape}")
        
        # Test backbone forward
        try:
            features = model.backbone(rgb, depth)
            if isinstance(features, tuple) and len(features) == 2:
                features = features[0]
            
            print(f"\nBackbone output:")
            print(f"  Number of feature maps: {len(features)}")
            for i, feat in enumerate(features):
                print(f"  Feature {i}: {feat.shape}")
            
            # Test full model forward
            outputs = model(rgb, depth)
            if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
                outputs = outputs[0]
                if isinstance(outputs, list):
                    outputs = outputs[0]
            
            print(f"\nModel output:")
            print(f"  Output shape: {outputs.shape}")
            print(f"  Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
            
            # Check predictions
            predictions = jt.argmax(outputs, dim=1)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            print(f"\nPredictions:")
            print(f"  Prediction shape: {predictions.shape}")
            unique_preds = np.unique(predictions.numpy())
            print(f"  Unique predictions: {unique_preds[:20]}...")  # Show first 20
            print(f"  Number of unique predictions: {len(unique_preds)}")
            
            # Compare with labels
            unique_labels = np.unique(label.numpy())
            print(f"  Unique labels: {unique_labels[:20]}...")
            print(f"  Number of unique labels: {len(unique_labels)}")
            
            # Check if model predicts all classes
            print(f"\nClass coverage:")
            print(f"  Model predicts classes 0-39: {set(range(40)).issubset(set(unique_preds))}")
            print(f"  Max predicted class: {unique_preds.max()}")
            print(f"  Max label class: {unique_labels.max()}")
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    debug_data_and_model()
