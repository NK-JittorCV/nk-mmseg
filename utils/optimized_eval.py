#!/usr/bin/env python3
"""
Optimized evaluation script to achieve PyTorch-level performance
"""

import os
import sys
import time
import argparse
import numpy as np
import jittor as jt
from jittor import nn

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module
from utils.dataloader.dataloader import get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.val_mm import evaluate_msf
from utils.jt_utils import load_model
from utils.engine.engine import Engine
from models import build_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimized DFormer Evaluation')
    parser.add_argument('--config', default='local_configs.NYUDepthv2.DFormer_Large', help='config file path')
    parser.add_argument('--checkpoint', default='checkpoints/trained/NYUv2_DFormer_Large.pth', help='checkpoint file path')
    parser.add_argument('--scales', nargs='+', type=float, default=[0.5, 0.75, 1.0, 1.25, 1.5], help='evaluation scales')
    parser.add_argument('--flip', action='store_true', default=True, help='use flip augmentation')
    parser.add_argument('--sliding', action='store_true', default=True, help='use sliding window inference')
    
    return parser.parse_args()

def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=== Optimized DFormer Evaluation ===")
    
    # Load config
    config = getattr(import_module(args.config), "C")
    
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
    
    print(f"Dataset: {config.dataset_name}")
    print(f"Number of classes: {val_loader.dataset.num_classes}")
    print(f"Total samples: {len(val_loader.dataset)}")
    
    # Create model
    model = build_model(config)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model = load_model(model, args.checkpoint)
    else:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    model.eval()
    
    print(f"Starting optimized evaluation with:")
    print(f"  Scales: {args.scales}")
    print(f"  Flip augmentation: {args.flip}")
    print(f"  Sliding window: {args.sliding}")
    
    # Run optimized evaluation
    start_time = time.time()
    
    metric = evaluate_msf(
        model, val_loader, config, 
        scales=args.scales, 
        flip=args.flip, 
        sliding=args.sliding
    )
    
    end_time = time.time()
    
    # Get results
    results = metric.get_results()
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZED EVALUATION RESULTS")
    print("="*60)
    print(f"mIoU: {results['mIoU']:.4f}")
    print(f"mAcc: {results['mAcc']:.4f}")
    print(f"Overall Acc: {results['Overall_Acc']:.4f}")
    print(f"FWIoU: {results['FWIoU']:.4f}")
    print(f"Evaluation time: {end_time - start_time:.2f}s")
    
    # Print per-class IoU
    print(f"\nPer-class IoU:")
    for i in range(min(40, len(results['IoU_per_class']))):
        print(f"Class {i:2d}: {results['IoU_per_class'][i]:.4f}")
    
    # Performance analysis
    target_miou = 0.584  # PyTorch baseline
    current_miou = results['mIoU']
    gap = target_miou - current_miou
    
    print(f"\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"Target mIoU (PyTorch baseline): {target_miou:.4f}")
    print(f"Current mIoU (Jittor optimized): {current_miou:.4f}")
    print(f"Performance gap: {gap:.4f} ({gap/target_miou*100:.2f}%)")
    
    if gap < 0.01:
        print("🎉 SUCCESS: Performance gap < 1%! Target achieved!")
    elif gap < 0.02:
        print("✅ GOOD: Performance gap < 2%. Very close to target!")
    elif gap < 0.03:
        print("⚠️  CLOSE: Performance gap < 3%. Need minor optimizations.")
    else:
        print("❌ GAP: Significant performance gap. Need further optimization.")

if __name__ == '__main__':
    main()
