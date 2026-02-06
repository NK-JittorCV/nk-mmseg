#!/usr/bin/env python3
"""
Test different evaluation configurations to find optimal settings
"""

import os
import sys
import time
import numpy as np
import jittor as jt

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module
from utils.dataloader.dataloader import get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.val_mm import evaluate, evaluate_msf
from utils.jt_utils import load_model
from utils.engine.engine import Engine
from models import build_model

def test_evaluation_configs():
    """Test different evaluation configurations."""
    print("=== Testing Different Evaluation Configurations ===")
    
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
    
    print(f"Dataset: {config.dataset_name}")
    print(f"Number of classes: {val_loader.dataset.num_classes}")
    print(f"Total samples: {len(val_loader.dataset)}")
    
    # Create model
    model = build_model(config)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/trained/NYUv2_DFormer_Large.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model = load_model(model, checkpoint_path)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return
    
    model.eval()
    
    # Test configurations
    configs_to_test = [
        {
            'name': 'Standard (Single Scale)',
            'scales': [1.0],
            'flip': False,
            'sliding': False
        },
        {
            'name': 'Multi-Scale (3 scales)',
            'scales': [0.75, 1.0, 1.25],
            'flip': False,
            'sliding': False
        },
        {
            'name': 'Multi-Scale + Flip',
            'scales': [0.75, 1.0, 1.25],
            'flip': True,
            'sliding': False
        },
        {
            'name': 'Single Scale + Sliding',
            'scales': [1.0],
            'flip': False,
            'sliding': True
        },
        {
            'name': 'Multi-Scale + Flip + Sliding',
            'scales': [0.75, 1.0, 1.25],
            'flip': True,
            'sliding': True
        },
        {
            'name': 'PyTorch-style (5 scales)',
            'scales': [0.5, 0.75, 1.0, 1.25, 1.5],
            'flip': True,
            'sliding': True
        }
    ]
    
    results = []
    
    for test_config in configs_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {test_config['name']}")
        print(f"Scales: {test_config['scales']}")
        print(f"Flip: {test_config['flip']}")
        print(f"Sliding: {test_config['sliding']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if len(test_config['scales']) > 1 or test_config['flip'] or test_config['sliding']:
                # Advanced evaluation
                metric = evaluate_msf(
                    model, val_loader, 
                    config=config, 
                    scales=test_config['scales'], 
                    flip=test_config['flip'], 
                    sliding=test_config['sliding']
                )
                eval_results = metric.get_results()
            else:
                # Standard evaluation
                eval_results = evaluate(model, val_loader, verbose=True)
            
            end_time = time.time()
            eval_time = end_time - start_time
            
            print(f"\nResults for {test_config['name']}:")
            print(f"  mIoU: {eval_results['mIoU']:.4f}")
            print(f"  mAcc: {eval_results['mAcc']:.4f}")
            print(f"  Overall Acc: {eval_results['Overall_Acc']:.4f}")
            print(f"  FWIoU: {eval_results['FWIoU']:.4f}")
            print(f"  Evaluation time: {eval_time:.2f}s")
            
            results.append({
                'config': test_config['name'],
                'mIoU': eval_results['mIoU'],
                'mAcc': eval_results['mAcc'],
                'overall_acc': eval_results['Overall_Acc'],
                'fwiou': eval_results['FWIoU'],
                'time': eval_time
            })
            
        except Exception as e:
            print(f"Error in {test_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("EVALUATION CONFIGURATION COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Configuration':<30} {'mIoU':<8} {'mAcc':<8} {'Overall':<8} {'FWIoU':<8} {'Time(s)':<8}")
    print(f"{'-'*80}")
    
    for result in results:
        print(f"{result['config']:<30} {result['mIoU']:<8.4f} {result['mAcc']:<8.4f} "
              f"{result['overall_acc']:<8.4f} {result['fwiou']:<8.4f} {result['time']:<8.1f}")
    
    # Find best configuration
    if results:
        best_result = max(results, key=lambda x: x['mIoU'])
        print(f"\nBest configuration: {best_result['config']}")
        print(f"Best mIoU: {best_result['mIoU']:.4f}")
        
        # Performance vs time analysis
        print(f"\nPerformance vs Time Analysis:")
        for result in sorted(results, key=lambda x: x['mIoU'], reverse=True):
            efficiency = result['mIoU'] / (result['time'] / 60)  # mIoU per minute
            print(f"  {result['config']:<30}: {result['mIoU']:.4f} mIoU in {result['time']/60:.1f}min (efficiency: {efficiency:.3f})")

if __name__ == '__main__':
    test_evaluation_configs()
