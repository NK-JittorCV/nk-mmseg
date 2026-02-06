#!/usr/bin/env python3
"""
Final performance test to validate all optimizations
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

def run_final_test():
    """Run final performance test with all optimizations."""
    print("🚀 FINAL PERFORMANCE TEST - DFormer Jittor vs PyTorch")
    print("="*70)
    
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
    
    print(f"📊 Dataset: {config.dataset_name}")
    print(f"📊 Number of classes: {val_loader.dataset.num_classes}")
    print(f"📊 Total samples: {len(val_loader.dataset)}")
    
    # Create model
    model = build_model(config)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/trained/NYUv2_DFormer_Large.pth"
    if os.path.exists(checkpoint_path):
        print(f"✅ Loading checkpoint from {checkpoint_path}")
        model = load_model(model, checkpoint_path)
    else:
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return
    
    model.eval()
    
    # Test configurations
    test_configs = [
        {
            'name': '🔥 FINAL OPTIMIZED (5-scale + flip + sliding)',
            'scales': [0.5, 0.75, 1.0, 1.25, 1.5],
            'flip': True,
            'sliding': True,
            'target': 0.584  # PyTorch baseline
        },
        {
            'name': '⚡ FAST OPTIMIZED (3-scale + flip + sliding)',
            'scales': [0.75, 1.0, 1.25],
            'flip': True,
            'sliding': True,
            'target': 0.575  # Expected slightly lower
        },
        {
            'name': '📊 BASELINE (single scale)',
            'scales': [1.0],
            'flip': False,
            'sliding': False,
            'target': 0.537  # Our previous baseline
        }
    ]
    
    results = []
    
    for test_config in test_configs:
        print(f"\n{test_config['name']}")
        print("-" * 60)
        print(f"Scales: {test_config['scales']}")
        print(f"Flip: {test_config['flip']}")
        print(f"Sliding: {test_config['sliding']}")
        
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
                eval_results = evaluate(model, val_loader, verbose=False)
            
            end_time = time.time()
            eval_time = end_time - start_time
            
            miou = eval_results['mIoU']
            target = test_config['target']
            gap = target - miou
            gap_percent = (gap / target) * 100
            
            print(f"\n📈 RESULTS:")
            print(f"  mIoU: {miou:.4f}")
            print(f"  mAcc: {eval_results['mAcc']:.4f}")
            print(f"  Overall Acc: {eval_results['Overall_Acc']:.4f}")
            print(f"  FWIoU: {eval_results['FWIoU']:.4f}")
            print(f"  Time: {eval_time:.1f}s")
            
            print(f"\n🎯 PERFORMANCE vs TARGET:")
            print(f"  Target: {target:.4f}")
            print(f"  Achieved: {miou:.4f}")
            print(f"  Gap: {gap:.4f} ({gap_percent:+.2f}%)")
            
            if gap < 0.01:
                status = "🎉 EXCELLENT"
            elif gap < 0.02:
                status = "✅ VERY GOOD"
            elif gap < 0.03:
                status = "⚠️  GOOD"
            else:
                status = "❌ NEEDS WORK"
            
            print(f"  Status: {status}")
            
            results.append({
                'name': test_config['name'],
                'miou': miou,
                'target': target,
                'gap': gap,
                'gap_percent': gap_percent,
                'time': eval_time,
                'status': status
            })
            
        except Exception as e:
            print(f"❌ Error in {test_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'='*70}")
    print("🏆 FINAL PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    for result in results:
        print(f"{result['name']}")
        print(f"  mIoU: {result['miou']:.4f} (target: {result['target']:.4f})")
        print(f"  Gap: {result['gap']:+.4f} ({result['gap_percent']:+.2f}%)")
        print(f"  Time: {result['time']:.1f}s")
        print(f"  Status: {result['status']}")
        print()
    
    # Find best result
    if results:
        best_result = max(results, key=lambda x: x['miou'])
        print(f"🥇 BEST PERFORMANCE: {best_result['name']}")
        print(f"   mIoU: {best_result['miou']:.4f}")
        print(f"   Gap from PyTorch: {best_result['gap']:+.4f} ({best_result['gap_percent']:+.2f}%)")
        
        # Final verdict
        if best_result['gap'] < 0.01:
            print(f"\n🎉 SUCCESS! Jittor version achieves PyTorch-level performance!")
            print(f"   Performance gap < 1% - Target achieved!")
        elif best_result['gap'] < 0.02:
            print(f"\n✅ EXCELLENT! Very close to PyTorch performance!")
            print(f"   Performance gap < 2% - Practically equivalent!")
        elif best_result['gap'] < 0.03:
            print(f"\n⚠️  GOOD! Close to PyTorch performance!")
            print(f"   Performance gap < 3% - Minor difference!")
        else:
            print(f"\n❌ NEEDS IMPROVEMENT! Significant gap remains!")
            print(f"   Performance gap > 3% - Further optimization needed!")

if __name__ == '__main__':
    run_final_test()
