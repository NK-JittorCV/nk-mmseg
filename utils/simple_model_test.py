#!/usr/bin/env python3
"""
Simple model loading and inference test for all DFormer variants
Tests if models can load weights and perform forward pass without full evaluation
"""

import os
import sys
import time
import traceback

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_loading(config_path, checkpoint_path, model_name):
    """Test if a model can load weights and perform forward pass."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Import required modules
        from importlib import import_module
        from models import build_model
        from utils.jt_utils import load_model
        import jittor as jt
        
        # Load config
        config = getattr(import_module(config_path), 'C')
        
        # Build model
        print("🔄 Building model...")
        model = build_model(config)
        
        # Load weights
        print("🔄 Loading weights...")
        model = load_model(model, checkpoint_path)
        
        model.eval()
        
        # Test forward pass
        print("🔄 Testing forward pass...")
        batch_size = 1
        height, width = 480, 640
        
        # Create dummy inputs
        rgb_input = jt.randn(batch_size, 3, height, width)
        
        # Check if this is DFormer (needs 3-channel depth) or DFormerv2 (needs 1-channel depth)
        if 'DFormerv2' in model_name:
            depth_input = jt.randn(batch_size, 1, height, width)
        else:
            depth_input = jt.randn(batch_size, 3, height, width)  # DFormer uses 3-channel depth
        
        with jt.no_grad():
            start_time = time.time()
            output = model(rgb_input, depth_input)
            end_time = time.time()
        
        # Handle different output formats
        if isinstance(output, (tuple, list)):
            output = output[0]
            if isinstance(output, list):
                output = output[0]
        
        inference_time = (end_time - start_time) * 1000
        
        print(f"✅ {model_name} test successful!")
        print(f"   Input shapes: RGB {rgb_input.shape}, Depth {depth_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Inference time: {inference_time:.2f}ms")
        print(f"   Output range: [{output.min():.6f}, {output.max():.6f}]")
        
        # Check if output has reasonable number of classes
        expected_classes = 40  # NYUDepthv2 and SUN-RGBD both have 40 classes
        if output.shape[1] == expected_classes:
            print(f"   ✅ Correct number of classes: {expected_classes}")
        else:
            print(f"   ⚠️  Unexpected number of classes: {output.shape[1]} (expected {expected_classes})")
        
        return True
        
    except Exception as e:
        print(f"❌ {model_name} test failed: {e}")
        print("Error details:")
        traceback.print_exc()
        return False

def main():
    """Test all available models."""
    print("Starting simple model loading and inference tests...")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Model configurations to test
    models_to_test = [
        # DFormerv2 series (priority)
        {
            'name': 'DFormerv2-L',
            'nyu_config': 'local_configs.NYUDepthv2.DFormerv2_L',
            'nyu_checkpoint': 'checkpoints/trained/DFormerv2_Large_NYU.pth',
            'sunrgbd_config': 'local_configs.SUNRGBD.DFormerv2_L',
            'sunrgbd_checkpoint': 'checkpoints/trained/DFormerv2_Large_SUNRGBD.pth'
        },
        {
            'name': 'DFormerv2-B',
            'nyu_config': 'local_configs.NYUDepthv2.DFormerv2_B',
            'nyu_checkpoint': 'checkpoints/trained/DFormerv2_Base_NYU.pth',
            'sunrgbd_config': 'local_configs.SUNRGBD.DFormerv2_B',
            'sunrgbd_checkpoint': 'checkpoints/trained/DFormerv2_Base_SUNRGBD.pth'
        },
        {
            'name': 'DFormerv2-S',
            'nyu_config': 'local_configs.NYUDepthv2.DFormerv2_S',
            'nyu_checkpoint': 'checkpoints/trained/DFormerv2_Small_NYU.pth',
            'sunrgbd_config': 'local_configs.SUNRGBD.DFormerv2_S',
            'sunrgbd_checkpoint': 'checkpoints/trained/DFormerv2_Small_SUNRGBD.pth'
        },
        # DFormer series
        {
            'name': 'DFormer-L',
            'nyu_config': 'local_configs.NYUDepthv2.DFormer_Large',
            'nyu_checkpoint': 'checkpoints/trained/NYUv2_DFormer_Large.pth',
            'sunrgbd_config': 'local_configs.SUNRGBD.DFormer_Large',
            'sunrgbd_checkpoint': 'checkpoints/trained/SUNRGBD_DFormer_Large.pth'
        },
        {
            'name': 'DFormer-B',
            'nyu_config': 'local_configs.NYUDepthv2.DFormer_Base',
            'nyu_checkpoint': 'checkpoints/trained/NYUv2_DFormer_Base.pth',
            'sunrgbd_config': 'local_configs.SUNRGBD.DFormer_Base',
            'sunrgbd_checkpoint': 'checkpoints/trained/SUNRGBD_DFormer_Base.pth'
        },
        {
            'name': 'DFormer-S',
            'nyu_config': 'local_configs.NYUDepthv2.DFormer_Small',
            'nyu_checkpoint': 'checkpoints/trained/NYUv2_DFormer_Small.pth',
            'sunrgbd_config': 'local_configs.SUNRGBD.DFormer_Small',
            'sunrgbd_checkpoint': 'checkpoints/trained/SUNRGBD_DFormer_Small.pth'
        },
        {
            'name': 'DFormer-T',
            'nyu_config': 'local_configs.NYUDepthv2.DFormer_Tiny',
            'nyu_checkpoint': 'checkpoints/trained/NYUv2_DFormer_Tiny.pth',
            'sunrgbd_config': 'local_configs.SUNRGBD.DFormer_Tiny',
            'sunrgbd_checkpoint': 'checkpoints/trained/SUNRGBD_DFormer_Tiny.pth'
        }
    ]
    
    results = {}
    
    for model_info in models_to_test:
        model_name = model_info['name']
        results[model_name] = {}
        
        # Test NYUDepthv2 version
        print(f"\n🔄 Testing {model_name} (NYUDepthv2 config)...")
        nyu_success = test_model_loading(
            model_info['nyu_config'],
            model_info['nyu_checkpoint'],
            f"{model_name}-NYU"
        )
        results[model_name]['nyu'] = nyu_success
        
        # Test SUN-RGBD version
        print(f"\n🔄 Testing {model_name} (SUN-RGBD config)...")
        sunrgbd_success = test_model_loading(
            model_info['sunrgbd_config'],
            model_info['sunrgbd_checkpoint'],
            f"{model_name}-SUNRGBD"
        )
        results[model_name]['sunrgbd'] = sunrgbd_success
    
    # Print summary
    print(f"\n{'='*80}")
    print("MODEL LOADING TEST SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Model':<15} {'NYUDepthv2':<12} {'SUN-RGBD':<12} {'Overall':<10}")
    print("-" * 50)
    
    for model_info in models_to_test:
        model_name = model_info['name']
        if model_name in results:
            nyu_status = "✅ PASS" if results[model_name]['nyu'] else "❌ FAIL"
            sunrgbd_status = "✅ PASS" if results[model_name]['sunrgbd'] else "❌ FAIL"
            overall_status = "✅ PASS" if (results[model_name]['nyu'] and results[model_name]['sunrgbd']) else "❌ FAIL"
            
            print(f"{model_name:<15} {nyu_status:<12} {sunrgbd_status:<12} {overall_status:<10}")
    
    print(f"{'='*80}")
    
    # Count successful models
    total_tests = len(models_to_test) * 2  # 2 datasets per model
    successful_tests = sum([
        sum([results[model]['nyu'], results[model]['sunrgbd']]) 
        for model in results
    ])
    
    print(f"Summary: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("🎉 All models loaded successfully! Ready for full evaluation.")
    else:
        print("⚠️  Some models failed to load. Check the error messages above.")
    
    return results

if __name__ == '__main__':
    main()
