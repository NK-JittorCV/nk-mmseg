#!/usr/bin/env python3
"""
Debug script for DFormerv2-Small segmentation fault issue
"""

import os
import sys
import time
import gc
import traceback

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """Test if the model can be loaded without issues."""
    print("=== Testing DFormerv2-Small Model Loading ===")
    
    try:
        from importlib import import_module
        from models import build_model
        from utils.jt_utils import load_model
        import jittor as jt
        
        # Load configuration
        config = getattr(import_module('local_configs.NYUDepthv2.DFormerv2_S'), 'C')
        print(f"✓ Configuration loaded successfully")
        print(f"  Backbone: {config.backbone}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Eval crop size: {config.eval_crop_size}")
        
        # Build model
        model = build_model(config)
        print(f"✓ Model built successfully")
        
        # Load weights
        checkpoint_path = 'checkpoints/trained/DFormerv2_Small_NYU.pth'
        if os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}")
            model = load_model(model, checkpoint_path)
            print(f"✓ Weights loaded successfully")
        else:
            print(f"⚠ Checkpoint not found: {checkpoint_path}")
            return False
        
        model.eval()
        print(f"✓ Model set to evaluation mode")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_forward_pass_small():
    """Test forward pass with very small inputs."""
    print("\n=== Testing Forward Pass with Small Inputs ===")
    
    try:
        from importlib import import_module
        from models import build_model
        from utils.jt_utils import load_model
        import jittor as jt
        
        # Load model
        config = getattr(import_module('local_configs.NYUDepthv2.DFormerv2_S'), 'C')
        model = build_model(config)
        model = load_model(model, 'checkpoints/trained/DFormerv2_Small_NYU.pth')
        model.eval()
        
        # Test with very small input
        test_sizes = [
            (1, 3, 240, 320),   # Half size
            (1, 3, 120, 160),   # Quarter size
            (1, 3, 60, 80),     # Eighth size
        ]
        
        for i, (batch_size, channels, height, width) in enumerate(test_sizes):
            print(f"\nTest {i+1}: Input size {height}x{width}")
            
            try:
                rgb_input = jt.randn(batch_size, channels, height, width)
                depth_input = jt.randn(batch_size, 1, height, width)
                
                print(f"  Input created: RGB {rgb_input.shape}, Depth {depth_input.shape}")
                
                with jt.no_grad():
                    start_time = time.time()
                    output = model(rgb_input, depth_input)
                    end_time = time.time()
                
                # Handle output format
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if isinstance(output, list):
                    output = output[0]
                
                print(f"  ✓ Forward pass successful!")
                print(f"  Output shape: {output.shape}")
                print(f"  Inference time: {(end_time - start_time)*1000:.2f}ms")
                
                # Force garbage collection
                del rgb_input, depth_input, output
                gc.collect()
                
            except Exception as e:
                print(f"  ✗ Forward pass failed: {e}")
                traceback.print_exc()
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        traceback.print_exc()
        return False

def test_forward_pass_standard():
    """Test forward pass with standard evaluation size."""
    print("\n=== Testing Forward Pass with Standard Size ===")
    
    try:
        from importlib import import_module
        from models import build_model
        from utils.jt_utils import load_model
        import jittor as jt
        
        # Load model
        config = getattr(import_module('local_configs.NYUDepthv2.DFormerv2_S'), 'C')
        model = build_model(config)
        model = load_model(model, 'checkpoints/trained/DFormerv2_Small_NYU.pth')
        model.eval()
        
        # Test with standard evaluation size
        batch_size, channels, height, width = 1, 3, 480, 640
        
        print(f"Testing with standard size: {height}x{width}")
        
        rgb_input = jt.randn(batch_size, channels, height, width)
        depth_input = jt.randn(batch_size, 1, height, width)
        
        print(f"Input created: RGB {rgb_input.shape}, Depth {depth_input.shape}")
        
        with jt.no_grad():
            start_time = time.time()
            output = model(rgb_input, depth_input)
            end_time = time.time()
        
        # Handle output format
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(output, list):
            output = output[0]
        
        print(f"✓ Standard forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Inference time: {(end_time - start_time)*1000:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"✗ Standard forward pass failed: {e}")
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage patterns."""
    print("\n=== Testing Memory Usage ===")
    
    try:
        import jittor as jt
        
        # Check initial memory
        print(f"Initial memory state:")
        print(f"  Available memory: {jt.flags.device_memory_info}")
        
        # Force garbage collection
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        return False

def test_layer_scale_parameters():
    """Test LayerScale parameter mapping specifically."""
    print("\n=== Testing LayerScale Parameters ===")
    
    try:
        from importlib import import_module
        from models import build_model
        import jittor as jt
        
        # Load model without weights first
        config = getattr(import_module('local_configs.NYUDepthv2.DFormerv2_S'), 'C')
        model = build_model(config)
        
        # Check model parameters
        jittor_params = {}
        for name, param in model.named_parameters():
            jittor_params[name] = param.shape
        
        # Check for LayerScale parameters
        gamma1_params = [name for name in jittor_params.keys() if 'gamma1' in name]
        gamma2_params = [name for name in jittor_params.keys() if 'gamma2' in name]
        
        print(f"Found {len(gamma1_params)} gamma1 parameters")
        print(f"Found {len(gamma2_params)} gamma2 parameters")
        
        if len(gamma1_params) > 0:
            print(f"Sample gamma1 parameter: {gamma1_params[0]} -> {jittor_params[gamma1_params[0]]}")
        if len(gamma2_params) > 0:
            print(f"Sample gamma2 parameter: {gamma2_params[0]} -> {jittor_params[gamma2_params[0]]}")
        
        return True
        
    except Exception as e:
        print(f"✗ LayerScale test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all debug tests."""
    print("Starting DFormerv2-Small debug session...")
    print("="*60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("LayerScale Parameters", test_layer_scale_parameters),
        ("Memory Usage", test_memory_usage),
        ("Forward Pass (Small)", test_forward_pass_small),
        ("Forward Pass (Standard)", test_forward_pass_standard),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
            results[test_name] = False
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("DEBUG SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! DFormerv2-Small should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == '__main__':
    main()
