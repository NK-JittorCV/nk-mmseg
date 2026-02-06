#!/usr/bin/env python3
"""
Test script to verify visualization functionality
"""

import os
import sys
import numpy as np
import jittor as jt
from importlib import import_module

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataloader.dataloader import get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.val_mm import evaluate
from utils.jt_utils import load_model
from utils.engine.engine import Engine
from models import build_model

def test_visualization():
    """Test visualization functionality with a small subset of data."""
    print("🎨 Testing DFormer Jittor Visualization")
    print("="*50)
    
    # Load config
    config = getattr(import_module('local_configs.NYUDepthv2.DFormer_Large'), "C")
    
    # Set device
    jt.flags.use_cuda = 1
    
    # Create engine
    engine = Engine()
    
    # Create data loader with small batch size for testing
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
        return False
    
    model.eval()
    
    # Create output directory
    output_dir = "output/visualization_test"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Test with first 5 samples only
    print("\n🔍 Testing visualization with first 5 samples...")
    
    # Create a subset of the data loader
    test_samples = []
    for i, sample in enumerate(val_loader):
        test_samples.append(sample)
        if i >= 4:  # Only test with 5 samples
            break
    
    # Create a simple data loader for testing
    class TestDataLoader:
        def __init__(self, samples):
            self.samples = samples
            self.dataset = val_loader.dataset
        
        def __iter__(self):
            return iter(self.samples)
        
        def __len__(self):
            return len(self.samples)
    
    test_loader = TestDataLoader(test_samples)
    
    # Run evaluation with visualization
    try:
        print("🚀 Running evaluation with visualization...")
        results = evaluate(
            model, 
            test_loader, 
            verbose=True, 
            save_dir=output_dir,
            config=config
        )
        
        print(f"\n📈 Results:")
        print(f"  mIoU: {results['mIoU']:.4f}")
        print(f"  mAcc: {results['mAcc']:.4f}")
        print(f"  Overall Acc: {results['Overall_Acc']:.4f}")
        
        # Check if visualization files were created
        pred_files = [f for f in os.listdir(output_dir) if f.endswith('_pred.png')]
        
        print(f"\n🎨 Visualization Results:")
        print(f"  Generated {len(pred_files)} prediction images")
        
        if pred_files:
            print(f"  Sample files:")
            for i, file in enumerate(pred_files[:3]):  # Show first 3 files
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"    {file} ({file_size} bytes)")
            
            # Check if files are colored (not just grayscale)
            import cv2
            sample_file = os.path.join(output_dir, pred_files[0])
            img = cv2.imread(sample_file)
            if img is not None:
                # Check if image has color (not grayscale)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # Check if it's actually colored (not just RGB grayscale)
                    is_colored = not np.allclose(img[:,:,0], img[:,:,1]) or not np.allclose(img[:,:,1], img[:,:,2])
                    if is_colored:
                        print(f"  ✅ SUCCESS: Images are properly colored!")
                        return True
                    else:
                        print(f"  ⚠️  WARNING: Images are grayscale (RGB format but no color)")
                        return False
                else:
                    print(f"  ❌ ERROR: Images are grayscale")
                    return False
            else:
                print(f"  ❌ ERROR: Could not read generated image")
                return False
        else:
            print(f"  ❌ ERROR: No prediction images were generated")
            return False
            
    except Exception as e:
        print(f"❌ Error during visualization test: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_color_palette():
    """Check if color palette files exist and are valid."""
    print("\n🎨 Checking Color Palette Files")
    print("="*40)
    
    # Check NYU color map
    nyu_palette_path = "utils/nyucmap.npy"
    if os.path.exists(nyu_palette_path):
        try:
            palette = np.load(nyu_palette_path)
            print(f"✅ NYU color palette found: {palette.shape}")
            print(f"   Data type: {palette.dtype}")
            print(f"   Value range: [{palette.min()}, {palette.max()}]")
            
            # Show first few colors
            print(f"   Sample colors:")
            for i in range(min(5, len(palette))):
                print(f"     Class {i}: RGB{tuple(palette[i])}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading NYU palette: {e}")
            return False
    else:
        print(f"❌ NYU color palette not found at: {nyu_palette_path}")
        return False

if __name__ == '__main__':
    print("🧪 DFormer Jittor Visualization Test")
    print("="*60)
    
    # Check prerequisites
    palette_ok = check_color_palette()
    
    if palette_ok:
        # Run visualization test
        viz_ok = test_visualization()
        
        if viz_ok:
            print(f"\n🎉 SUCCESS: Visualization test passed!")
            print(f"   DFormer-Jittor now generates colored prediction images!")
        else:
            print(f"\n❌ FAILED: Visualization test failed!")
            print(f"   Check the error messages above for details.")
    else:
        print(f"\n❌ FAILED: Color palette files missing!")
        print(f"   Please ensure nyucmap.npy is in the utils/ directory.")
