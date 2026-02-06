"""
Test cases for DFormer models
"""

import os
import sys
import unittest
import numpy as np
import jittor as jt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from utils.config import load_config


class TestDFormerModels(unittest.TestCase):
    """Test cases for DFormer models."""
    
    def setUp(self):
        """Set up test environment."""
        jt.flags.use_cuda = 0  # Use CPU for testing
        self.batch_size = 2
        self.height = 480
        self.width = 640
        self.num_classes = 40
        
        # Create synthetic input data
        self.rgb_input = jt.randn(self.batch_size, 3, self.height, self.width)
        self.depth_input = jt.randn(self.batch_size, 3, self.height, self.width)
        
    def test_model_creation(self):
        """Test model creation with different configurations."""
        # Test with minimal config
        class MockConfig:
            backbone = 'DFormer-Base'
            decoder = 'ham'
            decoder_embed_dim = 512
            num_classes = 40
            drop_path_rate = 0.1
            aux_rate = 0.0
        
        config = MockConfig()
        
        try:
            model = build_model(config)
            self.assertIsNotNone(model)
            print("✓ Model creation test passed")
        except Exception as e:
            self.fail(f"Model creation failed: {e}")
    
    def test_model_forward(self):
        """Test model forward pass."""
        class MockConfig:
            backbone = 'DFormer-Base'
            decoder = 'ham'
            decoder_embed_dim = 512
            num_classes = 40
            drop_path_rate = 0.1
            aux_rate = 0.0
        
        config = MockConfig()
        
        try:
            model = build_model(config)
            model.eval()
            
            # Test forward pass
            with jt.no_grad():
                output = model(self.rgb_input, self.depth_input)
            
            # Check output shape
            if isinstance(output, dict):
                output = output['out']
            
            expected_shape = (self.batch_size, self.num_classes, self.height, self.width)
            self.assertEqual(output.shape, expected_shape)
            print("✓ Model forward pass test passed")
            
        except Exception as e:
            self.fail(f"Model forward pass failed: {e}")
    
    def test_model_training_mode(self):
        """Test model in training mode."""
        class MockConfig:
            backbone = 'DFormer-Base'
            decoder = 'ham'
            decoder_embed_dim = 512
            num_classes = 40
            drop_path_rate = 0.1
            aux_rate = 0.0
        
        config = MockConfig()
        
        try:
            model = build_model(config)
            model.train()
            
            # Test forward pass in training mode
            output = model(self.rgb_input, self.depth_input)
            
            # Check output shape
            if isinstance(output, dict):
                output = output['out']
            
            expected_shape = (self.batch_size, self.num_classes, self.height, self.width)
            self.assertEqual(output.shape, expected_shape)
            print("✓ Model training mode test passed")
            
        except Exception as e:
            self.fail(f"Model training mode failed: {e}")
    
    def test_model_gradient_flow(self):
        """Test gradient flow through model."""
        class MockConfig:
            backbone = 'DFormer-Base'
            decoder = 'ham'
            decoder_embed_dim = 512
            num_classes = 40
            drop_path_rate = 0.1
            aux_rate = 0.0
        
        config = MockConfig()
        
        try:
            model = build_model(config)
            model.train()
            
            # Create target
            target = jt.randint(0, self.num_classes, (self.batch_size, self.height, self.width))
            
            # Forward pass
            output = model(self.rgb_input, self.depth_input)
            if isinstance(output, dict):
                output = output['out']
            
            # Compute loss
            loss = jt.nn.cross_entropy_loss(output, target, ignore_index=255)
            
            # Backward pass
            optimizer = jt.optim.SGD(model.parameters(), lr=0.01)
            optimizer.step(loss)
            
            self.assertIsNotNone(loss.item())
            print("✓ Model gradient flow test passed")
            
        except Exception as e:
            self.fail(f"Model gradient flow failed: {e}")


class TestModelVariants(unittest.TestCase):
    """Test different model variants."""
    
    def setUp(self):
        """Set up test environment."""
        jt.flags.use_cuda = 0
        self.batch_size = 1
        self.height = 480
        self.width = 640
        self.rgb_input = jt.randn(self.batch_size, 3, self.height, self.width)
        self.depth_input = jt.randn(self.batch_size, 3, self.height, self.width)
    
    def test_different_backbones(self):
        """Test different backbone configurations."""
        backbones = ['DFormer-Tiny', 'DFormer-Small', 'DFormer-Base', 'DFormer-Large']
        
        for backbone in backbones:
            with self.subTest(backbone=backbone):
                class MockConfig:
                    backbone = backbone
                    decoder = 'ham'
                    decoder_embed_dim = 512
                    num_classes = 40
                    drop_path_rate = 0.1
                    aux_rate = 0.0
                
                config = MockConfig()
                
                try:
                    model = build_model(config)
                    model.eval()
                    
                    with jt.no_grad():
                        output = model(self.rgb_input, self.depth_input)
                    
                    if isinstance(output, dict):
                        output = output['out']
                    
                    expected_shape = (self.batch_size, 40, self.height, self.width)
                    self.assertEqual(output.shape, expected_shape)
                    print(f"✓ {backbone} test passed")
                    
                except Exception as e:
                    self.fail(f"{backbone} test failed: {e}")


if __name__ == '__main__':
    print("Running DFormer model tests...")
    unittest.main(verbosity=2)
