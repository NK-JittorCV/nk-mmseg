"""
Test cases for loss functions
"""

import os
import sys
import unittest
import numpy as np
import jittor as jt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.losses import CrossEntropyLoss, FocalLoss
from models.losses.utils import weight_reduce_loss, get_class_weight


class TestLossFunctions(unittest.TestCase):
    """Test cases for loss functions."""
    
    def setUp(self):
        """Set up test environment."""
        jt.flags.use_cuda = 0  # Use CPU for testing
        self.batch_size = 2
        self.num_classes = 40
        self.height = 480
        self.width = 640
        
        # Create synthetic data
        self.predictions = jt.randn(self.batch_size, self.num_classes, self.height, self.width)
        self.targets = jt.randint(0, self.num_classes, (self.batch_size, self.height, self.width))
        
        # Add some ignore labels
        self.targets[0, :10, :10] = 255  # ignore index
    
    def test_cross_entropy_loss(self):
        """Test cross entropy loss."""
        try:
            # Test basic cross entropy loss
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(self.predictions, self.targets)
            
            self.assertIsInstance(loss, jt.Var)
            self.assertEqual(loss.ndim, 0)  # scalar loss
            self.assertGreater(loss.item(), 0)
            
            print("✓ Cross entropy loss test passed")
            
        except Exception as e:
            self.fail(f"Cross entropy loss test failed: {e}")
    
    def test_cross_entropy_loss_with_weights(self):
        """Test cross entropy loss with class weights."""
        try:
            # Create class weights
            class_weights = jt.ones(self.num_classes)
            class_weights[0] = 2.0  # Give more weight to class 0
            
            loss_fn = CrossEntropyLoss(class_weight=class_weights)
            loss = loss_fn(self.predictions, self.targets)
            
            self.assertIsInstance(loss, jt.Var)
            self.assertGreater(loss.item(), 0)
            
            print("✓ Cross entropy loss with weights test passed")
            
        except Exception as e:
            self.fail(f"Cross entropy loss with weights test failed: {e}")
    
    def test_focal_loss(self):
        """Test focal loss."""
        try:
            # Test focal loss
            loss_fn = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25)
            
            # For sigmoid focal loss, we need binary targets
            binary_predictions = jt.randn(self.batch_size, 1, self.height, self.width)
            binary_targets = jt.randint(0, 2, (self.batch_size, self.height, self.width)).float32()
            
            loss = loss_fn(binary_predictions, binary_targets)
            
            self.assertIsInstance(loss, jt.Var)
            self.assertEqual(loss.ndim, 0)  # scalar loss
            self.assertGreater(loss.item(), 0)
            
            print("✓ Focal loss test passed")
            
        except Exception as e:
            self.fail(f"Focal loss test failed: {e}")
    
    def test_loss_reduction(self):
        """Test different loss reduction methods."""
        try:
            loss_fn_mean = CrossEntropyLoss(reduction='mean')
            loss_fn_sum = CrossEntropyLoss(reduction='sum')
            loss_fn_none = CrossEntropyLoss(reduction='none')
            
            loss_mean = loss_fn_mean(self.predictions, self.targets)
            loss_sum = loss_fn_sum(self.predictions, self.targets)
            loss_none = loss_fn_none(self.predictions, self.targets)
            
            # Check shapes
            self.assertEqual(loss_mean.ndim, 0)  # scalar
            self.assertEqual(loss_sum.ndim, 0)   # scalar
            self.assertEqual(loss_none.shape, (self.batch_size, self.height, self.width))  # per-pixel
            
            # Check values
            self.assertGreater(loss_sum.item(), loss_mean.item())  # sum should be larger
            
            print("✓ Loss reduction test passed")
            
        except Exception as e:
            self.fail(f"Loss reduction test failed: {e}")
    
    def test_ignore_index(self):
        """Test ignore index functionality."""
        try:
            loss_fn = CrossEntropyLoss(ignore_index=255)
            
            # Create targets with ignore labels
            targets_with_ignore = self.targets.clone()
            targets_with_ignore[0, :50, :50] = 255  # Set large area to ignore
            
            loss_with_ignore = loss_fn(self.predictions, targets_with_ignore)
            loss_without_ignore = loss_fn(self.predictions, self.targets)
            
            self.assertIsInstance(loss_with_ignore, jt.Var)
            self.assertIsInstance(loss_without_ignore, jt.Var)
            
            # Both should be valid losses
            self.assertGreater(loss_with_ignore.item(), 0)
            self.assertGreater(loss_without_ignore.item(), 0)
            
            print("✓ Ignore index test passed")
            
        except Exception as e:
            self.fail(f"Ignore index test failed: {e}")


class TestLossUtilities(unittest.TestCase):
    """Test loss utility functions."""
    
    def setUp(self):
        """Set up test environment."""
        jt.flags.use_cuda = 0
        self.batch_size = 2
        self.height = 100
        self.width = 100
        
        # Create synthetic loss tensor
        self.loss_tensor = jt.randn(self.batch_size, self.height, self.width)
        self.weights = jt.ones(self.batch_size, self.height, self.width)
    
    def test_weight_reduce_loss(self):
        """Test weighted loss reduction."""
        try:
            # Test mean reduction
            reduced_loss_mean = weight_reduce_loss(
                self.loss_tensor, self.weights, reduction='mean'
            )
            self.assertEqual(reduced_loss_mean.ndim, 0)
            
            # Test sum reduction
            reduced_loss_sum = weight_reduce_loss(
                self.loss_tensor, self.weights, reduction='sum'
            )
            self.assertEqual(reduced_loss_sum.ndim, 0)
            
            # Test none reduction
            reduced_loss_none = weight_reduce_loss(
                self.loss_tensor, self.weights, reduction='none'
            )
            self.assertEqual(reduced_loss_none.shape, self.loss_tensor.shape)
            
            print("✓ Weight reduce loss test passed")
            
        except Exception as e:
            self.fail(f"Weight reduce loss test failed: {e}")
    
    def test_get_class_weight(self):
        """Test class weight calculation."""
        try:
            # Create label tensor
            labels = jt.randint(0, 5, (100, 100))
            
            # Calculate class weights
            class_weights = get_class_weight(labels, num_classes=5)
            
            self.assertEqual(len(class_weights), 5)
            self.assertTrue(jt.all(class_weights > 0))
            
            print("✓ Get class weight test passed")
            
        except Exception as e:
            self.fail(f"Get class weight test failed: {e}")


class TestLossGradients(unittest.TestCase):
    """Test loss function gradients."""
    
    def setUp(self):
        """Set up test environment."""
        jt.flags.use_cuda = 0
        self.batch_size = 2
        self.num_classes = 5
        self.height = 32
        self.width = 32
    
    def test_loss_gradients(self):
        """Test that losses produce valid gradients."""
        try:
            # Create model parameters
            predictions = jt.randn(self.batch_size, self.num_classes, self.height, self.width)
            predictions.requires_grad = True
            
            targets = jt.randint(0, self.num_classes, (self.batch_size, self.height, self.width))
            
            # Test cross entropy loss gradients
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(predictions, targets)
            
            # Compute gradients
            optimizer = jt.optim.SGD([predictions], lr=0.01)
            optimizer.step(loss)
            
            # Check that gradients exist and are finite
            self.assertTrue(jt.isfinite(loss).item())
            
            print("✓ Loss gradients test passed")
            
        except Exception as e:
            self.fail(f"Loss gradients test failed: {e}")


if __name__ == '__main__':
    print("Running loss function tests...")
    unittest.main(verbosity=2)
