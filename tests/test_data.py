"""
Test cases for data loading and preprocessing
"""

import os
import sys
import unittest
import numpy as np
import jittor as jt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataloader import RGBXDataset, get_train_loader, get_val_loader
from utils.transforms import Compose, ToTensor, Normalize, RandomScale, RandomCrop


class TestDataLoading(unittest.TestCase):
    """Test cases for data loading."""
    
    def setUp(self):
        """Set up test environment."""
        self.num_classes = 40
        self.image_height = 480
        self.image_width = 640
        
        # Create mock config
        class MockConfig:
            dataset_name = "NYUDepthv2"
            dataset_path = "datasets/NYUDepthv2"
            rgb_root_folder = "datasets/NYUDepthv2/RGB"
            rgb_format = ".jpg"
            gt_root_folder = "datasets/NYUDepthv2/Label"
            gt_format = ".png"
            gt_transform = True
            x_root_folder = "datasets/NYUDepthv2/Depth"
            x_format = ".png"
            x_is_single_channel = True
            train_source = "datasets/NYUDepthv2/train.txt"
            eval_source = "datasets/NYUDepthv2/test.txt"
            num_classes = 40
            background = 255
            image_height = 480
            image_width = 640
            norm_mean = np.array([0.485, 0.456, 0.406])
            norm_std = np.array([0.229, 0.224, 0.225])
            batch_size = 4
            num_workers = 2
            train_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        
        self.config = MockConfig()
    
    def test_transforms(self):
        """Test data transforms."""
        try:
            # Test individual transforms
            to_tensor = ToTensor()
            normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            random_scale = RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5])
            random_crop = RandomCrop(crop_size=[480, 640])
            
            # Test compose
            transform = Compose([
                to_tensor,
                normalize
            ])
            
            # Create dummy image
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Apply transform
            transformed = transform(dummy_image)
            
            self.assertIsInstance(transformed, jt.Var)
            self.assertEqual(transformed.shape, (3, 480, 640))
            print("✓ Transform test passed")
            
        except Exception as e:
            self.fail(f"Transform test failed: {e}")
    
    def test_synthetic_dataset(self):
        """Test dataset with synthetic data."""
        try:
            # Create synthetic dataset
            dataset = self._create_synthetic_dataset()
            
            # Test dataset length
            self.assertGreater(len(dataset), 0)
            
            # Test data loading
            sample = dataset[0]
            self.assertIn('rgb', sample)
            self.assertIn('modal', sample)
            self.assertIn('label', sample)
            
            # Check data types and shapes
            self.assertIsInstance(sample['rgb'], jt.Var)
            self.assertIsInstance(sample['modal'], jt.Var)
            self.assertIsInstance(sample['label'], jt.Var)
            
            self.assertEqual(sample['rgb'].shape, (3, self.image_height, self.image_width))
            self.assertEqual(sample['modal'].shape, (3, self.image_height, self.image_width))
            self.assertEqual(sample['label'].shape, (self.image_height, self.image_width))
            
            print("✓ Synthetic dataset test passed")
            
        except Exception as e:
            self.fail(f"Synthetic dataset test failed: {e}")
    
    def test_dataloader_creation(self):
        """Test dataloader creation."""
        try:
            # Test with synthetic data
            dataset = self._create_synthetic_dataset()
            
            # Create dataloader
            from jittor.dataset import DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=True,
                num_workers=0  # Use 0 for testing
            )
            
            # Test batch loading
            for batch in dataloader:
                self.assertIn('rgb', batch)
                self.assertIn('modal', batch)
                self.assertIn('label', batch)
                
                # Check batch dimensions
                self.assertEqual(batch['rgb'].shape[0], 2)  # batch size
                self.assertEqual(batch['modal'].shape[0], 2)
                self.assertEqual(batch['label'].shape[0], 2)
                
                break  # Test only first batch
            
            print("✓ DataLoader creation test passed")
            
        except Exception as e:
            self.fail(f"DataLoader creation test failed: {e}")
    
    def _create_synthetic_dataset(self):
        """Create synthetic dataset for testing."""
        class SyntheticDataset:
            def __init__(self, num_samples=10):
                self.num_samples = num_samples
                self.transform = Compose([
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                # Generate synthetic RGB image
                rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # Generate synthetic depth image
                modal = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # Generate synthetic label
                label = np.random.randint(0, 40, (480, 640), dtype=np.uint8)
                
                # Apply transforms
                rgb_tensor = self.transform(rgb)
                modal_tensor = self.transform(modal)
                label_tensor = jt.array(label)
                
                return {
                    'rgb': rgb_tensor,
                    'modal': modal_tensor,
                    'label': label_tensor
                }
        
        return SyntheticDataset()


class TestDataAugmentation(unittest.TestCase):
    """Test data augmentation functions."""
    
    def test_random_scale(self):
        """Test random scale augmentation."""
        try:
            scale_transform = RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5])
            
            # Create dummy data
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            label = np.random.randint(0, 40, (480, 640), dtype=np.uint8)
            
            # Apply transform
            scaled_image, scaled_label = scale_transform(image, label)
            
            self.assertIsInstance(scaled_image, np.ndarray)
            self.assertIsInstance(scaled_label, np.ndarray)
            self.assertEqual(len(scaled_image.shape), 3)
            self.assertEqual(len(scaled_label.shape), 2)
            
            print("✓ Random scale test passed")
            
        except Exception as e:
            self.fail(f"Random scale test failed: {e}")
    
    def test_random_crop(self):
        """Test random crop augmentation."""
        try:
            crop_transform = RandomCrop(crop_size=[480, 640])
            
            # Create dummy data (larger than crop size)
            image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            label = np.random.randint(0, 40, (600, 800), dtype=np.uint8)
            
            # Apply transform
            cropped_image, cropped_label = crop_transform(image, label)
            
            self.assertEqual(cropped_image.shape, (480, 640, 3))
            self.assertEqual(cropped_label.shape, (480, 640))
            
            print("✓ Random crop test passed")
            
        except Exception as e:
            self.fail(f"Random crop test failed: {e}")


if __name__ == '__main__':
    print("Running data loading tests...")
    unittest.main(verbosity=2)
