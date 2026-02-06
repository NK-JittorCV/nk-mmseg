"""
Inference utilities for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import os
import cv2
import numpy as np
import jittor as jt
from jittor import nn

from utils.transforms import Compose, Normalize, ToTensor
from utils.jt_utils import load_model


class InferenceEngine:
    """Inference engine for DFormer model."""
    
    def __init__(self, model, config=None, device=None):
        """Initialize inference engine.
        
        Args:
            model: DFormer model instance
            config: Model configuration
            device: Device to run inference on
        """
        self.model = model
        self.config = config
        self.model.eval()
        
        # Setup preprocessing
        self.setup_preprocessing()
    
    def setup_preprocessing(self):
        """Setup preprocessing pipeline."""
        if self.config:
            mean = getattr(self.config, 'norm_mean', [0.485, 0.456, 0.406])
            std = getattr(self.config, 'norm_std', [0.229, 0.224, 0.225])
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
    
    def preprocess_image(self, image_path, modal_path=None):
        """Preprocess input images.
        
        Args:
            image_path (str): Path to RGB image
            modal_path (str, optional): Path to depth/modal image
            
        Returns:
            tuple: Preprocessed RGB and modal tensors
        """
        # Load RGB image
        rgb_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Load modal image if provided
        modal_image = None
        if modal_path and os.path.exists(modal_path):
            modal_image = cv2.imread(modal_path)
            if modal_image is not None:
                modal_image = cv2.cvtColor(modal_image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        rgb_tensor = self.transform(rgb_image).unsqueeze(0)
        
        modal_tensor = None
        if modal_image is not None:
            modal_tensor = self.transform(modal_image).unsqueeze(0)
        
        return rgb_tensor, modal_tensor, rgb_image.shape[:2]
    
    def predict(self, rgb_tensor, modal_tensor=None):
        """Run model prediction.
        
        Args:
            rgb_tensor (jt.Var): RGB input tensor
            modal_tensor (jt.Var, optional): Modal input tensor
            
        Returns:
            jt.Var: Prediction logits
        """
        with jt.no_grad():
            if modal_tensor is not None:
                output = self.model(rgb_tensor, modal_tensor)
            else:
                output = self.model(rgb_tensor)
            
            if isinstance(output, dict):
                output = output['out']
            
            return output
    
    def postprocess(self, prediction, original_size):
        """Postprocess prediction.
        
        Args:
            prediction (jt.Var): Model prediction
            original_size (tuple): Original image size (H, W)
            
        Returns:
            np.ndarray: Processed prediction mask
        """
        # Get class predictions
        pred_mask = jt.argmax(prediction, dim=1).squeeze(0)
        
        # Resize to original size
        if pred_mask.shape != original_size:
            pred_mask = nn.interpolate(
                pred_mask.unsqueeze(0).unsqueeze(0).float32(),
                size=original_size,
                mode='nearest'
            ).squeeze().int32()
        
        return pred_mask.numpy()
    
    def infer_single_image(self, image_path, modal_path=None, save_path=None):
        """Infer on a single image.
        
        Args:
            image_path (str): Path to RGB image
            modal_path (str, optional): Path to modal image
            save_path (str, optional): Path to save prediction
            
        Returns:
            np.ndarray: Prediction mask
        """
        # Preprocess
        rgb_tensor, modal_tensor, original_size = self.preprocess_image(
            image_path, modal_path
        )
        
        # Predict
        prediction = self.predict(rgb_tensor, modal_tensor)
        
        # Postprocess
        pred_mask = self.postprocess(prediction, original_size)
        
        # Save if requested
        if save_path:
            self.save_prediction(pred_mask, save_path)
        
        return pred_mask
    
    def save_prediction(self, pred_mask, save_path):
        """Save prediction mask.
        
        Args:
            pred_mask (np.ndarray): Prediction mask
            save_path (str): Path to save
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, pred_mask.astype(np.uint8))


def create_color_palette(num_classes):
    """Create color palette for visualization.
    
    Args:
        num_classes (int): Number of classes
        
    Returns:
        np.ndarray: Color palette
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    
    for i in range(num_classes):
        # Generate distinct colors
        palette[i, 0] = (i * 67) % 256
        palette[i, 1] = (i * 113) % 256
        palette[i, 2] = (i * 197) % 256
    
    return palette


def visualize_prediction(image, prediction, palette=None, alpha=0.5):
    """Visualize prediction overlay on image.
    
    Args:
        image (np.ndarray): Original RGB image
        prediction (np.ndarray): Prediction mask
        palette (np.ndarray, optional): Color palette
        alpha (float): Overlay transparency
        
    Returns:
        np.ndarray: Visualization image
    """
    if palette is None:
        num_classes = int(prediction.max()) + 1
        palette = create_color_palette(num_classes)
    
    # Create colored prediction
    colored_pred = palette[prediction]
    
    # Blend with original image
    if image.shape[:2] != prediction.shape:
        image = cv2.resize(image, (prediction.shape[1], prediction.shape[0]))
    
    visualization = cv2.addWeighted(image, 1 - alpha, colored_pred, alpha, 0)
    
    return visualization


def batch_inference(model, image_dir, modal_dir=None, output_dir=None, config=None):
    """Run batch inference on a directory of images.
    
    Args:
        model: DFormer model
        image_dir (str): Directory containing RGB images
        modal_dir (str, optional): Directory containing modal images
        output_dir (str, optional): Directory to save predictions
        config: Model configuration
    """
    engine = InferenceEngine(model, config)
    
    # Get image files
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        # Find corresponding modal image
        modal_path = None
        if modal_dir:
            modal_path = os.path.join(modal_dir, image_file)
            if not os.path.exists(modal_path):
                modal_path = None
        
        # Run inference
        pred_mask = engine.infer_single_image(image_path, modal_path)
        
        # Save prediction
        if output_dir:
            save_path = os.path.join(output_dir, 
                                   os.path.splitext(image_file)[0] + '_pred.png')
            engine.save_prediction(pred_mask, save_path)
        
        print(f"Processed: {image_file}")


def load_model_for_inference(config_path, checkpoint_path):
    """Load model for inference.
    
    Args:
        config_path (str): Path to config file
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        model: Loaded model
    """
    # Load config
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # Create model
    from models import build_model
    model = build_model(config)
    
    # Load checkpoint
    if checkpoint_path:
        model = load_model(model, checkpoint_path)
    
    model.eval()
    return model, config
