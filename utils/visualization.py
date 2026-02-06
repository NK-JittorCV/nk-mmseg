"""
Visualization utilities for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap


# NYU Depth v2 color palette
NYU_PALETTE = np.array([
    [0, 0, 0],        # void
    [128, 0, 0],      # wall
    [0, 128, 0],      # floor
    [128, 128, 0],    # cabinet
    [0, 0, 128],      # bed
    [128, 0, 128],    # chair
    [0, 128, 128],    # sofa
    [128, 128, 128],  # table
    [64, 0, 0],       # door
    [192, 0, 0],      # window
    [64, 128, 0],     # bookshelf
    [192, 128, 0],    # picture
    [64, 0, 128],     # counter
    [192, 0, 128],    # blinds
    [64, 128, 128],   # desk
    [192, 128, 128],  # shelves
    [0, 64, 0],       # curtain
    [128, 64, 0],     # dresser
    [0, 192, 0],      # pillow
    [128, 192, 0],    # mirror
    [0, 64, 128],     # floor mat
    [128, 64, 128],   # clothes
    [0, 192, 128],    # ceiling
    [128, 192, 128],  # books
    [64, 64, 0],      # fridge
    [192, 64, 0],     # tv
    [64, 192, 0],     # paper
    [192, 192, 0],    # towel
    [64, 64, 128],    # shower curtain
    [192, 64, 128],   # box
    [64, 192, 128],   # whiteboard
    [192, 192, 128],  # person
    [0, 0, 64],       # night stand
    [128, 0, 64],     # toilet
    [0, 128, 64],     # sink
    [128, 128, 64],   # lamp
    [0, 0, 192],      # bathtub
    [128, 0, 192],    # bag
    [0, 128, 192],    # other struct
    [128, 128, 192],  # other furntr
    [64, 0, 64],      # other prop
], dtype=np.uint8)


# SUNRGBD color palette
SUNRGBD_PALETTE = np.array([
    [0, 0, 0],        # void
    [119, 119, 119],  # wall
    [244, 243, 131],  # floor
    [137, 28, 157],   # cabinet
    [150, 255, 255],  # bed
    [54, 114, 113],   # chair
    [0, 0, 176],      # sofa
    [255, 69, 0],     # table
    [87, 112, 255],   # door
    [0, 163, 33],     # window
    [255, 150, 255],  # bookshelf
    [255, 180, 10],   # picture
    [101, 70, 86],    # counter
    [38, 230, 0],     # blinds
    [255, 120, 70],   # desk
    [117, 41, 121],   # shelves
    [150, 255, 0],    # curtain
    [132, 0, 255],    # dresser
    [24, 209, 255],   # pillow
    [191, 130, 35],   # mirror
    [219, 200, 109],  # floor mat
    [154, 62, 86],    # clothes
    [255, 208, 186],  # ceiling
    [0, 71, 84],      # books
    [255, 0, 118],    # fridge
    [255, 0, 0],      # tv
    [168, 0, 0],      # paper
    [0, 255, 0],      # towel
    [0, 255, 127],    # shower curtain
    [255, 255, 0],    # box
    [0, 0, 255],      # whiteboard
    [255, 0, 255],    # person
    [127, 255, 212],  # night stand
    [227, 207, 87],   # toilet
    [43, 85, 0],      # sink
    [85, 85, 0],      # lamp
    [127, 0, 0],      # bathtub
    [0, 127, 0],      # bag
    [72, 0, 118],     # other struct
    [118, 0, 118],    # other furntr
    [76, 76, 76],     # other prop
], dtype=np.uint8)


def get_palette(dataset='nyudepthv2'):
    """Get color palette for dataset.
    
    Args:
        dataset (str): Dataset name
        
    Returns:
        np.ndarray: Color palette
    """
    if dataset.lower() == 'nyudepthv2':
        return NYU_PALETTE
    elif dataset.lower() == 'sunrgbd':
        return SUNRGBD_PALETTE
    else:
        # Generate default palette
        num_classes = 41  # Default
        return generate_palette(num_classes)


def generate_palette(num_classes):
    """Generate color palette for given number of classes.
    
    Args:
        num_classes (int): Number of classes
        
    Returns:
        np.ndarray: Generated color palette
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    
    for i in range(num_classes):
        palette[i, 0] = (i * 67) % 256
        palette[i, 1] = (i * 113) % 256
        palette[i, 2] = (i * 197) % 256
    
    return palette


def colorize_mask(mask, palette):
    """Colorize segmentation mask.
    
    Args:
        mask (np.ndarray): Segmentation mask
        palette (np.ndarray): Color palette
        
    Returns:
        np.ndarray: Colorized mask
    """
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for class_id in range(len(palette)):
        colored_mask[mask == class_id] = palette[class_id]
    
    return colored_mask


def overlay_mask(image, mask, palette, alpha=0.5):
    """Overlay segmentation mask on image.
    
    Args:
        image (np.ndarray): Original image
        mask (np.ndarray): Segmentation mask
        palette (np.ndarray): Color palette
        alpha (float): Overlay transparency
        
    Returns:
        np.ndarray: Image with overlay
    """
    colored_mask = colorize_mask(mask, palette)
    
    # Resize if needed
    if image.shape[:2] != mask.shape:
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
    
    # Blend images
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    return overlay


def visualize_results(image, prediction, target=None, palette=None, 
                     dataset='nyudepthv2', save_path=None, show=True):
    """Visualize segmentation results.
    
    Args:
        image (np.ndarray): Original RGB image
        prediction (np.ndarray): Prediction mask
        target (np.ndarray, optional): Ground truth mask
        palette (np.ndarray, optional): Color palette
        dataset (str): Dataset name
        save_path (str, optional): Path to save visualization
        show (bool): Whether to show the plot
    """
    if palette is None:
        palette = get_palette(dataset)
    
    # Setup subplot layout
    num_plots = 2 if target is None else 4
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction
    pred_colored = colorize_mask(prediction, palette)
    axes[1].imshow(pred_colored)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    if target is not None:
        # Ground truth
        target_colored = colorize_mask(target, palette)
        axes[2].imshow(target_colored)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        # Overlay
        overlay = overlay_mask(image, prediction, palette)
        axes[3].imshow(overlay)
        axes[3].set_title('Prediction Overlay')
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def create_legend(palette, class_names=None, save_path=None):
    """Create legend for color palette.
    
    Args:
        palette (np.ndarray): Color palette
        class_names (list, optional): Class names
        save_path (str, optional): Path to save legend
    """
    num_classes = len(palette)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=(3, num_classes * 0.3))
    
    for i, (color, name) in enumerate(zip(palette, class_names)):
        rect = patches.Rectangle((0, i), 1, 0.8, 
                               facecolor=color/255.0, edgecolor='black')
        ax.add_patch(rect)
        ax.text(1.1, i + 0.4, name, va='center', fontsize=10)
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, num_classes)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def save_prediction_image(prediction, save_path, palette=None, dataset='nyudepthv2'):
    """Save prediction as colored image.
    
    Args:
        prediction (np.ndarray): Prediction mask
        save_path (str): Path to save image
        palette (np.ndarray, optional): Color palette
        dataset (str): Dataset name
    """
    if palette is None:
        palette = get_palette(dataset)
    
    colored_pred = colorize_mask(prediction, palette)
    
    # Convert RGB to BGR for OpenCV
    colored_pred_bgr = cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, colored_pred_bgr)


def compare_predictions(image, pred1, pred2, labels=None, palette=None, 
                       dataset='nyudepthv2', save_path=None):
    """Compare two predictions side by side.
    
    Args:
        image (np.ndarray): Original image
        pred1 (np.ndarray): First prediction
        pred2 (np.ndarray): Second prediction
        labels (list, optional): Labels for predictions
        palette (np.ndarray, optional): Color palette
        dataset (str): Dataset name
        save_path (str, optional): Path to save comparison
    """
    if palette is None:
        palette = get_palette(dataset)
    
    if labels is None:
        labels = ['Prediction 1', 'Prediction 2']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # First prediction
    pred1_colored = colorize_mask(pred1, palette)
    axes[1].imshow(pred1_colored)
    axes[1].set_title(labels[0])
    axes[1].axis('off')
    
    # Second prediction
    pred2_colored = colorize_mask(pred2, palette)
    axes[2].imshow(pred2_colored)
    axes[2].set_title(labels[1])
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_confusion_matrix_plot(confusion_matrix, class_names=None, save_path=None):
    """Create confusion matrix visualization.
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix
        class_names (list, optional): Class names
        save_path (str, optional): Path to save plot
    """
    num_classes = confusion_matrix.shape[0]
    
    if class_names is None:
        class_names = [f'C{i}' for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if confusion_matrix[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
