import jittor as jt
from jittor import nn
import jittor.nn as F
from .decode_head import BaseDecodeHead, ConvModule


class FCNHead(BaseDecodeHead):
    """Simple FCN head for semantic segmentation."""

    def __init__(self, in_channels, num_classes, kernel_size=3, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCNHead, self).__init__(
            in_channels=in_channels, 
            channels=in_channels,
            num_classes=num_classes,
            **kwargs
        )
        self.kernel_size = kernel_size
        
        # Build convolution layer
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=kernel_size, padding=kernel_size//2)
        
        # Build norm layer
        if norm_layer is not None:
            self.norm = norm_layer(in_channels)
        else:
            self.norm = None
            
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.1)

    def execute(self, inputs):
        """Execute function."""
        if isinstance(inputs, list):
            x = inputs[-1]  # Use the last feature map
        else:
            x = inputs
            
        # Apply normalization and activation
        if self.norm is not None:
            x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Classification
        output = self.conv_seg(x)
        
        return output 