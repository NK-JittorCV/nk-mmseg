"""
UPerNet (Unified Perceptual Parsing Network) decoder converted to Jittor.
"""

import jittor as jt
from jittor import nn


class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.
    
    This head is the implementation of UPerNet 
    <https://arxiv.org/abs/1807.10221>.
    
    Args:
        in_channels (list): Input channels for each level
        num_classes (int): Number of classes  
        channels (int): Channels for intermediate features
        pool_scales (tuple): Pooling scales used in PSP Module
        norm_layer: Normalization layer
        dropout_ratio (float): Dropout ratio
        align_corners (bool): Whether to align corners in interpolation
    """

    def __init__(self,
                 in_channels=(96, 192, 384, 768),
                 num_classes=40,
                 channels=512,
                 pool_scales=(1, 2, 3, 6),
                 norm_layer=nn.BatchNorm2d,
                 dropout_ratio=0.1,
                 align_corners=False):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        
        # PSP Module
        self.psp_modules = PPM(
            pool_scales, 
            self.in_channels[-1], 
            self.channels, 
            norm_layer=norm_layer, 
            align_corners=align_corners
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels[-1] + len(pool_scales) * self.channels, self.channels, 3, padding=1),
            norm_layer(self.channels),
            nn.ReLU(),
        )
        
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_ch in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(in_ch, self.channels, 1), 
                norm_layer(self.channels), 
                nn.ReLU()
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, 3, padding=1),
                norm_layer(self.channels),
                nn.ReLU(),
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(self.in_channels) * self.channels, self.channels, 3, padding=1),
            norm_layer(self.channels),
            nn.ReLU(),
        )
        
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = jt.concat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def execute(self, inputs):
        """Forward function."""
        # Build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))

        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # Build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # Append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        
        fpn_outs = jt.concat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        
        if self.dropout is not None:
            output = self.dropout(output)
            
        output = self.conv_seg(output)
        return output


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.
    
    Args:
        pool_scales (tuple): Pooling scales used in Pooling Pyramid Module
        in_channel (int): Input channels
        channels (int): Channels after modules, before conv_seg
        norm_layer: Normalization layer
        align_corners (bool): Whether to align corners in interpolation
    """

    def __init__(self, pool_scales, in_channel, channels, norm_layer, align_corners=False):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channel = in_channel
        self.channels = channels
        
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(self.in_channel, self.channels, 1),
                    norm_layer(self.channels),
                    nn.ReLU(),
                )
            )

    def execute(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = nn.interpolate(
                ppm_out, size=x.shape[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs 