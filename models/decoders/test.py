"""
Test MLP Decoder converted from PyTorch to Jittor.
Uses all 4 feature levels (unlike LMLPDecoder which uses 3).
"""

import jittor as jt
from jittor import nn


class MLP(nn.Module):
    """Linear embedding layer."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def execute(self, x):
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class DecoderHead(nn.Module):
    """Test decoder head using MLP projections for all 4 feature levels."""
    
    def __init__(self,
                 in_channels=(64, 128, 320, 512),
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in, c2_in, c3_in, c4_in = self.in_channels

        # MLP projections for all 4 levels
        self.linear_c4 = MLP(input_dim=c4_in, embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=c3_in, embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=c2_in, embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=c1_in, embed_dim=embed_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1),
            norm_layer(embed_dim),
            nn.ReLU(),
        )

        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

    def execute(self, inputs):
        """Forward function."""
        # Input features at 4 scales: 1/4, 1/8, 1/16, 1/32
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape
        target_size = c1.shape[2:]

        # Project each level to embedding dimension
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = nn.interpolate(_c4, size=target_size, mode="bilinear", align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = nn.interpolate(_c3, size=target_size, mode="bilinear", align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = nn.interpolate(_c2, size=target_size, mode="bilinear", align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # Fuse all levels
        _c = self.linear_fuse(jt.concat([_c4, _c3, _c2, _c1], dim=1))
        
        if self.dropout is not None:
            x = self.dropout(_c)
        else:
            x = _c
        
        x = self.linear_pred(x)
        return x 