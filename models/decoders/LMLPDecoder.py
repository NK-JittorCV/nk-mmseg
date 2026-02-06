"""
Lightweight MLP Decoder converted from PyTorch to Jittor.
"""

import jittor as jt
from jittor import nn

from ..encoders import DFormer  # resolves circular import only when needed


class MLP(nn.Module):
    """Linear embedding layer."""

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def execute(self, x):
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class DecoderHead(nn.Module):
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

        c2_in, c3_in, c4_in = in_channels[1:]
        self.linear_c4 = MLP(c4_in, embed_dim)
        self.linear_c3 = MLP(c3_in, embed_dim)
        self.linear_c2 = MLP(c2_in, embed_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim, 1),
            norm_layer(embed_dim),
            nn.ReLU(),
        )
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None

    def execute(self, features):
        # assume inputs list with 4 levels, we use last three
        c2, c3, c4 = features[1:]
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, h, w)
        _c4 = nn.interpolate(_c4, size=c2.shape[2:], mode='bilinear', align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = nn.interpolate(_c3, size=c2.shape[2:], mode='bilinear', align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])

        x = jt.concat([_c4, _c3, _c2], dim=1)
        x = self.linear_fuse(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear_pred(x)
        return x 