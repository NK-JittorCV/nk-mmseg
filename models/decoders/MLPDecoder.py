import jittor as jt
from jittor import nn
import jittor.nn as F
from .decode_head import BaseDecodeHead, ConvModule, resize


class MLPDecoder(nn.Module):
    """MLP Decoder for semantic segmentation."""

    def __init__(self, in_channels, num_classes, norm_layer=nn.BatchNorm2d, embed_dim=256):
        super(MLPDecoder, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Build feature projection layers
        self.linear_c4 = nn.Conv2d(in_channels[-1], embed_dim, 1)
        self.linear_c3 = nn.Conv2d(in_channels[-2], embed_dim, 1)
        self.linear_c2 = nn.Conv2d(in_channels[-3], embed_dim, 1)
        self.linear_c1 = nn.Conv2d(in_channels[-4], embed_dim, 1)

        # Fusion layers
        self.linear_fuse = nn.Conv2d(embed_dim * 4, embed_dim, 1)
        self.norm = norm_layer(embed_dim)

        # Classification layer
        self.classifier = nn.Conv2d(embed_dim, num_classes, 1)

        # Dropout
        self.dropout = nn.Dropout2d(0.1)

    def execute(self, inputs):
        """Execute function."""
        c1, c2, c3, c4 = inputs

        # Project features
        c4 = self.linear_c4(c4)
        c3 = self.linear_c3(c3)
        c2 = self.linear_c2(c2)
        c1 = self.linear_c1(c1)

        # Upsample to the same size
        c4 = F.interpolate(c4, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c3 = F.interpolate(c3, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c2 = F.interpolate(c2, size=c1.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate features
        _c = jt.concat([c4, c3, c2, c1], dim=1)

        # Fusion
        _c = self.linear_fuse(_c)
        _c = self.norm(_c)
        _c = F.relu(_c)
        _c = self.dropout(_c)

        # Classification
        x = self.classifier(_c)

        return x


class DecoderHead(BaseDecodeHead):
    """MLP Decoder Head."""

    def __init__(self, in_channels, num_classes, norm_layer=nn.BatchNorm2d, embed_dim=256, **kwargs):
        super(DecoderHead, self).__init__(
            in_channels=in_channels,
            channels=embed_dim,
            num_classes=num_classes,
            **kwargs
        )
        self.decoder = MLPDecoder(in_channels, num_classes, norm_layer, embed_dim)

    def execute(self, inputs):
        """Execute function."""
        return self.decoder(inputs) 