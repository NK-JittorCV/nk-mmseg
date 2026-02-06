"""
Utility modules converted from PyTorch to Jittor.
Only used by geometry‐enhanced DFormer variants.
"""

import math
import jittor as jt
from jittor import nn
from utils.dformer_utils import trunc_normal_

# ------------------------------
# Channel and Spatial weighting
# ------------------------------

class ChannelWeights(nn.Module):
    """Channel attention using global pooling (avg+max)."""

    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = dim * 4 // reduction
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim * 2),
            nn.Sigmoid(),
        )

    def execute(self, x1, x2):
        B, _, H, W = x1.shape
        x = jt.concat([x1, x2], dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        mx = self.max_pool(x).view(B, self.dim * 2)
        y = jt.concat([avg, mx], dim=1)
        y = self.mlp(y).view(B, self.dim * 2, 1)
        ch_w = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)
        return ch_w


class SpatialWeights(nn.Module):
    """Spatial attention."""

    def __init__(self, dim, reduction=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(dim // reduction, 2, 1),
            nn.Sigmoid(),
        )

    def execute(self, x1, x2):
        B, _, H, W = x1.shape
        x = jt.concat([x1, x2], dim=1)
        sp_w = self.conv(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return sp_w


class FeatureRectifyModule(nn.Module):
    """Feature Rectify Module (FRM)."""

    def __init__(self, dim, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim, reduction)
        self.spatial_weights = SpatialWeights(dim, reduction)

    def execute(self, x1, x2):
        ch_w = self.channel_weights(x1, x2)
        sp_w = self.spatial_weights(x1, x2)
        out1 = x1 + self.lambda_c * ch_w[1] * x2 + self.lambda_s * sp_w[1] * x2
        out2 = x2 + self.lambda_c * ch_w[0] * x1 + self.lambda_s * sp_w[0] * x1
        return out1, out2

# ------------------------------
# Cross Attention utilities
# ------------------------------

class CrossAttention(nn.Module):
    """Cross attention used in CrossPath."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def execute(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q2 = x2.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k1, v1 = self.kv1(x1).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = self.kv2(x2).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = nn.softmax(ctx1, dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = nn.softmax(ctx2, dim=-2)

        out1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C)
        out2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C)
        return out1, out2


class CrossPath(nn.Module):
    """Two-stream cross-attention path."""

    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act = nn.ReLU()
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def execute(self, x1, x2):
        y1, u1 = self.act(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = jt.concat([y1, v1], dim=-1)
        y2 = jt.concat([y2, v2], dim=-1)
        out1 = self.norm1(x1 + self.end_proj1(y1))
        out2 = self.norm2(x2 + self.end_proj2(y2))
        return out1, out2


class ChannelEmbed(nn.Module):
    """Channel-wise embedding back to 2D feature maps."""

    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, 1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, 3, 1, 1, bias=True, groups=out_channels // reduction),
            nn.ReLU(),
            nn.Conv2d(out_channels // reduction, out_channels, 1, bias=True),
            norm_layer(out_channels),
        )
        self.norm = norm_layer(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def execute(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        residual = self.residual(x)
        x = self.channel_embed(x)
        return self.norm(residual + x)


class FeatureFusionModule(nn.Module):
    """Feature fusion based on CrossPath + ChannelEmbed."""

    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim, reduction, num_heads, nn.LayerNorm)
        self.channel_emb = ChannelEmbed(dim * 2, dim, reduction, norm_layer)

    def execute(self, x1, x2):
        B, C, H, W = x1.shape
        x1_f = x1.reshape(B, C, H * W).transpose(1, 2)
        x2_f = x2.reshape(B, C, H * W).transpose(1, 2)
        x1_f, x2_f = self.cross(x1_f, x2_f)
        merge = jt.concat([x1_f, x2_f], dim=-1)
        merge = self.channel_emb(merge, H, W)
        return merge 