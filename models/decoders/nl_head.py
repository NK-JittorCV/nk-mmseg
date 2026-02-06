"""
Non-Local Neural Networks decoder converted to Jittor.
Simplified implementation without mmcv/mmseg dependencies.
"""

import jittor as jt
from jittor import nn
from abc import ABCMeta
from typing import Dict, Optional
from .decode_head import BaseDecodeHead, ConvModule


def normal_init(module, std=0.01):
    """Normal initialization."""
    if hasattr(module, 'weight'):
        nn.init.gauss_(module.weight, 0, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0)


def constant_init(module, val):
    """Constant initialization.""" 
    if hasattr(module, 'weight'):
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, val)


class _NonLocalNd(nn.Module, metaclass=ABCMeta):
    """Basic Non-local module."""

    def __init__(self, in_channels, reduction=2, use_scale=True, conv_cfg=None, norm_cfg=None, mode='embedded_gaussian'):
        super(_NonLocalNd, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']

        # g, theta, phi are actually `nn.Conv2d`. Here we use ConvModule for
        # potential usage.
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.theta = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.phi = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        """Initialize weights."""
        if self.mode != 'dot_product':
            normal_init(self.theta.conv, std=std)
            normal_init(self.phi.conv, std=std)
        normal_init(self.g.conv, std=std)
        if zeros_init:
            if self.conv_out.norm is not None:
                constant_init(self.conv_out.norm, 0)
            else:
                constant_init(self.conv_out.conv, 0)
        else:
            if self.conv_out.norm is not None:
                normal_init(self.conv_out.norm, std=std)
            else:
                normal_init(self.conv_out.conv, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        """Embedded gaussian with softmax."""
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = jt.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        """Dot product."""
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = jt.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def execute(self, x):
        """Execute function."""
        n = x.size(0)

        # g_x: [N, HxWx..., C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxWx..., C], phi_x: [N, C, HxWx...]
        if self.mode == 'embedded_gaussian':
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)
            pairwise_func = self.embedded_gaussian
        elif self.mode == 'dot_product':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = x.view(n, self.in_channels, -1)
            pairwise_func = self.dot_product
        else:
            raise NotImplementedError

        # pairwise_weight: [N, HxWx..., HxWx...]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxWx..., C]
        y = jt.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W, ...]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels, *x.shape[2:])

        output = x + self.conv_out(y)

        return output


class NonLocal2d(_NonLocalNd):
    """2D Non-local module."""

    def __init__(self, in_channels, **kwargs):
        super(NonLocal2d, self).__init__(in_channels, **kwargs)


class NLHead(BaseDecodeHead):
    """Non-Local Head for semantic segmentation."""

    def __init__(self, reduction=2, use_scale=True, mode='embedded_gaussian', **kwargs):
        super(NLHead, self).__init__(input_transform='multiple_select', **kwargs)
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        
        self.nl_block = NonLocal2d(
            in_channels=self.channels,
            reduction=reduction,
            use_scale=use_scale,
            mode=mode)
        
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def execute(self, inputs):
        """Execute function."""
        inputs = self._transform_inputs(inputs)
        
        if isinstance(inputs, list):
            # If multiple inputs, take the last one
            x = inputs[-1]
        else:
            x = inputs
            
        output = self.nl_block(x)
        output = self.cls_seg_forward(output)
        return output