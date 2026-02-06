import jittor as jt
from jittor import nn
import jittor.nn as F
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Optional, Union


class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for decode heads."""

    def __init__(
        self,
        in_channels: Union[int, List[int]],
        channels: int,
        num_classes: int,
        dropout_ratio: float = 0.1,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = dict(type='ReLU'),
        in_index: Union[int, List[int]] = -1,
        input_transform: Optional[str] = None,
        loss_decode: Optional[dict] = None,
        sampler: Optional[dict] = None,
        align_corners: bool = False,
        init_cfg: Optional[dict] = None
    ):
        super(BaseDecodeHead, self).__init__()
        
        # Set attributes
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.input_transform = input_transform
        self.align_corners = align_corners
        
        # Handle multiple input channels
        if isinstance(in_channels, list):
            self.in_channels = in_channels
        else:
            self.in_channels = [in_channels]
        
        # Handle multiple input indices
        if isinstance(in_index, int):
            self.in_index = [in_index]
        else:
            self.in_index = in_index
        
        # Build classifier
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        
        # Build dropout layer
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        
        # Build activation layer
        if act_cfg is not None:
            if act_cfg['type'] == 'ReLU':
                self.activation = nn.ReLU()
            elif act_cfg['type'] == 'GELU':
                self.activation = nn.GELU()
            else:
                self.activation = nn.ReLU()
        else:
            self.activation = None

    def _transform_inputs(self, inputs: List[jt.Var]) -> Union[jt.Var, List[jt.Var]]:
        """Transform inputs for decoder."""
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners
                ) for x in inputs
            ]
            inputs = jt.concat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index[0]]
        
        return inputs

    @abstractmethod
    def execute(self, inputs: List[jt.Var]) -> jt.Var:
        """Placeholder for execute function."""
        pass

    def cls_seg_forward(self, feat: jt.Var) -> jt.Var:
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.cls_seg(feat)
        return output

    def execute_test(self, inputs: List[jt.Var], img_metas: List[dict], test_cfg: dict) -> List[jt.Var]:
        """Execute function for testing."""
        return [self.execute(inputs)]


class ConvModule(nn.Module):
    """Conv-Norm-Act module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = dict(type='ReLU'),
        inplace: bool = True,
        with_spectral_norm: bool = False,
        padding_mode: str = 'zeros',
        order: Tuple[str, ...] = ('conv', 'norm', 'act')
    ):
        super(ConvModule, self).__init__()
        
        # Build conv layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Build norm layer
        if norm_cfg is not None:
            if norm_cfg.get('type') == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_cfg.get('type') == 'SyncBN':
                self.norm = nn.BatchNorm2d(out_channels)
            else:
                self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
        
        # Build activation layer
        if act_cfg is not None:
            if act_cfg.get('type') == 'ReLU':
                self.activate = nn.ReLU()
            elif act_cfg.get('type') == 'GELU':
                self.activate = nn.GELU()
            elif act_cfg.get('type') == 'LeakyReLU':
                self.activate = nn.LeakyReLU(0.01)
            else:
                self.activate = nn.ReLU()
        else:
            self.activate = None

    def execute(self, x: jt.Var) -> jt.Var:
        """Execute function."""
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


def resize(
    input: jt.Var,
    size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = 'nearest',
    align_corners: Optional[bool] = None
) -> jt.Var:
    """Resize a tensor."""
    if size is not None:
        return F.interpolate(input, size=size, mode=mode, align_corners=align_corners)
    else:
        return F.interpolate(input, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = dict(type='ReLU'),
        dw_norm_cfg: Optional[dict] = None,
        dw_act_cfg: Optional[dict] = dict(type='ReLU'),
        pw_norm_cfg: Optional[dict] = None,
        pw_act_cfg: Optional[dict] = dict(type='ReLU'),
    ):
        super(DepthwiseSeparableConvModule, self).__init__()
        
        # Depthwise convolution
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,
            act_cfg=dw_act_cfg,
            bias=False
        )
        
        # Pointwise convolution
        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
            bias=False
        )

    def execute(self, x: jt.Var) -> jt.Var:
        """Execute function."""
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x 