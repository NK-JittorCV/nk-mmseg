"""
Decoder modules for DFormer Jittor implementation.
"""

from .ham_head import LightHamHead
from .MLPDecoder import MLPDecoder, DecoderHead as MLPDecoderHead
from .decode_head import BaseDecodeHead as DecoderHead
from .fcnhead import FCNHead
from .LMLPDecoder import DecoderHead as LMLPDecoderHead
from .UPernet import UPerHead
from .deeplabv3plus import DeepLabV3Plus
from .nl_head import NLHead, NonLocal2d
from .test import DecoderHead as TestDecoderHead

__all__ = [
    'LightHamHead',
    'MLPDecoder', 'MLPDecoderHead',
    'DecoderHead',
    'FCNHead',
    'LMLPDecoderHead',
    'UPerHead',
    'DeepLabV3Plus', 
    'NLHead', 'NonLocal2d',
    'TestDecoderHead'
] 