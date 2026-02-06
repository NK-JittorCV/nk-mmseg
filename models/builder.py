import jittor as jt
from jittor import nn
import jittor.nn as F
from typing import List, Optional, Dict, Any
import warnings


class EncoderDecoder(nn.Module):
    """Encoder-Decoder architecture for semantic segmentation."""

    def __init__(
        self,
        cfg,
        criterion=None,
        norm_layer=nn.BatchNorm2d,
        syncbn=False,
    ):
        super(EncoderDecoder, self).__init__()
        self.norm_layer = norm_layer
        self.cfg = cfg

        # Select backbone
        if cfg.backbone == "DFormer-Large":
            from .encoders.DFormer import DFormer_Large as backbone
            self.channels = [96, 192, 288, 576]
        elif cfg.backbone == "DFormer-Base":
            from .encoders.DFormer import DFormer_Base as backbone
            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == "DFormer-Small":
            from .encoders.DFormer import DFormer_Small as backbone
            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == "DFormer-Tiny":
            from .encoders.DFormer import DFormer_Tiny as backbone
            self.channels = [32, 64, 128, 256]
        elif cfg.backbone == "DFormerv2_L":
            from .encoders.DFormerv2 import DFormerv2_L as backbone
            self.channels = [112, 224, 448, 640]
        elif cfg.backbone == "DFormerv2_B":
            from .encoders.DFormerv2 import DFormerv2_B as backbone
            self.channels = [80, 160, 320, 512]
        elif cfg.backbone == "DFormerv2_S":
            from .encoders.DFormerv2 import DFormerv2_S as backbone
            self.channels = [64, 128, 256, 512]
        else:
            raise NotImplementedError(f"Backbone {cfg.backbone} not implemented")

        # Build norm config
        if syncbn:
            norm_cfg = dict(type="SyncBN", requires_grad=True)
        else:
            norm_cfg = dict(type="BN", requires_grad=True)

        # Build backbone
        if hasattr(cfg, 'drop_path_rate') and cfg.drop_path_rate is not None:
            self.backbone = backbone(drop_path_rate=cfg.drop_path_rate, norm_cfg=norm_cfg)
        else:
            self.backbone = backbone(drop_path_rate=0.1, norm_cfg=norm_cfg)

        # Build decoder
        self.aux_head = None

        if cfg.decoder == "MLPDecoder":
            print("Using MLP Decoder")
            print(f"Channels: {self.channels}")
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(
                in_channels=self.channels,
                num_classes=cfg.num_classes,
                norm_layer=norm_layer,
                embed_dim=cfg.decoder_embed_dim,
            )
            # Enable debug for this model
            self._debug_shapes = True

        elif cfg.decoder == "ham":
            print("Using Ham Decoder")
            from .decoders.ham_head import LightHamHead as DecoderHead
            self.decode_head = DecoderHead(
                in_channels=self.channels[1:],
                num_classes=cfg.num_classes,
                in_index=[1, 2, 3],
                norm_cfg=norm_cfg,
                channels=cfg.decoder_embed_dim,
            )
            
            # Auxiliary head
            if hasattr(cfg, 'aux_rate') and cfg.aux_rate != 0:
                from .decoders.fcnhead import FCNHead
                self.aux_index = 2
                self.aux_rate = cfg.aux_rate
                print("aux rate is set to", str(self.aux_rate))
                self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            print("Using simple FCN decoder")
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(
                in_channels=self.channels[-1], 
                kernel_size=3, 
                num_classes=cfg.num_classes, 
                norm_layer=norm_layer
            )

        # Set criterion
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        else:
            self.criterion = criterion

        # Initialize weights
        if hasattr(cfg, 'pretrained_model') and cfg.pretrained_model:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def init_weights(self, cfg, pretrained=None):
        """Initialize the weights in backbone."""
        if pretrained is not None:
            print(f"Loading pretrained model from {pretrained}")
            # TODO: implement loading pretrained weights
            pass

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation map."""
        # Extract features using backbone (RGB and modal inputs are processed separately)
        features = self.backbone(rgb, modal_x)

        # Handle different backbone output formats
        if isinstance(features, tuple) and len(features) == 2:
            # If backbone returns (features, aux_info), use features
            features = features[0]

        # Debug: print feature shapes
        if hasattr(self, '_debug_shapes'):
            print(f"Feature shapes: {[f.shape for f in features]}")
            print(f"Expected channels: {self.channels}")

        # Decode features
        out = self.decode_head(features)

        # Upsample to original size
        out = F.interpolate(out, size=rgb.shape[2:], mode='bilinear', align_corners=False)

        return out

    def execute(self, rgb, modal_x=None, label=None):
        """Execute function."""
        # Get prediction
        pred = self.encode_decode(rgb, modal_x)

        if self.training and label is not None:
            # Compute loss
            loss = self.criterion(pred, label)

            # Auxiliary loss
            if hasattr(self, 'aux_head') and self.aux_head is not None:
                aux_features = self.backbone(rgb, modal_x)
                if isinstance(aux_features, tuple) and len(aux_features) == 2:
                    aux_features = aux_features[0]
                aux_pred = self.aux_head(aux_features[self.aux_index])
                aux_pred = F.interpolate(aux_pred, size=rgb.shape[2:], mode='bilinear', align_corners=False)
                aux_loss = self.criterion(aux_pred, label)
                loss = loss + self.aux_rate * aux_loss

            return [pred], loss
        else:
            return [pred], None

    def forward(self, rgb, modal_x=None, label=None):
        """Forward function for compatibility."""
        return self.execute(rgb, modal_x, label)


class Config:
    """Configuration class for DFormer models."""
    
    def __init__(self, **kwargs):
        # Set default values
        self.backbone = "DFormer-Base"
        self.decoder = "ham"
        self.num_classes = 40
        self.decoder_embed_dim = 512
        self.drop_path_rate = 0.1
        self.aux_rate = 0.0
        self.pretrained_model = None
        
        # Update with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


def build_model(config_or_backbone, decoder=None, num_classes=None, **kwargs) -> EncoderDecoder:
    """Build a DFormer model.

    Args:
        config_or_backbone: Either a config object or backbone string
        decoder: Decoder type (if config_or_backbone is string)
        num_classes: Number of classes (if config_or_backbone is string)
        **kwargs: Additional arguments
    """
    if isinstance(config_or_backbone, str):
        # Legacy interface: build_model(backbone, decoder, num_classes)
        cfg = Config(
            backbone=config_or_backbone,
            decoder=decoder,
            num_classes=num_classes,
            **kwargs
        )
    else:
        # New interface: build_model(config)
        cfg = config_or_backbone

    return EncoderDecoder(cfg)