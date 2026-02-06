import jittor as jt
from jittor import nn
import jittor.nn as F
from .decode_head import BaseDecodeHead, ConvModule, resize


class _MatrixDecomposition2DBase(nn.Module):
    """Base class for 2D matrix decomposition methods."""

    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault("SPATIAL", True)
        self.S = args.setdefault("MD_S", 1)
        self.D = args.setdefault("MD_D", 512)
        self.R = args.setdefault("MD_R", 64)
        self.train_steps = args.setdefault("TRAIN_STEPS", 6)
        self.eval_steps = args.setdefault("EVAL_STEPS", 7)
        self.inv_t = args.setdefault("INV_T", 100)
        self.eta = args.setdefault("ETA", 0.9)
        self.rand_init = args.setdefault("RAND_INIT", True)

        print("spatial", self.spatial)
        print("S", self.S)
        print("D", self.D)
        print("R", self.R)
        print("train_steps", self.train_steps)
        print("eval_steps", self.eval_steps)
        print("inv_t", self.inv_t)
        print("eta", self.eta)
        print("rand_init", self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        """Build bases for matrix decomposition."""
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        """Local step for matrix decomposition."""
        raise NotImplementedError

    def local_inference(self, x, bases):
        """Local inference for matrix decomposition."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = jt.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        """Compute coefficients."""
        raise NotImplementedError

    def execute(self, x, return_bases=False):
        """Execute matrix decomposition."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, "bases"):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.bases = bases

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = jt.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    """2D Non-negative Matrix Factorization."""

    def __init__(self, args=dict()):
        super().__init__(args)
        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        """Build bases for NMF."""
        bases = jt.rand((B * S, D, R))
        # Normalize bases along dimension 1
        norm = jt.norm(bases, dim=1, keepdim=True)
        bases = bases / (norm + 1e-8)
        return bases

    def local_step(self, x, bases, coef):
        """Local step for NMF."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = jt.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = jt.bmm(coef, jt.bmm(bases.transpose(1, 2), bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = jt.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = jt.bmm(bases, jt.bmm(coef.transpose(1, 2), coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        """Compute coefficients for NMF."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = jt.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = jt.bmm(coef, jt.bmm(bases.transpose(1, 2), bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    """Hamburger module for HAM head."""

    def __init__(self, ham_channels=512, ham_kwargs=dict(), norm_cfg=None, **kwargs):
        super().__init__()

        self.ham_in = ConvModule(ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)
        self.ham = NMF2D(ham_kwargs)
        self.ham_out = ConvModule(ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def execute(self, x):
        """Execute Hamburger module."""
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy)
        return ham


class LightHamHead(BaseDecodeHead):
    """Light HAM Head for semantic segmentation.
    
    This head is the implementation of HAM (Hamburger) Net.
    Is Attention Better Than Matrix Decomposition?
    """

    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super(LightHamHead, self).__init__(input_transform="multiple_select", **kwargs)
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels, 
            self.channels, 
            1, 
            conv_cfg=self.conv_cfg, 
            norm_cfg=self.norm_cfg, 
            act_cfg=self.act_cfg
        )

    def execute(self, inputs):
        """Execute function."""
        inputs = self._transform_inputs(inputs)
        # Resize all inputs to the same size
        if isinstance(inputs, list):
            if len(inputs) > 1:
                inputs = [
                    resize(
                        x,
                        size=inputs[0].shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners
                    ) for x in inputs
                ]
            inputs = jt.concat(inputs, dim=1)
        
        # Squeeze channels
        feats = self.squeeze(inputs)
        
        # Apply hamburger
        feats = self.hamburger(feats)
        
        # Align channels
        feats = self.align(feats)
        
        # Final classification
        output = self.cls_seg_forward(feats)
        
        return output 