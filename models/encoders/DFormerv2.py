"""
DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation
Code: https://github.com/VCIP-RGBD/DFormer

Author: yinbow
Email: bowenyin@mail.nankai.edu.cn

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import jittor as jt
from jittor import nn
import jittor.nn as F
import math
from typing import List, Tuple
from collections import OrderedDict


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Initialize tensor with truncated normal distribution.

    Args:
        tensor: Tensor to initialize
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        a: Lower bound for truncation
        b: Upper bound for truncation
    """
    # Use the implementation from utils.dformer_utils
    from utils.dformer_utils import trunc_normal_ as _trunc_normal_impl
    return _trunc_normal_impl(tensor, mean, std, a, b)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def execute(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        random_tensor = jt.rand(shape, dtype=x.dtype)
        keep_prob_tensor = jt.full_like(random_tensor, keep_prob)
        random_tensor = jt.add(random_tensor, keep_prob_tensor)
        random_tensor = jt.floor(random_tensor)  # binarize

        if self.scale_by_keep:
            random_tensor = jt.divide(random_tensor, keep_prob_tensor)
        
        return x * random_tensor


def load_state_dict(model, state_dict, strict=True):
    """Load state dict to model."""
    missing_keys = []
    unexpected_keys = []
    
    model_state_dict = model.state_dict()
    
    for key, value in state_dict.items():
        if key in model_state_dict:
            try:
                model_state_dict[key].assign(value)
            except:
                print(f"Failed to load parameter {key}")
                missing_keys.append(key)
        else:
            unexpected_keys.append(key)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(f"Error loading state dict: missing {len(missing_keys)} keys, unexpected {len(unexpected_keys)} keys")


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    """Load checkpoint from file."""
    try:
        checkpoint = jt.load(filename)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        load_state_dict(model, state_dict, strict)
        return checkpoint
    except Exception as e:
        print(f"Failed to load checkpoint from {filename}: {e}")
        return None


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D tensors."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def execute(self, x):
        """
        input shape (b c h w)
        """
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        x = self.norm(x)  # (b h w c)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Use simple approach to avoid initialization issues
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
        )

    def execute(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)
        return x


class DWConv2d(nn.Module):
    """Depthwise convolution 2d."""

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def execute(self, x):
        """
        input (b h w c)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def execute(self, x):
        """
        x: B H W C
        """
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.reduction(x)  # (b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (b oh ow oc)
        return x


def angle_transform(x, sin, cos):
    """
    x: [B, num_heads, H, W, C_head]
    sin, cos: [B, H, W, C_head]
    """
    # Reshape sin and cos to broadcast correctly for multi-head attention
    sin = sin.unsqueeze(1)  # -> [B, 1, H, W, C_head]
    cos = cos.unsqueeze(1)  # -> [B, 1, H, W, C_head]

    # Apply rotary angle transformation, matching PyTorch's implementation
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated_x = jt.stack([-x2, x1], dim=-1).flatten(-2)  # flatten last two dims

    return x * cos + rotated_x * sin


class GeoPriorGen(nn.Module):
    """Geometry Prior Generator for depth-aware attention."""

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** jt.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.weight = jt.ones(2, 1, 1, 1)
        decay = jt.log(
            1 - 2 ** (-initial_value - heads_range * jt.arange(num_heads, dtype=jt.float32) / num_heads)
        )
        # Register as buffers (non-trainable parameters)
        self.angle = angle
        self.decay = decay

    def generate_depth_decay(self, H, W, depth_grid):
        """Generate 2d depth decay mask."""
        B, C, H_grid, W_grid = depth_grid.shape
        # Take the first channel if multiple channels exist
        if C > 1:
            depth_grid = depth_grid[:, 0:1, :, :]  # [B, 1, H, W]
        grid_d = depth_grid.reshape(B, H_grid * W_grid, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_d

    def generate_pos_decay(self, H, W):
        """Generate 2d position decay mask."""
        index_h = jt.arange(H)
        index_w = jt.arange(W)
        grid_h, grid_w = jt.meshgrid(index_h, index_w)
        grid = jt.stack([grid_h, grid_w], dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_depth_decay(self, H, W, depth_grid):
        """Generate 1d depth decay mask."""
        B, C, H_grid, W_grid = depth_grid.shape
        if C > 1:
            depth_grid = depth_grid[:, 0:1, :, :]
        
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()

        num_heads = self.decay.shape[0]
        decay_expanded = self.decay.reshape(1, num_heads, 1, 1, 1, 1)

        mask = mask.unsqueeze(1).expand(B, num_heads, C, H_grid, W_grid, W_grid) * decay_expanded
        return mask.squeeze(2)


    def generate_1d_decay(self, l):
        """Generate 1d decay mask."""
        index = jt.arange(l)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()

        num_heads = self.decay.shape[0]
        decay_expanded = self.decay.reshape(1, num_heads, 1, 1)

        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, num_heads, l, l) * decay_expanded
        return mask.squeeze(0)

    def execute(self, HW_tuple, depth_map, split_or_not=False):
        """
        depth_map: depth patches
        HW_tuple: (H, W)
        H * W == l
        """
        B, C, H, W = depth_map.shape
        depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)
        
        # Match dimensions for sin/cos generation
        index = jt.arange(HW_tuple[0] * HW_tuple[1])
        sin = jt.sin(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)
        cos = jt.cos(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)
        # Expand to match batch size
        sin = sin.unsqueeze(0).expand(B, -1, -1, -1)
        cos = cos.unsqueeze(0).expand(B, -1, -1, -1)
        
        if split_or_not:
            mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))
            mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)

            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])

            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w

            geo_prior = ((sin, cos), (mask_h, mask_w))

        else:
            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
            mask_p = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])

            mask = self.weight[0] * mask_p.unsqueeze(0) + self.weight[1] * mask_d
            geo_prior = ((sin, cos), mask)

        return geo_prior


class Decomposed_GSA(nn.Module):
    """Decomposed Geometry Self-Attention module."""

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * value_factor, bias=True)
        
        # Add LEPE (Learned Position Encoding) - critical component from PyTorch version
        self.lepe = DWConv2d(embed_dim * value_factor, 5, 1, 2)
        
        self.out_proj = nn.Linear(embed_dim * value_factor, embed_dim, bias=True)
        self.reset_parameters()

    def execute(self, x, rel_pos, split_or_not=False):
        """
        x: B, H, W, C
        rel_pos: relative position encoding
        """
        B, H, W, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        lepe = self.lepe(v)

        if split_or_not:
            # Decomposed attention
            ((sin, cos), (mask_h, mask_w)) = rel_pos
            
            k = k * self.scaling
            
            q = q.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
            k = k.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
            
            qr = angle_transform(q, sin, cos)
            kr = angle_transform(k, sin, cos)
            
            qr_w = qr.transpose(1, 2)
            kr_w = kr.transpose(1, 2)
            v_reshaped = v.reshape(B, H, W, self.num_heads, -1).permute(0, 1, 3, 2, 4)
            
            qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
            qk_mat_w = qk_mat_w + mask_w.transpose(1, 2)
            qk_mat_w = F.softmax(qk_mat_w, dim=-1)
            v_reshaped = jt.matmul(qk_mat_w, v_reshaped)
            
            qr_h = qr.permute(0, 3, 1, 2, 4)
            kr_h = kr.permute(0, 3, 1, 2, 4)
            v_reshaped = v_reshaped.permute(0, 3, 2, 1, 4)
            
            qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
            qk_mat_h = qk_mat_h + mask_h.transpose(1, 2)
            qk_mat_h = F.softmax(qk_mat_h, dim=-1)
            output = jt.matmul(qk_mat_h, v_reshaped)
            
            output = output.permute(0, 3, 1, 2, 4).reshape(B, H, W, -1)
        else:
            # Full attention
            (sin, cos), mask = rel_pos
            k = k * self.scaling
            
            q_reshaped = q.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
            k_reshaped = k.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
            
            # Use angle_transform directly
            qr = angle_transform(q_reshaped, sin, cos)
            kr = angle_transform(k_reshaped, sin, cos)
            
            qr = qr.flatten(2, 3)
            kr = kr.flatten(2, 3)
            vr = v.reshape(B, H, W, self.num_heads, -1).permute(0, 3, 1, 2, 4).flatten(2, 3)
            
            qk_mat = qr @ kr.transpose(-2, -1)
            qk_mat = qk_mat + mask
            qk_mat = F.softmax(qk_mat, dim=-1)
            output = jt.matmul(qk_mat, vr)
            
            output = output.transpose(1, 2).reshape(B, H, W, -1)

        output = output + lepe
        output = self.out_proj(output)
        return output
    
    def reset_parameters(self):
        """Reset parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Full_GSA(nn.Module):
    """Full Geometry Self-Attention module."""

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * value_factor, bias=True)
        
        # Add LEPE (Learned Position Encoding) - critical component from PyTorch version
        self.lepe = DWConv2d(embed_dim * value_factor, 5, 1, 2)
        
        self.out_proj = nn.Linear(embed_dim * value_factor, embed_dim, bias=True)
        self.reset_parameters()

    def execute(self, x, rel_pos, split_or_not=False):
        """
        x: B, H, W, C
        rel_pos: relative position encoding
        """
        B, H, W, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Apply LEPE to values
        lepe = self.lepe(v)

        (sin, cos), mask = rel_pos
        assert H * W == mask.shape[3]
        
        # Apply scaling to k
        k = k * self.scaling
        
        # Reshape for multi-head attention - following PyTorch exactly
        q_reshaped = q.view(B, H, W, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (B, num_heads, H, W, key_dim)
        k_reshaped = k.view(B, H, W, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (B, num_heads, H, W, key_dim)
        
        # Apply rotary position encoding
        qr = angle_transform(q_reshaped, sin, cos)
        kr = angle_transform(k_reshaped, sin, cos)
        
        # Flatten spatial dimensions
        qr = qr.flatten(2, 3)  # (B, num_heads, H*W, key_dim)
        kr = kr.flatten(2, 3)  # (B, num_heads, H*W, key_dim)
        vr = v.reshape(B, H, W, self.num_heads, -1).permute(0, 3, 1, 2, 4).flatten(2, 3)  # (B, num_heads, H*W, head_dim)
        
        # Compute attention
        qk_mat = qr @ kr.transpose(-2, -1)
        qk_mat = qk_mat + mask
        qk_mat = F.softmax(qk_mat, dim=-1)
        output = jt.matmul(qk_mat, vr)
        
        # Reshape back
        output = output.transpose(1, 2).reshape(B, H, W, -1)  # (B, H, W, embed_dim*factor)
        
        # Add LEPE and apply output projection
        output = output + lepe
        output = self.out_proj(output)
        return output
    
    def reset_parameters(self):
        """Reset parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class FeedForwardNetwork(nn.Module):
    """Feed Forward Network module."""

    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.activation_fn = activation_fn
        self.subln = subln
        self.subconv = subconv

        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

        # Initialize ffn_layernorm based on subln parameter
        if subln:
            self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps)
        else:
            self.ffn_layernorm = None

        if subconv:
            self.dwconv = DWConv2d(ffn_dim, 3, 1, 1)

        # Implement dropout layers
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        if activation_dropout > 0:
            self.activation_dropout = nn.Dropout(activation_dropout)
        else:
            self.activation_dropout = nn.Identity()

    def reset_parameters(self):
        """Reset parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Reset ffn_layernorm if it exists
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def execute(self, x):
        """
        x: B, H, W, C
        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout(x)

        residual = x
        if self.subconv:
            x = self.dwconv(x)

        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)

        x = x + residual
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class RGBD_Block(nn.Module):
    """RGBD Block for DFormerv2."""

    def __init__(
        self,
        split_or_not,
        embed_dim,
        num_heads,
        ffn_dim,
        drop_path=0.0,
        layerscale=False,
        layer_init_values=1e-5,
        init_value=2,
        heads_range=4,
    ):
        super().__init__()
        self.split_or_not = split_or_not
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.layerscale = layerscale

        # Layer normalization - match PyTorch naming
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        # Geometry prior generator - match PyTorch naming
        self.Geo = GeoPriorGen(embed_dim, num_heads, init_value, heads_range)
        
        # Attention module - match PyTorch naming
        if split_or_not:
            self.Attention = Decomposed_GSA(embed_dim, num_heads)
        else:
            self.Attention = Full_GSA(embed_dim, num_heads)

        # Feed forward network
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        
        # CNN positional encoding - critical component from PyTorch version
        self.cnn_pos_encode = DWConv2d(embed_dim, 3, 1, 1)

        # Implement DropPath
        if drop_path > 0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        # Layer scale parameters
        if layerscale:
            self.gamma1 = jt.full((1, 1, 1, embed_dim), layer_init_values)
            self.gamma2 = jt.full((1, 1, 1, embed_dim), layer_init_values)

    def execute(self, x, x_e, split_or_not=False):
        """
        x: RGB features [B, H, W, C]
        x_e: Depth features [B, C_in, H_in, W_in]
        """
        # Apply CNN positional encoding - this is critical from PyTorch version
        x = x + self.cnn_pos_encode(x)
        
        B, H, W, C = x.shape
        depth_map = x_e # x_e is already in BCHW format
        geo_prior = self.Geo((H, W), depth_map, split_or_not)

        # Self-attention with layer scale
        if self.layerscale:
            x = x + self.drop_path(self.gamma1 * self.Attention(self.layer_norm1(x), geo_prior, split_or_not))
            x = x + self.drop_path(self.gamma2 * self.ffn(self.layer_norm2(x)))
        else:
            x = x + self.drop_path(self.Attention(self.layer_norm1(x), geo_prior, split_or_not))
            x = x + self.drop_path(self.ffn(self.layer_norm2(x)))

        return x # Return only x, not x_e


class BasicLayer(nn.Module):
    """Basic layer for DFormerv2."""

    def __init__(
        self,
        embed_dim,
        out_dim,
        depth,
        num_heads,
        init_value,
        heads_range,
        ffn_dim=96.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        split_or_not=False,
        downsample=None,
        use_checkpoint=False,
        layerscale=False,
        layer_init_values=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.split_or_not = split_or_not

        # Build blocks
        self.blocks = nn.ModuleList([
            RGBD_Block(
                split_or_not=split_or_not,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=int(embed_dim * ffn_dim) if isinstance(ffn_dim, float) else ffn_dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                layerscale=layerscale,
                layer_init_values=layer_init_values,
                init_value=init_value,
                heads_range=heads_range,
            )
            for i in range(depth)
        ])

        # Downsample layer
        if downsample is not None:
            self.downsample = downsample(embed_dim, out_dim, norm_layer)
        else:
            self.downsample = None

    def execute(self, x, x_e):
        """
        x: RGB features [B, H, W, C]
        x_e: Depth features [B, 1, H, W]
        """
        # Process through all blocks with depth features
        for block in self.blocks:
            x = block(x, x_e, split_or_not=self.split_or_not)

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x


class dformerv2(nn.Module):
    """DFormerv2 model for RGBD semantic segmentation."""

    def __init__(
        self,
        out_indices=(0, 1, 2, 3),
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        projection=1024,
        norm_cfg=None,
        layerscales=[False, False, False, False],
        layer_init_values=1e-6,
        norm_eval=True,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate
        self.norm_eval = norm_eval

        # Patch embedding - only for RGB, depth is processed differently
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=embed_dims[0])
        # Note: PyTorch version doesn't use separate depth patch embedding

        # Calculate drop path rates
        dpr = jt.linspace(0, drop_path_rate, sum(depths)).tolist()

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            split_or_not = (i != 3)  # Use decomposed attention for all layers except the last one (layer 3)
            layer = BasicLayer(
                embed_dim=embed_dims[i],
                out_dim=embed_dims[i + 1] if i < len(depths) - 1 else embed_dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                init_value=init_values[i],
                heads_range=heads_ranges[i],
                ffn_dim=int(mlp_ratios[i] * embed_dims[i]),
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                split_or_not=split_or_not,
                downsample=PatchMerging if i < len(depths) - 1 else None,
                layerscale=layerscales[i],
                layer_init_values=layer_init_values,
            )
            self.layers.append(layer)

        # Add extra normalization layers - critical component from PyTorch version
        self.extra_norms = nn.ModuleList()
        for i in range(3):  # For layers 1, 2, 3 (skip layer 0)
            self.extra_norms.append(nn.LayerNorm(embed_dims[i + 1]))

        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        if pretrained is not None:
            try:
                # Import load_checkpoint from utils
                from utils.dformer_utils import load_checkpoint

                checkpoint = load_checkpoint(self, pretrained, strict=False)
                print(f"Successfully loaded checkpoint from {pretrained}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                # Fall back to default initialization
                self.apply(self._init_weights)
        else:
            self.apply(self._init_weights)

    def no_weight_decay(self):
        """Return parameters that should not use weight decay."""
        return {'pos_embed', 'cls_token'}

    def no_weight_decay_keywords(self):
        """Return parameter keywords that should not use weight decay."""
        return {'relative_position_bias_table', 'angle', 'decay'}

    def execute(self, x, x_e):
        """
        x: RGB input [B, 3, H, W]
        x_e: Depth input [B, 1, H, W]
        """
        # RGB patch embedding
        x = self.patch_embed(x)  # [B, H/4, W/4, C]
        
        # Depth input processing - extract first channel and keep single channel
        x_e = x_e[:, 0, :, :].unsqueeze(1)  # [B, 1, H, W] - keep single channel like PyTorch

        outs = []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x_out, x = layer(x, x_e)
            
            if i in self.out_indices:
                # Apply extra normalization for layers 1, 2, 3 (skip layer 0)
                if i != 0:
                    x_out = self.extra_norms[i - 1](x_out)
                
                # Convert back to BCHW format
                out = x_out.permute(0, 3, 1, 2)
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Set training mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


def DFormerv2_S(pretrained=False, **kwargs):
    """DFormerv2 Small model."""
    model = dformerv2(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],  # Fixed to match PyTorch version (was [2, 2, 8, 2])
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        **kwargs
    )
    if pretrained:
        model.init_weights(pretrained)
    return model


def DFormerv2_B(pretrained=False, **kwargs):
    """DFormerv2 Base model."""
    model = dformerv2(
        embed_dims=[80, 160, 320, 512],
        depths=[4, 8, 25, 8],  # Fixed to match PyTorch version
        num_heads=[5, 5, 10, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[5, 5, 6, 6],  # Fixed to match PyTorch version
        mlp_ratios=[4, 4, 3, 3],
        layerscales=[False, False, True, True],  # Added from PyTorch version
        layer_init_values=1e-6,
        **kwargs
    )
    if pretrained:
        model.init_weights(pretrained)
    return model


def DFormerv2_L(pretrained=False, **kwargs):
    """DFormerv2 Large model."""
    model = dformerv2(
        embed_dims=[112, 224, 448, 640],
        depths=[4, 8, 25, 8],  # Fixed to match PyTorch version
        num_heads=[7, 7, 14, 20],
        init_values=[2, 2, 2, 2],
        heads_ranges=[6, 6, 6, 6],  # Fixed to match PyTorch version
        mlp_ratios=[4, 4, 3, 3],
        layerscales=[False, False, True, True],  # Added from PyTorch version
        layer_init_values=1e-6,
        **kwargs
    )
    if pretrained:
        model.init_weights(pretrained)
    return model 