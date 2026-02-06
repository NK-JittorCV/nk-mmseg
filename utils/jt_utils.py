"""
Jittor utilities for DFormer implementation
Adapted from PyTorch pyt_utils for Jittor framework
"""

import os
import sys
import time
import random
import argparse
import logging
from collections import OrderedDict, defaultdict

import jittor as jt
import numpy as np


def ensure_dir(path):
    """Ensure directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def link_file(src, target):
    """Create symbolic link."""
    if os.path.islink(target) or os.path.exists(target):
        os.remove(target)
    os.symlink(src, target)


def extant_file(x):
    """Check if file exists."""
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def parse_devices(input_devices):
    """Parse device string."""
    if input_devices.endswith('*'):
        devices = list(range(jt.flags.use_cuda))
        return devices
    else:
        devices = [int(d) for d in input_devices.split(',')]
        return devices


def load_pytorch_weights(model, pytorch_checkpoint_path):
    """Load PyTorch weights into Jittor model with precise conversion."""
    try:
        # Try to use precise converter
        from precise_weight_converter import convert_weights_precisely
        print("Using precise weight converter...")
        return convert_weights_precisely()
    except ImportError:
        print("Precise converter not available, using basic conversion...")
        return load_pytorch_weights_basic(model, pytorch_checkpoint_path)
    except Exception as e:
        print(f"Precise converter failed: {e}, falling back to basic conversion...")
        return load_pytorch_weights_basic(model, pytorch_checkpoint_path)


def load_pytorch_weights_basic(model, pytorch_checkpoint_path):
    """Basic PyTorch weights loading (fallback)."""
    import torch

    print(f"Loading PyTorch weights from: {pytorch_checkpoint_path}")

    # Load PyTorch checkpoint
    try:
        pytorch_checkpoint = torch.load(pytorch_checkpoint_path, map_location='cpu')
        if 'model' in pytorch_checkpoint:
            pytorch_state_dict = pytorch_checkpoint['model']
        elif 'state_dict' in pytorch_checkpoint:
            pytorch_state_dict = pytorch_checkpoint['state_dict']
        else:
            pytorch_state_dict = pytorch_checkpoint
    except Exception as e:
        print(f"Failed to load PyTorch checkpoint: {e}")
        return model

    # Get Jittor model state dict
    jittor_state_dict = model.state_dict()

    # Convert compatible weights
    converted_count = 0
    skipped_count = 0
    total_pytorch_params = len(pytorch_state_dict)

    print(f"Converting {total_pytorch_params} PyTorch parameters to Jittor format...")

    # Parameter name mapping from PyTorch to Jittor
    param_mapping = {
        # Decoder head mappings
        'decode_head.conv_seg.weight': 'decode_head.cls_seg.weight',
        'decode_head.conv_seg.bias': 'decode_head.cls_seg.bias',

        # BatchNorm to LayerNorm/GroupNorm mappings for decoder head
        'decode_head.squeeze.bn.weight': 'decode_head.squeeze.norm.weight',
        'decode_head.squeeze.bn.bias': 'decode_head.squeeze.norm.bias',
        'decode_head.squeeze.bn.running_mean': 'decode_head.squeeze.norm.running_mean',
        'decode_head.squeeze.bn.running_var': 'decode_head.squeeze.norm.running_var',

        'decode_head.hamburger.ham_out.bn.weight': 'decode_head.hamburger.ham_out.norm.weight',
        'decode_head.hamburger.ham_out.bn.bias': 'decode_head.hamburger.ham_out.norm.bias',
        'decode_head.hamburger.ham_out.bn.running_mean': 'decode_head.hamburger.ham_out.norm.running_mean',
        'decode_head.hamburger.ham_out.bn.running_var': 'decode_head.hamburger.ham_out.norm.running_var',

        'decode_head.align.bn.weight': 'decode_head.align.norm.weight',
        'decode_head.align.bn.bias': 'decode_head.align.norm.bias',
        'decode_head.align.bn.running_mean': 'decode_head.align.norm.running_mean',
        'decode_head.align.bn.running_var': 'decode_head.align.norm.running_var',
    }

    # Function to convert parameter names
    def convert_param_name(pytorch_key):
        # Apply direct mappings first
        if pytorch_key in param_mapping:
            return param_mapping[pytorch_key]

        # Convert gamma_1 -> gamma1, gamma_2 -> gamma2 for DFormerv2 LayerScale
        if 'gamma_1' in pytorch_key:
            return pytorch_key.replace('gamma_1', 'gamma1')
        elif 'gamma_2' in pytorch_key:
            return pytorch_key.replace('gamma_2', 'gamma2')

        # Skip backbone norm parameters that don't exist in Jittor model
        if any(x in pytorch_key for x in ['backbone.norm0', 'backbone.norm1', 'backbone.norm2', 'backbone.norm3']):
            return None  # Signal to skip this parameter

        return pytorch_key

    # Convert weights with exact key and shape matches
    for pytorch_key, pytorch_tensor in pytorch_state_dict.items():
        # Skip num_batches_tracked parameters (not needed in Jittor)
        if 'num_batches_tracked' in pytorch_key:
            skipped_count += 1
            continue

        # Map parameter names if needed
        jittor_key = convert_param_name(pytorch_key)

        # Skip parameters that should not be converted (None return)
        if jittor_key is None:
            skipped_count += 1
            if skipped_count <= 10:  # Print first few skipped keys for debugging
                print(f"⚠ Skipped (not in Jittor model): {pytorch_key}")
            continue

        # Check if the key exists in Jittor model
        if jittor_key in jittor_state_dict:
            try:
                # Convert PyTorch tensor to numpy then to Jittor
                numpy_tensor = pytorch_tensor.detach().cpu().numpy()
                jittor_tensor = jt.array(numpy_tensor)

                # Check shape compatibility
                expected_shape = jittor_state_dict[jittor_key].shape
                if tuple(jittor_tensor.shape) == tuple(expected_shape):
                    jittor_state_dict[jittor_key] = jittor_tensor
                    converted_count += 1
                    if converted_count <= 10 or jittor_key != pytorch_key:  # Print first 10 and all mapped keys
                        if jittor_key != pytorch_key:
                            print(f"✓ Converted (mapped): {pytorch_key} -> {jittor_key}")
                        else:
                            print(f"✓ Converted: {pytorch_key}")
                else:
                    print(f"✗ Shape mismatch for {jittor_key}: expected {expected_shape}, got {jittor_tensor.shape}")
                    skipped_count += 1
            except Exception as e:
                print(f"✗ Error converting {pytorch_key} -> {jittor_key}: {e}")
                skipped_count += 1
        else:
            skipped_count += 1

    # Load the updated state dict
    try:
        model.load_state_dict(jittor_state_dict)
        print(f"Successfully converted and loaded {converted_count}/{total_pytorch_params} parameters")
        print(f"Skipped {skipped_count} parameters (incompatible or not needed)")
        if converted_count > 0:
            print("✓ Model weights loaded successfully!")
        else:
            print("⚠ Warning: No weights were loaded - model uses random initialization")
        return model
    except Exception as e:
        print(f"Error loading converted state dict: {e}")
        return model


def load_model(model, model_file, is_restore=False):
    """Load model from checkpoint file."""
    t_start = time.time()

    if model_file is None:
        return model

    # Check if it's a PyTorch checkpoint
    if isinstance(model_file, str) and model_file.endswith('.pth'):
        try:
            # Try to load as PyTorch weights first
            return load_pytorch_weights(model, model_file)
        except Exception as e:
            print(f"Failed to load as PyTorch weights, trying Jittor format: {e}")

    # Original Jittor loading logic
    if isinstance(model_file, str):
        checkpoint = jt.load(model_file)
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint.keys():
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = model_file

    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        print('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        print('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    print("Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
        t_ioend - t_start, t_end - t_ioend))

    return model


def save_model(model, model_file):
    """Save model to checkpoint file."""
    ensure_dir(os.path.dirname(model_file))
    jt.save(model.state_dict(), model_file)
    print("Model saved to {}".format(model_file))


def all_reduce_tensor(tensor, op=None, world_size=1):
    """All reduce tensor across processes."""
    if jt.world_size > 1:
        return jt.distributed.all_reduce(tensor)
    return tensor


def reduce_tensor(tensor):
    """Reduce tensor across processes."""
    # For single GPU, just return the tensor
    return tensor


def get_world_size():
    """Get world size."""
    return 1


def get_rank():
    """Get current rank."""
    return 0


def is_main_process():
    """Check if current process is main process."""
    return True


def synchronize():
    """Synchronize all processes."""
    pass


def time_synchronized():
    """Get synchronized time."""
    jt.sync_all()
    return time.time()


class AverageMeter(object):
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Timer utility."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self, average=True):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def count_parameters(model):
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def format_time(seconds):
    """Format time in seconds to readable string."""
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
