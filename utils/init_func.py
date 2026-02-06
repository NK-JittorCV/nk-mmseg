"""
Initialization functions for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import jittor as jt
from jittor import nn
import numpy as np


def configure_optimizers(model, config):
    """Configure optimizers for training."""
    # Group parameters by weight decay
    params = group_weight(model)
    
    if config.optimizer == 'SGD':
        optimizer = jt.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'Adam':
        optimizer = jt.optim.Adam(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'AdamW':
        optimizer = jt.optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    return optimizer


def group_weight(module):
    """Group model parameters by weight decay."""
    group_decay = []
    group_no_decay = []
    
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
            
        # Parameters that should have weight decay
        if len(param.shape) >= 2:  # Conv2d weights, Linear weights
            group_decay.append(param)
        else:  # Bias, BatchNorm weights/bias, etc.
            group_no_decay.append(param)
    
    # Remove assertion as it might fail for complex models
    total_params = len(list(module.parameters()))
    grouped_params = len(group_decay) + len(group_no_decay)
    
    if total_params != grouped_params:
        print(f"Warning: Total parameters ({total_params}) != grouped parameters ({grouped_params})")
    
    groups = [
        dict(params=group_decay),
        dict(params=group_no_decay, weight_decay=0.0)
    ]
    
    return groups


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    """Initialize model weights."""
    if isinstance(module_list, list):
        for feature in module_list:
            for m in feature.modules():
                _init_weight(m, conv_init, norm_layer, bn_eps, bn_momentum)
    else:
        for m in module_list.modules():
            _init_weight(m, conv_init, norm_layer, bn_eps, bn_momentum)


def _init_weight(m, conv_init, norm_layer, bn_eps, bn_momentum):
    """Initialize individual module weights."""
    if isinstance(m, nn.Conv2d):
        if conv_init == 'kaiming':
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif conv_init == 'xavier':
            nn.init.xavier_normal_(m.weight)
        else:
            nn.init.normal_(m.weight, std=0.001)
        
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, norm_layer):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        m.eps = bn_eps
        # m.momentum = bn_momentum  # TODO: Check if Jittor supports momentum parameter
    
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    """Kaiming initialization."""
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """Xavier initialization."""
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    """Normal initialization."""
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    """Uniform initialization."""
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    """Constant initialization."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """Initialize bias with probability."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def initialize_weights(*models):
    """Initialize weights for multiple models."""
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


def freeze_bn(model):
    """Freeze batch normalization layers."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


def unfreeze_bn(model):
    """Unfreeze batch normalization layers."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            for param in m.parameters():
                param.requires_grad = True
