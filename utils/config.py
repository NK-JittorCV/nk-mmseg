"""
Configuration management utilities for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import os
import os.path as osp
import importlib.util
import argparse
from easydict import EasyDict as edict


def load_config(config_path):
    """Load configuration from file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        edict: Configuration object
    """
    if not osp.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config module
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Get config object
    if hasattr(config_module, 'C'):
        config = config_module.C
    elif hasattr(config_module, 'config'):
        config = config_module.config
    else:
        raise AttributeError("Config file must define 'C' or 'config' object")
    
    return config


def merge_config(base_config, override_config):
    """Merge two configuration objects.
    
    Args:
        base_config (edict): Base configuration
        override_config (dict): Override configuration
        
    Returns:
        edict: Merged configuration
    """
    merged = edict(base_config.copy())
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def update_config_from_args(config, args):
    """Update configuration from command line arguments.
    
    Args:
        config (edict): Configuration object
        args (argparse.Namespace): Command line arguments
        
    Returns:
        edict: Updated configuration
    """
    # Update common parameters
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.batch_size = args.batch_size
    
    if hasattr(args, 'lr') and args.lr is not None:
        config.lr = args.lr
    
    if hasattr(args, 'epochs') and args.epochs is not None:
        config.nepochs = args.epochs
    
    if hasattr(args, 'num_workers') and args.num_workers is not None:
        config.num_workers = args.num_workers
    
    if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    
    if hasattr(args, 'log_dir') and args.log_dir is not None:
        config.log_dir = args.log_dir
    
    return config


def create_config_parser():
    """Create argument parser for configuration override.
    
    Returns:
        argparse.ArgumentParser: Argument parser
    """
    parser = argparse.ArgumentParser(description='DFormer Configuration')
    
    # Model arguments
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--backbone', help='backbone model name')
    parser.add_argument('--pretrained', help='pretrained model path')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, help='batch size')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--num-workers', type=int, help='number of workers')
    
    # Dataset arguments
    parser.add_argument('--dataset', help='dataset name')
    parser.add_argument('--data-root', help='dataset root directory')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', help='checkpoint directory')
    parser.add_argument('--log-dir', help='log directory')
    
    # Device arguments
    parser.add_argument('--device', default='cuda', help='device to use')
    
    return parser


def validate_config(config):
    """Validate configuration object.
    
    Args:
        config (edict): Configuration object
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = [
        'dataset_name', 'num_classes', 'backbone',
        'batch_size', 'lr', 'nepochs'
    ]
    
    for field in required_fields:
        if not hasattr(config, field):
            raise ValueError(f"Missing required config field: {field}")
    
    # Validate numeric fields
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    if config.lr <= 0:
        raise ValueError("lr must be positive")
    
    if config.nepochs <= 0:
        raise ValueError("nepochs must be positive")
    
    if config.num_classes <= 0:
        raise ValueError("num_classes must be positive")


def print_config(config, title="Configuration"):
    """Print configuration in a formatted way.
    
    Args:
        config (edict): Configuration object
        title (str): Title for the configuration
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    def _print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                _print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    _print_dict(config)
    print()


def save_config(config, save_path):
    """Save configuration to file.
    
    Args:
        config (edict): Configuration object
        save_path (str): Path to save configuration
    """
    import json
    
    # Convert edict to regular dict for JSON serialization
    def _convert_edict(obj):
        if isinstance(obj, edict):
            return {k: _convert_edict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_edict(item) for item in obj]
        else:
            return obj
    
    config_dict = _convert_edict(config)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def get_model_configs():
    """Get available model configurations.
    
    Returns:
        dict: Dictionary of available configurations
    """
    configs = {
        'nyudepthv2': {
            'DFormer_Tiny': 'local_configs/NYUDepthv2/DFormer_Tiny.py',
            'DFormer_Small': 'local_configs/NYUDepthv2/DFormer_Small.py',
            'DFormer_Base': 'local_configs/NYUDepthv2/DFormer_Base.py',
            'DFormer_Large': 'local_configs/NYUDepthv2/DFormer_Large.py',
            'DFormerv2_S': 'local_configs/NYUDepthv2/DFormerv2_S.py',
            'DFormerv2_B': 'local_configs/NYUDepthv2/DFormerv2_B.py',
            'DFormerv2_L': 'local_configs/NYUDepthv2/DFormerv2_L.py',
        },
        'sunrgbd': {
            'DFormer_Tiny': 'local_configs/SUNRGBD/DFormer_Tiny.py',
            'DFormer_Small': 'local_configs/SUNRGBD/DFormer_Small.py',
            'DFormer_Base': 'local_configs/SUNRGBD/DFormer_Base.py',
            'DFormer_Large': 'local_configs/SUNRGBD/DFormer_Large.py',
            'DFormerv2_S': 'local_configs/SUNRGBD/DFormerv2_S.py',
            'DFormerv2_B': 'local_configs/SUNRGBD/DFormerv2_B.py',
            'DFormerv2_L': 'local_configs/SUNRGBD/DFormerv2_L.py',
        }
    }
    
    return configs


def get_config_by_name(dataset, model):
    """Get configuration file path by dataset and model name.
    
    Args:
        dataset (str): Dataset name
        model (str): Model name
        
    Returns:
        str: Configuration file path
    """
    configs = get_model_configs()
    
    if dataset.lower() not in configs:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if model not in configs[dataset.lower()]:
        raise ValueError(f"Unknown model: {model} for dataset: {dataset}")
    
    return configs[dataset.lower()][model]


def setup_config(config_path, args=None):
    """Setup configuration from file and arguments.
    
    Args:
        config_path (str): Path to configuration file
        args (argparse.Namespace, optional): Command line arguments
        
    Returns:
        edict: Setup configuration
    """
    # Load base configuration
    config = load_config(config_path)
    
    # Update from arguments if provided
    if args is not None:
        config = update_config_from_args(config, args)
    
    # Validate configuration
    validate_config(config)
    
    return config
