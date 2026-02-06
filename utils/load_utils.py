"""
Model loading utilities for DFormer Jittor implementation.
"""

import jittor as jt
import re
import os
import pickle


def get_dist_info():
    """Get distributed training info."""
    # TODO: Add MPI support for distributed training
    rank = 0
    world_size = 1
    return rank, world_size


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state dict to Jittor module.
    
    Args:
        module: Jittor module
        state_dict: State dictionary
        strict: Whether to strictly enforce that the keys match
        logger: Logger instance
    """
    unexpected_keys = []
    missing_keys = []
    
    # Get module parameters and buffers
    module_keys = set()
    for name, _ in module.named_parameters():
        module_keys.add(name)
    
    # Load parameters
    for name, param in state_dict.items():
        if name in module_keys:
            try:
                # Get the parameter from module
                target_param = module
                for attr in name.split('.'):
                    target_param = getattr(target_param, attr)
                
                # Convert to Jittor array if needed
                if not isinstance(param, jt.Var):
                    if hasattr(param, 'detach'):  # PyTorch tensor
                        param = jt.array(param.detach().cpu().numpy())
                    else:
                        param = jt.array(param)
                
                # Assign the parameter
                target_param.assign(param)
                
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to load parameter {name}: {e}")
                else:
                    print(f"Warning: Failed to load parameter {name}: {e}")
        else:
            unexpected_keys.append(name)
    
    # Check for missing keys
    for name in module_keys:
        if name not in state_dict:
            # Skip batch norm tracking parameters
            if "num_batches_tracked" not in name:
                missing_keys.append(name)
    
    # Report results
    rank, _ = get_dist_info()
    if rank == 0:
        if unexpected_keys:
            msg = f"Unexpected keys in state dict: {unexpected_keys}"
            if strict:
                raise RuntimeError(msg)
            else:
                print(f"Warning: {msg}")
        
        if missing_keys:
            msg = f"Missing keys in state dict: {missing_keys}"
            if strict:
                raise RuntimeError(msg)
            else:
                print(f"Warning: {msg}")
    
    return missing_keys, unexpected_keys


def load_pretrain(model, filename, strict=False, revise_keys=[(r"^module\.", "")]):
    """Load pretrained model weights.
    
    Args:
        model: Jittor model
        filename: Path to checkpoint file
        strict: Whether to strictly enforce key matching
        revise_keys: List of (pattern, replacement) for key revision
        
    Returns:
        checkpoint: Loaded checkpoint dictionary
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")
    
    # Load checkpoint
    if filename.endswith('.pkl') or filename.endswith('.pth'):
        try:
            # Try to load as Jittor checkpoint
            checkpoint = jt.load(filename)
        except:
            # Fall back to pickle
            with open(filename, 'rb') as f:
                checkpoint = pickle.load(f)
    else:
        raise ValueError(f"Unsupported checkpoint format: {filename}")
    
    # Extract state dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Revise keys if needed
    for pattern, replacement in revise_keys:
        state_dict = {re.sub(pattern, replacement, k): v for k, v in state_dict.items()}
    
    # Load state dict
    missing_keys, unexpected_keys = load_state_dict(model, state_dict, strict)
    
    return checkpoint


def save_checkpoint(model, filename, epoch=None, optimizer=None, **kwargs):
    """Save model checkpoint.
    
    Args:
        model: Jittor model
        filename: Save path
        epoch: Current epoch
        optimizer: Optimizer state
        **kwargs: Additional items to save
    """
    checkpoint = {
        'state_dict': {name: param for name, param in model.named_parameters()},
        'epoch': epoch,
        **kwargs
    }
    
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    
    # Create directory if not exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save checkpoint
    jt.save(checkpoint, filename)
    print(f"Checkpoint saved to: {filename}")


def convert_pytorch_to_jittor(pytorch_path, jittor_path):
    """Convert PyTorch checkpoint to Jittor format.
    
    Args:
        pytorch_path: Path to PyTorch checkpoint
        jittor_path: Path to save Jittor checkpoint
    """
    import torch
    
    # Load PyTorch checkpoint
    checkpoint = torch.load(pytorch_path, map_location='cpu')
    
    # Convert tensors to numpy arrays for Jittor
    if isinstance(checkpoint, dict):
        converted = {}
        for key, value in checkpoint.items():
            if isinstance(value, torch.Tensor):
                converted[key] = value.detach().cpu().numpy()
            elif isinstance(value, dict):
                converted[key] = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                                 for k, v in value.items()}
            else:
                converted[key] = value
        checkpoint = converted
    
    # Save as Jittor checkpoint
    os.makedirs(os.path.dirname(jittor_path), exist_ok=True)
    with open(jittor_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Converted PyTorch checkpoint to Jittor format: {jittor_path}") 