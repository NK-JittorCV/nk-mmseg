"""
Data loader utilities for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import cv2
import jittor as jt
import numpy as np
from jittor.dataset import Dataset
import random

from utils.transforms import (
    generate_random_crop_pos,
    random_crop_pad_to_shape,
    normalize,
)


def seed_worker(worker_id):
    """Seed worker for reproducible data loading."""
    worker_seed = jt.get_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DistributedSampler:
    """Distributed sampler for Jittor."""

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            num_replicas = jt.world_size if hasattr(jt, 'world_size') else 1
        if rank is None:
            rank = jt.rank if hasattr(jt, 'rank') else 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(np.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = np.random.Generator(np.random.PCG64(seed=self.seed + self.epoch))
            indices = g.permutation(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def random_mirror(rgb, gt, modal_x):
    """Apply random horizontal mirroring to RGB, ground truth, and modal data."""
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        
        # Handle single channel modal_x
        if len(modal_x.shape) == 3 and modal_x.shape[2] == 1:
            # Single channel image, need to preserve channel dimension
            modal_x = cv2.flip(modal_x[:, :, 0], 1)
            modal_x = np.expand_dims(modal_x, axis=2)
        else:
            # Multi-channel image
            modal_x = cv2.flip(modal_x, 1)
            
    return rgb, gt, modal_x


def random_scale(rgb, gt, modal_x, scales):
    """Apply random scaling to RGB, ground truth, and modal data."""
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    
    # Handle single channel modal_x
    if len(modal_x.shape) == 3 and modal_x.shape[2] == 1:
        # Single channel image, need to preserve channel dimension
        modal_x = cv2.resize(modal_x[:, :, 0], (sw, sh), interpolation=cv2.INTER_LINEAR)
        modal_x = np.expand_dims(modal_x, axis=2)
    else:
        # Multi-channel image
        modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
    
    return rgb, gt, modal_x, scale


class TrainPre(object):
    """Training data preprocessing class."""
    
    def __init__(self, norm_mean, norm_std, sign=False, config=None):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign

    def __call__(self, rgb, gt, modal_x):
        """Apply preprocessing to training data."""
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        if self.config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, self.config.train_scale_array)

        rgb = normalize(rgb, self.norm_mean, self.norm_std)

        # Use fixed depth normalization parameters to match PyTorch version
        # This is critical for model performance!
        if len(modal_x.shape) == 3 and modal_x.shape[2] == 1:
            # Single channel depth image (DFormerv2)
            modal_x = normalize(modal_x, [0.48], [0.28])
        else:
            # Multi-channel depth image (DFormerv1) - use same parameters as PyTorch
            modal_x = normalize(modal_x, [0.48, 0.48, 0.48], [0.28, 0.28, 0.28])

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        rgb = rgb.transpose(2, 0, 1)
        modal_x = modal_x.transpose(2, 0, 1)

        return rgb, gt, modal_x


class ValPre(object):
    """Validation data preprocessing class."""
    
    def __init__(self, norm_mean, norm_std, sign=False, config=None):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign

    def __call__(self, rgb, gt, modal_x):
        """Apply preprocessing to validation data."""
        rgb = normalize(rgb, self.norm_mean, self.norm_std)

        # Use fixed depth normalization parameters to match PyTorch version
        # This is critical for model performance!
        if len(modal_x.shape) == 3 and modal_x.shape[2] == 1:
            # Single channel depth image (DFormerv2)
            modal_x = normalize(modal_x, [0.48], [0.28])
        else:
            # Multi-channel depth image (DFormerv1) - use same parameters as PyTorch
            modal_x = normalize(modal_x, [0.48, 0.48, 0.48], [0.28, 0.28, 0.28])

        rgb = rgb.transpose(2, 0, 1)
        modal_x = modal_x.transpose(2, 0, 1)

        return rgb, gt, modal_x


def get_train_loader(engine, dataset_cls, config):
    """Create training data loader."""
    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
        "dataset_name": config.dataset_name,
        "backbone": config.backbone,
    }

    train_preprocess = TrainPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config)

    train_dataset = dataset_cls(
        data_setting,
        "train",
        train_preprocess,
        config.batch_size * config.niters_per_epoch,
    )

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine is not None and hasattr(engine, 'distributed') and engine.distributed:
        train_sampler = DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    # Reduce num_workers for training to prevent issues
    train_num_workers = min(2, config.num_workers)  # Use at most 2 workers for training

    train_loader = train_dataset.set_attrs(
        batch_size=batch_size,
        shuffle=is_shuffle,
        num_workers=train_num_workers,
        drop_last=True
    )

    return train_loader, train_sampler


def get_val_loader(engine, dataset_cls, config, val_batch_size=None):
    """Create validation data loader."""
    if val_batch_size is None:
        val_batch_size = config.batch_size

    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
        "dataset_name": config.dataset_name,
        "backbone": config.backbone,
    }

    val_preprocess = ValPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config)

    val_dataset = dataset_cls(data_setting, "val", val_preprocess)

    val_sampler = None
    is_shuffle = False
    batch_size = val_batch_size

    if engine is not None and hasattr(engine, 'distributed') and engine.distributed:
        val_sampler = DistributedSampler(val_dataset)
        batch_size = val_batch_size // engine.world_size
        is_shuffle = False

    # Set num_workers to 0 for validation to prevent deadlocks completely
    val_num_workers = 0  # Use single-threaded loading for validation to avoid deadlocks

    val_loader = val_dataset.set_attrs(
        batch_size=batch_size,
        shuffle=is_shuffle,
        num_workers=val_num_workers,
        drop_last=False
    )

    return val_loader, val_sampler
