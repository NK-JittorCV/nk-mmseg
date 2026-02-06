"""
Base configuration for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C

C.seed = 12345

# Directory paths
C.root_dir = "datasets"
C.abs_dir = osp.realpath(".")

# Training configuration
C.lr = 0.01
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.0005
C.batch_size = 8
C.nepochs = 500
C.niters_per_epoch = 100
C.num_workers = 8
C.train_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

# Model configuration
C.backbone = 'mit_b2'
C.pretrained_model = None
C.norm_layer = 'BN'
C.bn_eps = 1e-5
C.bn_momentum = 0.1

# Loss configuration
C.loss_type = 'CrossEntropyLoss'
C.ignore_label = 255

# Evaluation configuration
C.eval_stride_rate = 2/3
C.eval_scale_array = [1.0]
C.eval_flip = False
C.eval_crop_size = [480, 640]

# Optimizer configuration
C.optimizer = 'SGD'
C.use_warmup = True
C.warmup_iters = 1500

# Checkpoint configuration
C.checkpoint_start_epoch = 0
C.checkpoint_step = 50

# Logging configuration
C.log_dir = './log'
C.tb_dir = './tb'

# Device configuration
C.device = 'cuda'
C.use_cuda = True
