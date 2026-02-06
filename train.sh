#!/bin/bash

# DFormer Jittor Training Script
# Adapted from PyTorch version for Jittor framework

GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="1"

# Memory optimization environment variables
export JT_SYNC=0
export JT_LAZY=1
export CUDA_LAUNCH_BLOCKING=0

# For single GPU testing, use GPUS=1 and CUDA_VISIBLE_DEVICES="0"
# GPUS=1
# export CUDA_VISIBLE_DEVICES="0"

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    python utils/train.py \
    --config=local_configs.NYUDepthv2.DFormerv2_S \
    --gpus=$GPUS \
    --no-sliding \
    --no-syncbn \
    --no-mst \
    --no-amp \
    --no-val_amp \
    --no-pad_SUNRGBD \
    --no-use_seed

# Available configurations for DFormers on NYUDepthv2:
# local_configs.NYUDepthv2.DFormer_Large
# local_configs.NYUDepthv2.DFormer_Base
# local_configs.NYUDepthv2.DFormer_Small
# local_configs.NYUDepthv2.DFormer_Tiny
# local_configs.NYUDepthv2.DFormerv2_S
# local_configs.NYUDepthv2.DFormerv2_B
# local_configs.NYUDepthv2.DFormerv2_L

# Available configurations for DFormers on SUNRGBD:
# local_configs.SUNRGBD.DFormer_Large
# local_configs.SUNRGBD.DFormer_Base
# local_configs.SUNRGBD.DFormer_Small
# local_configs.SUNRGBD.DFormer_Tiny
# local_configs.SUNRGBD.DFormerv2_S
# local_configs.SUNRGBD.DFormerv2_B
# local_configs.SUNRGBD.DFormerv2_L
