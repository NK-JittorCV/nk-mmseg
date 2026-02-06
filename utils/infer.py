"""
DFormer Jittor Inference Script
Adapted from PyTorch version for Jittor framework

This script provides inference functionality for DFormer models using Jittor.
It supports both single image and batch inference with various model configurations.
"""

import argparse
import importlib
import os
import random
import sys
import time
from importlib import import_module

import numpy as np
import jittor as jt
from jittor import nn
try:
    from tensorboardX import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    print("Warning: tensorboardX not found, tensorboard logging disabled")
    SummaryWriter = None
    HAS_TENSORBOARD = False
from tqdm import tqdm

# Add project root to Python path to ensure proper imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.builder import EncoderDecoder as segmodel
from utils.dataloader.dataloader import ValPre, get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.init_func import group_weight, init_weight
from utils.lr_policy import WarmUpPolyLR
from utils.jt_utils import all_reduce_tensor, ensure_dir, link_file, load_model, parse_devices
from utils.val_mm import evaluate, evaluate_msf

# Set random seeds for reproducibility
# SEED=1
# np.random.seed(SEED)
# jt.set_global_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", help="used gpu number")
# parser.add_argument('-d', '--devices', default='0,1', type=str)
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", default=0)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
parser.add_argument("--save_path", default=None)
parser.add_argument("--checkpoint_dir")
parser.add_argument("--continue_fpath")
# Additional parameters for compatibility with infer.sh
parser.add_argument("--multi_scale", default=False, action="store_true", help="Enable multi-scale inference")
parser.add_argument("--scales", nargs='+', type=float, default=[1.0], help="Scales for multi-scale inference") 
parser.add_argument("--flip", default=False, action="store_true", help="Enable flip augmentation")
parser.add_argument("--eval_only", default=False, action="store_true", help="Only run evaluation, no saving")
# parser.add_argument('--save_path', '-p', default=None)
logger = get_logger()

# os.environ['MASTER_PORT'] = '169710'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config = getattr(import_module(args.config), "C")
    config.pad = False  # Do not pad when inference
    if "x_modal" not in config:
        config["x_modal"] = "d"
    
    # Enable Jittor CUDA if available
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
        logger.warning("CUDA not available, using CPU")

    val_loader, val_sampler = get_val_loader(engine, RGBXDataset, config, int(args.gpus))
    print(len(val_loader))

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        if HAS_TENSORBOARD and hasattr(config, 'tb_dir') and config.tb_dir:
            tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
            generate_tb_dir = config.tb_dir + "/tb"
            tb = SummaryWriter(log_dir=tb_dir)
            engine.link_tb(tb_dir, generate_tb_dir)
        else:
            logger.info("TensorBoard logging disabled (no tensorboardX or tb_dir not configured)")

    if engine.distributed:
        # Use SyncBatchNorm for distributed training in Jittor
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    model = segmodel(cfg=config, norm_layer=BatchNorm2d)
    
    # Load model weights - Jittor can directly load PyTorch checkpoints
    if args.continue_fpath:
        logger.info(f"Loading checkpoint from {args.continue_fpath}")
        try:
            # Load PyTorch checkpoint and get the state_dict
            import torch
            checkpoint = torch.load(args.continue_fpath, map_location='cpu')

            # In case the checkpoint is nested, e.g., {'model': state_dict}
            if 'model' in checkpoint:
                weight = checkpoint['model']
            else:
                weight = checkpoint

            # Create key mapping for PyTorch to Jittor conversion
            def convert_pytorch_keys_to_jittor(pytorch_state_dict):
                """Convert PyTorch checkpoint keys to Jittor model keys."""
                jittor_state_dict = {}

                for k, v in pytorch_state_dict.items():
                    new_key = k

                    # Convert PyTorch tensor to numpy array for Jittor
                    if hasattr(v, 'detach'):
                        v = v.detach().cpu().numpy()

                    # Key mappings for decode_head
                    if 'decode_head.conv_seg' in k:
                        new_key = k.replace('decode_head.conv_seg', 'decode_head.cls_seg')
                    elif 'decode_head.squeeze.bn' in k:
                        new_key = k.replace('decode_head.squeeze.bn', 'decode_head.squeeze.norm')
                    elif 'decode_head.hamburger.ham_out.bn' in k:
                        new_key = k.replace('decode_head.hamburger.ham_out.bn', 'decode_head.hamburger.ham_out.norm')
                    elif 'decode_head.align.bn' in k:
                        new_key = k.replace('decode_head.align.bn', 'decode_head.align.norm')

                    # Handle DFormerv2 LayerScale parameters: gamma_1 -> gamma1, gamma_2 -> gamma2
                    if 'gamma_1' in new_key:
                        new_key = new_key.replace('gamma_1', 'gamma1')
                    elif 'gamma_2' in new_key:
                        new_key = new_key.replace('gamma_2', 'gamma2')

                    # Skip num_batches_tracked keys as they are not needed in Jittor
                    if 'num_batches_tracked' in k:
                        continue

                    # Skip backbone norm keys that don't exist in Jittor model
                    if any(x in k for x in ['backbone.norm0', 'backbone.norm1', 'backbone.norm2', 'backbone.norm3']):
                        continue

                    jittor_state_dict[new_key] = v

                return jittor_state_dict

            # Convert PyTorch keys to Jittor keys
            converted_weight = convert_pytorch_keys_to_jittor(weight)

            # Non-strict loading of parameters
            model_dict = model.state_dict()

            # Filter checkpoint weights to match model keys
            load_dict = {k: v for k, v in converted_weight.items() if k in model_dict}

            # Convert numpy arrays to Jittor tensors and load
            jittor_load_dict = {}
            for k, v in load_dict.items():
                if isinstance(v, np.ndarray):
                    jittor_load_dict[k] = jt.array(v)
                else:
                    jittor_load_dict[k] = v

            # Load parameters into model
            model.load_parameters(jittor_load_dict)

            missing_keys = [k for k in model_dict.keys() if k not in converted_weight]
            unexpected_keys = [k for k in converted_weight.keys() if k not in model_dict]

            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

            logger.info("Model weights loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.warning("Continuing with random initialization")

    if engine.distributed:
        logger.info(".............distributed training.............")
        if jt.has_cuda:
            model.cuda()
            # Note: Jittor handles distributed training differently than PyTorch
            # The model doesn't need explicit DistributedDataParallel wrapping
        else:
            logger.warning("CUDA not available for distributed training")
    else:
        if jt.has_cuda:
            model.cuda()

    engine.register_state(dataloader=val_loader, model=model)

    logger.info("Begin testing:")
    best_miou = 0.0
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
        "class_names": config.class_names,
    }
    
    all_dev = [0]

    if engine.distributed:
        print("Multi GPU test")
        with jt.no_grad():
            model.eval()
            # Set all models in evaluation mode
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    
            # Use same parameters as PyTorch version
            eval_scales = [0.5, 0.75, 1.0, 1.25, 1.5] if args.multi_scale else [1.0]  # Single scale for faster inference
            eval_flip = True if args.flip else False  # Disable flip by default for faster inference
            eval_save_dir = args.save_path

            all_metrics = evaluate_msf(
                model,
                val_loader,
                config=config,
                scales=eval_scales,
                flip=eval_flip,
                sliding=False,  # Disable sliding for faster inference
                engine=engine,
                save_dir=eval_save_dir,
            )
            
            if engine.local_rank == 0:
                # Handle distributed metrics aggregation like PyTorch version
                if isinstance(all_metrics, list):
                    # Distributed case - aggregate metrics
                    metric = all_metrics[0]
                    for other_metric in all_metrics[1:]:
                        metric.update_hist(other_metric.hist)
                else:
                    metric = all_metrics

                ious, miou = metric.compute_iou()
                acc, macc = metric.compute_pixel_acc()
                f1, mf1 = metric.compute_f1()
                print(miou, "---------")
    else:
        with jt.no_grad():
            model.eval()
            # Set all BatchNorm layers in evaluation mode
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    
            # Use same parameters as PyTorch version
            eval_scales = [0.5, 0.75, 1.0, 1.25, 1.5] if args.multi_scale else [1.0]  # Single scale for faster inference
            eval_flip = True if args.flip else False  # Disable flip by default for faster inference
            eval_save_dir = args.save_path

            # Run multi-scale evaluation
            metric = evaluate_msf(
                model,
                val_loader,
                config=config,
                scales=eval_scales,
                flip=eval_flip,
                sliding=False,  # Disable sliding for faster inference
                engine=engine,
                save_dir=eval_save_dir,
            )

            ious, miou = metric.compute_iou()
            acc, macc = metric.compute_pixel_acc()
            f1, mf1 = metric.compute_f1()
            print("miou", miou)