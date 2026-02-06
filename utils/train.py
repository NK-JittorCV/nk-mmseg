#!/usr/bin/env python3
"""
DFormer Jittor Training Script
Adapted from PyTorch version for Jittor framework
"""

import argparse
import datetime
import os
import pprint
import random
import time
from importlib import import_module

import numpy as np
import os
import sys

os.environ.setdefault("use_cutlass", "0")

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR in sys.path:
    sys.path.remove(_CUR_DIR)

import jittor as jt
from jittor import nn

if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from models.builder import EncoderDecoder as segmodel
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.init_func import configure_optimizers, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.val_mm import evaluate, evaluate_msf
from utils.metric import SegmentationMetric
from utils.jt_utils import all_reduce_tensor


class GpuTimer:
    def __init__(self, beta=0.6):
        self.start_time = None
        self.stop_time = None
        self.mean_time = None
        self.beta = beta
        self.first_call = True

    def start(self):
        # Remove unnecessary sync for performance
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            print("Use start() before stop(). ")
            return
        jt.sync_all(True)  # Keep sync only at stop for accurate timing
        self.stop_time = time.perf_counter()
        elapsed = self.stop_time - self.start_time
        self.start_time = None
        if self.first_call:
            self.mean_time = elapsed
            self.first_call = False
        else:
            self.mean_time = self.beta * self.mean_time + (1 - self.beta) * elapsed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def is_eval(epoch, config):
    # Disable evaluation in early training to avoid memory issues
    if epoch <= 5:
        return False;  # No evaluation in first 5 epochs
    elif epoch <= 20:
        return epoch % 5 == 0  # Evaluate every 5 epochs in early training
    elif epoch <= 50:
        return epoch % 10 == 0  # Evaluate every 10 epochs in mid training
    else:
        return epoch > int(config.checkpoint_start_epoch) or epoch % 25 == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--gpus", default=1, type=int, help="used gpu number")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--show_image", "-s", default=False, action="store_true")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--continue_fpath")
    parser.add_argument("--sliding", default=False, action="store_true")
    parser.add_argument("--no-sliding", dest="sliding", action="store_false")
    parser.add_argument("--syncbn", default=True, action="store_true")
    parser.add_argument("--no-syncbn", dest="syncbn", action="store_false")
    parser.add_argument("--mst", default=True, action="store_true")
    parser.add_argument("--no-mst", dest="mst", action="store_false")
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--val_amp", default=False, action="store_true")
    parser.add_argument("--no-val_amp", dest="val_amp", action="store_false")
    parser.add_argument("--pad_SUNRGBD", default=False, action="store_true")
    parser.add_argument("--no-pad_SUNRGBD", dest="pad_SUNRGBD", action="store_false")
    parser.add_argument("--use_seed", default=True, action="store_true")
    parser.add_argument("--no-use_seed", dest="use_seed", action="store_false")
    parser.add_argument("--local-rank", default=0, type=int)

    args = parser.parse_args()
    
    # Set Jittor optimization flags for better GPU utilization and memory management
    jt.flags.use_cuda = 1
    jt.flags.lazy_execution = 1
    jt.flags.use_stat_allocator = 1

    # Additional memory optimization flags
    try:
        jt.flags.auto_mixed_precision_level = 0  # Disable AMP to save memory
    except AttributeError:
        pass  # Flag not available in this Jittor version

    # Set memory management
    import os
    os.environ['JT_SYNC'] = '0'  # Async execution for better memory management
    
    config = getattr(import_module(args.config), "C")
    
    engine = Engine(custom_parser=parser)
    engine.distributed = True if jt.world_size > 1 else False
    engine.local_rank = jt.rank
    engine.world_size = jt.world_size
    
    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)
    
    if args.pad_SUNRGBD and config.dataset_name != "SUNRGBD":
        args.pad_SUNRGBD = False
        logger.warning("pad_SUNRGBD is only used for SUNRGBD dataset")
    
    if (args.pad_SUNRGBD) and (not config.backbone.startswith("DFormerv2")):
        raise ValueError("DFormerv1 is not recommended with pad_SUNRGBD")
    
    if (not args.pad_SUNRGBD) and config.backbone.startswith("DFormerv2") and config.dataset_name == "SUNRGBD":
        raise ValueError("DFormerv2 is not recommended without pad_SUNRGBD")
    
    config.pad = args.pad_SUNRGBD
    
    if args.use_seed:
        set_seed(config.seed)
        logger.info(f"set seed {config.seed}")
    else:
        logger.info("use random seed")
    
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)
    
    # Reduce validation batch size to prevent memory issues and deadlocks
    val_dl_factor = 0.5  # Reduce validation batch size
    val_batch_size = max(1, int(config.batch_size * val_dl_factor)) if config.dataset_name != "SUNRGBD" else 1
    val_loader, val_sampler = get_val_loader(
        engine,
        RGBXDataset,
        config,
        val_batch_size=val_batch_size,
    )
    logger.info(f"val dataset len:{len(val_loader) * int(args.gpus)}")
    
    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + "/tb"
        pp = pprint.PrettyPrinter(indent=4)
        logger.info("config: \n%s", pp.pformat(config))
    
    logger.info("args parsed:")
    for k, v in args.__dict__.items():
        logger.info("%s: %s", k, str(v))
    
    criterion = nn.CrossEntropyLoss(ignore_index=config.background)
    
    if args.syncbn:
        try:
            BatchNorm2d = nn.SyncBatchNorm
            logger.info("using syncbn")
        except AttributeError:
            logger.warning("SyncBatchNorm not available in Jittor, using regular BatchNorm2d")
            BatchNorm2d = nn.BatchNorm2d
    else:
        BatchNorm2d = nn.BatchNorm2d
        logger.info("using regular bn")
    
    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=args.syncbn,
    )
    
    base_lr = config.lr
    params_list = group_weight(model)

    if config.optimizer == 'AdamW':
        optimizer = jt.optim.AdamW(params_list, lr=base_lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = jt.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError
    
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        optimizer,
        power=config.lr_power,
        max_iters=total_iteration,
        warmup_iters=config.niters_per_epoch * config.warm_up_epoch,
    )
    
    if engine.distributed:
        logger.info(".............distributed training.............")
        model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    logger.info("begin trainning:")

    miou, best_miou = 0.0, 0.0
    train_timer = GpuTimer()
    eval_timer = GpuTimer()
    
    for epoch in range(engine.state.epoch, config.nepochs + 1):
        model.train()
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        
        dataloader = iter(train_loader)
        sum_loss = 0
        
        train_timer.start()
        for idx in range(config.niters_per_epoch):
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            loss = model(imgs, modal_xs, gts)
            
            if isinstance(loss, tuple):
                if len(loss) == 2:
                    predictions, loss = loss
                else:
                    loss = loss[-1]

            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            
            optimizer.step(loss)
            
            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr_policy.step(current_idx)
            
            if engine.distributed:
                sum_loss += reduce_loss.item()
                current_lr = optimizer.lr if hasattr(optimizer, 'lr') else lr_policy.get_lr()[0]
                print_str = (
                    f"Epoch {epoch}/{config.nepochs} "
                    f"Iter {idx + 1}/{config.niters_per_epoch}: "
                    f"lr={current_lr:.4e} "
                    f"loss={reduce_loss.item():.4f} total_loss={(sum_loss / (idx + 1)):.4f}"
                )
            else:
                sum_loss += loss.item()
                current_lr = optimizer.lr if hasattr(optimizer, 'lr') else lr_policy.get_lr()[0]
                print_str = (
                    f"Epoch {epoch}/{config.nepochs} "
                    f"Iter {idx + 1}/{config.niters_per_epoch}: "
                    f"lr={current_lr:.4e} loss={loss.item():.4f} total_loss={(sum_loss / (idx + 1)):.4f}"
                )

            if ((idx + 1) % int(config.niters_per_epoch * 0.1) == 0 or idx == 0) and \
               ((engine.distributed and engine.local_rank == 0) or not engine.distributed):
                logger.info(print_str)

            # Memory cleanup every 10 iterations to prevent OOM
            if (idx + 1) % 10 == 0:
                jt.clean()
                # Reduce sync frequency to avoid performance issues
                if (idx + 1) % 20 == 0:
                    jt.sync_all()

        train_timer.stop()
        
        if is_eval(epoch, config):
            eval_timer.start()
            model.eval()

            # Force cleanup and reset before evaluation
            jt.clean()
            jt.gc()  # Force garbage collection

            # Recreate validation loader to avoid iterator conflicts
            logger.info("Recreating validation loader for eval...")
            val_loader, val_sampler = get_val_loader(
                engine,
                RGBXDataset,
                config,
                val_batch_size=val_batch_size,
            )

            # Reduce sync calls to prevent hanging
            if epoch % 5 == 0:  # Only sync every 5 epochs
                jt.sync_all()

            with jt.no_grad():
                if args.sliding:
                    # Use sliding window inference
                    from utils.val_mm import sliding_window_inference
                    metric = SegmentationMetric(val_loader.dataset.num_classes)
                    
                    for i, minibatch in enumerate(val_loader):
                        rgb = minibatch['data']
                        targets = minibatch['label']
                        modal = minibatch['modal_x']

                        pred_logits = sliding_window_inference(
                            model, rgb, modal, [512, 512], [256, 256], val_loader.dataset.num_classes
                        )

                        predictions = jt.argmax(pred_logits, dim=1)
                        metric.update(predictions.numpy(), targets.numpy())
                    
                    all_metrics = metric
                elif args.mst and epoch > 50:
                    # Use MST only after epoch 50 to avoid slowdown in early training
                    all_metrics = evaluate_msf(model, val_loader, [0.75, 1.0, 1.25], False, verbose=False)  # Reduced scales and no flip
                elif args.mst and epoch > 20:
                    # Use limited MST after epoch 20
                    all_metrics = evaluate_msf(model, val_loader, [1.0], False, verbose=False)  # Single scale, no flip
                else:
                    # Use single scale evaluation in early training (much faster)
                    logger.info(f"Starting single-scale evaluation for epoch {epoch}...")
                    all_metrics = evaluate(model, val_loader, verbose=False)
                    logger.info(f"Evaluation completed for epoch {epoch}")

                if engine.distributed:
                    if engine.local_rank == 0:
                        if args.sliding:
                            # For sliding window, all_metrics is a SegmentationMetric object
                            results = all_metrics.get_results()
                            miou = results['mIoU']
                            macc = results['mAcc']
                            f1 = mf1 = 0.0  # F1 score not available in current implementation
                        else:
                            # For other evaluation methods, handle as before but expect dict return
                            if isinstance(all_metrics, dict):
                                miou = all_metrics['mIoU']
                                macc = all_metrics['mAcc']
                                f1 = mf1 = 0.0
                            else:
                                metric = all_metrics[0]
                                for other_metric in all_metrics[1:]:
                                    metric.update_hist(other_metric.hist)
                                ious, miou = metric.compute_iou()
                                acc, macc = metric.compute_pixel_acc()
                                f1, mf1 = metric.compute_f1()
                else:
                    if args.sliding:
                        # For sliding window, all_metrics is a SegmentationMetric object
                        results = all_metrics.get_results()
                        miou = results['mIoU']
                        macc = results['mAcc']
                        f1 = mf1 = 0.0  # F1 score not available in current implementation
                    else:
                        # For other evaluation methods, expect dict return
                        if isinstance(all_metrics, dict):
                            miou = all_metrics['mIoU']
                            macc = all_metrics['mAcc']
                            f1 = mf1 = 0.0
                        else:
                            ious, miou = all_metrics.compute_iou()
                            acc, macc = all_metrics.compute_pixel_acc()
                            f1, mf1 = all_metrics.compute_f1()

                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    if miou > best_miou:
                        best_miou = miou
                        engine.save_and_link_checkpoint(
                            config.log_dir,
                            config.log_dir,
                            config.log_dir_link,
                            infor=f"_miou_{miou}",
                            metric=miou
                        )
                    logger.info(f"Epoch {epoch} validation result: mIoU {miou:.4f}, best mIoU {best_miou:.4f}")

            # Clear memory after evaluation
            jt.clean()
            # Reduce sync frequency to prevent hanging
            if epoch % 5 == 0:  # Only sync every 5 epochs
                jt.sync_all()
            eval_timer.stop()

        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            eval_count = sum(1 for i in range(epoch + 1, config.nepochs + 1) if is_eval(i, config))
            eval_time = eval_timer.mean_time if eval_timer.mean_time is not None else 0.0
            left_time = train_timer.mean_time * (config.nepochs - epoch) + eval_time * eval_count
            eta = (datetime.datetime.now() + datetime.timedelta(seconds=left_time)).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Avg train time: {train_timer.mean_time:.2f}s, avg eval time: {eval_time:.2f}s, left eval count: {eval_count}, ETA: {eta}")


if __name__ == "__main__":
    main()
