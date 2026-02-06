"""
Evaluation script for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import os
import sys
import time
import argparse
import numpy as np
import jittor as jt
from jittor import nn

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataloader.dataloader import get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.metric import SegmentationMetric, AverageMeter
from utils.val_mm import evaluate, evaluate_msf, sliding_window_inference
from utils.jt_utils import load_model, AverageMeter, Timer
from utils.engine.engine import Engine


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DFormer Evaluation')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--continue_fpath', required=True, help='checkpoint file path')
    parser.add_argument('--gpus', type=int, default=1, help='number of GPUs')
    parser.add_argument('--dataset', default='nyudepthv2', help='dataset name')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--scales', nargs='+', type=float, default=[1.0], help='evaluation scales')
    parser.add_argument('--multi_scale', action='store_true', help='use multi-scale evaluation')
    parser.add_argument('--flip', action='store_true', help='use flip augmentation')
    parser.add_argument('--sliding', action='store_true', help='use sliding window inference')
    parser.add_argument('--window-size', nargs=2, type=int, default=[512, 512], help='sliding window size')
    parser.add_argument('--stride', nargs=2, type=int, default=[256, 256], help='sliding window stride')
    parser.add_argument('--save-pred', action='store_true', help='save predictions')
    parser.add_argument('--pred-dir', default='./predictions', help='prediction save directory')
    parser.add_argument('--verbose', action='store_true', help='verbose output')

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load config
    from importlib import import_module
    config = getattr(import_module(args.config), "C")
    
    # Set device
    jt.flags.use_cuda = 1

    # Create engine (simplified for evaluation)
    engine = Engine()

    # Create data loader
    val_loader, val_sampler = get_val_loader(
        engine=engine,
        dataset_cls=RGBXDataset,
        config=config,
        val_batch_size=args.batch_size
    )
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {val_loader.dataset.num_classes}")
    print(f"Number of samples: {len(val_loader.dataset)}")
    
    # Create model
    from models import build_model
    model = build_model(config)
    
    # Load checkpoint
    if args.continue_fpath:
        print(f"Loading checkpoint from {args.continue_fpath}")
        model = load_model(model, args.continue_fpath)
    
    model.eval()
    
    # Create save directory
    if args.save_pred:
        os.makedirs(args.pred_dir, exist_ok=True)
    
    # Evaluation
    print("Starting evaluation...")
    start_time = time.time()
    
    # Use advanced evaluation settings to match PyTorch baseline
    if args.multi_scale or args.flip or args.sliding:
        # Multi-scale evaluation with flip and/or sliding window
        if args.multi_scale:
            # Prefer user-specified scales if provided, else fallback to default 5-scale
            scales = args.scales if args.scales and len(args.scales) > 0 else [0.5, 0.75, 1.0, 1.25, 1.5]
        else:
            scales = [1.0]  # Single scale

        print(f"Using advanced evaluation:")
        print(f"  Scales: {scales}")
        print(f"  Flip augmentation: {args.flip}")
        print(f"  Sliding window: {args.sliding}")

        metric = evaluate_msf(
            model, val_loader, config=config, scales=scales, flip=args.flip, sliding=args.sliding
        )
        results = metric.get_results()
    else:
        # Standard evaluation
        print("Using standard evaluation (single scale, no augmentation)")
        results = evaluate(model, val_loader, verbose=args.verbose)
    
    end_time = time.time()
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"mIoU: {results['mIoU']:.4f}")
    print(f"mAcc: {results['mAcc']:.4f}")
    print(f"Overall Acc: {results['Overall_Acc']:.4f}")
    print(f"FWIoU: {results['FWIoU']:.4f}")
    print(f"Evaluation time: {end_time - start_time:.2f}s")
    
    # Print per-class results
    if args.verbose:
        print("\nPer-class IoU:")
        for i, iou in enumerate(results['IoU_per_class']):
            print(f"Class {i:2d}: {iou:.4f}")
        
        print("\nPer-class Accuracy:")
        for i, acc in enumerate(results['Acc_per_class']):
            print(f"Class {i:2d}: {acc:.4f}")


def evaluate_sliding_window(model, data_loader, window_size, stride, verbose=False):
    """Evaluate model with sliding window inference."""
    model.eval()
    
    metric = SegmentationMetric(data_loader.dataset.num_classes)
    timer = Timer()
    
    with jt.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            timer.tic()
            
            if isinstance(images, (list, tuple)):
                rgb, modal = images
                pred_logits = sliding_window_inference(
                    model, rgb, modal, window_size, stride, data_loader.dataset.num_classes
                )
            else:
                pred_logits = sliding_window_inference(
                    model, images, None, window_size, stride, data_loader.dataset.num_classes
                )
            
            predictions = jt.argmax(pred_logits, dim=1)
            metric.update(predictions.numpy(), targets.numpy())
            
            if verbose and i % 10 == 0:
                print(f"Processed {i}/{len(data_loader)} batches, "
                      f"Time: {timer.toc():.3f}s")
    
    return metric.get_results()


def single_gpu_test(model, data_loader, show=False, out_dir=None, **show_kwargs):
    """Test with single GPU."""
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = jt.utils.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        with jt.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                dataset.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                    **show_kwargs)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


if __name__ == '__main__':
    main()
