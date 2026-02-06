#!/usr/bin/env python3
"""
Parametrized quick evaluation (evaluate first N samples) to verify fixes fast.
Usage example:
  python utils/quick_eval_param.py \
    --config local_configs.NYUDepthv2.DFormerv2_L \
    --checkpoint checkpoints/trained/DFormerv2_Large_NYU.pth \
    --num-samples 10
"""

import os
import sys
import time
import argparse
import numpy as np
import jittor as jt

# Add repo root to PYTHONPATH
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from importlib import import_module
from utils.dataloader.dataloader import get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.metric import SegmentationMetric
from utils.jt_utils import load_model
from utils.engine.engine import Engine
from models import build_model


def parse_args():
    p = argparse.ArgumentParser(description='Quick Param Evaluation')
    p.add_argument('--config', required=True, help='config module path, e.g., local_configs.NYUDepthv2.DFormer_Large')
    p.add_argument('--checkpoint', required=True, help='checkpoint path')
    p.add_argument('--num-samples', type=int, default=10, help='number of samples to eval')
    p.add_argument('--batch-size', type=int, default=1, help='val batch size')
    return p.parse_args()


def main():
    args = parse_args()

    # Jittor device
    jt.flags.use_cuda = 1

    # Load config
    config = getattr(import_module(args.config), 'C')

    engine = Engine()

    val_loader, _ = get_val_loader(
        engine=engine,
        dataset_cls=RGBXDataset,
        config=config,
        val_batch_size=args.batch_size,
    )

    print(f"Dataset: {config.dataset_name}")
    print(f"Num classes: {val_loader.dataset.num_classes}")
    print(f"Total samples in dataset: {len(val_loader.dataset)}")

    # Build model
    model = build_model(config)

    # Load weights
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model = load_model(model, args.checkpoint)
    else:
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 1

    model.eval()

    # Evaluate first N samples
    metric = SegmentationMetric(val_loader.dataset.num_classes)
    start_time = time.time()

    with jt.no_grad():
        for i, minibatch in enumerate(val_loader):
            if i >= args.num_samples:
                break
            try:
                images = minibatch['data']
                labels = minibatch['label']
                modal_xs = minibatch['modal_x']

                outputs = model(images, modal_xs)
                # Normalize output format
                if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
                    outputs = outputs[0]
                    if isinstance(outputs, list):
                        outputs = outputs[0]
                elif isinstance(outputs, dict):
                    outputs = outputs['out']

                preds = jt.argmax(outputs, dim=1)
                if isinstance(preds, tuple):
                    preds = preds[0]

                metric.update(preds.numpy(), labels.numpy())

                if i < 3:
                    print(f"Sample {i+1}: img {images.shape}, depth {modal_xs.shape}, pred {preds.shape}")
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue

    end_time = time.time()
    results = metric.get_results()

    print("\nQuick Param Eval Results:")
    print("-"*50)
    print(f"mIoU: {results['mIoU']:.4f}")
    print(f"mAcc: {results['mAcc']:.4f}")
    print(f"Overall Acc: {results['Overall_Acc']:.4f}")
    print(f"FWIoU: {results['FWIoU']:.4f}")
    print(f"Time: {end_time-start_time:.2f}s for {args.num_samples} samples")

    # print first 10 class IoUs
    print("First 10 class IoUs:")
    for ci in range(min(10, len(results['IoU_per_class']))):
        print(f"  Class {ci:02d}: {results['IoU_per_class'][ci]:.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

