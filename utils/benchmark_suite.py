#!/usr/bin/env python3
# Unified benchmarking script for DFormer-Jittor
# Measures: inference latency (single/batch), 1-epoch train time, memory, GPU util, metrics

import argparse
import json
import os
import threading
import time
import subprocess
from statistics import mean

import numpy as np
import jittor as jt
from jittor import nn

from importlib import import_module
from models.builder import EncoderDecoder as segmodel
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.val_mm import evaluate


def _nvidia_smi_query(device_index=0):
    try:
        out = subprocess.check_output([
            'nvidia-smi',
            f'--query-gpu=utilization.gpu,utilization.memory,memory.used',
            '--format=csv,noheader,nounits',
            f'-i', str(device_index)
        ], stderr=subprocess.DEVNULL).decode('utf-8').strip()
        parts = [p.strip() for p in out.split(',')]
        if len(parts) >= 3:
            return {
                'gpu_util': float(parts[0]),
                'mem_util': float(parts[1]),
                'mem_used_mb': float(parts[2])
            }
    except Exception:
        pass
    return None


class GPUSampler:
    def __init__(self, device_index=0, interval=0.2):
        self.device_index = device_index
        self.interval = interval
        self._stop = threading.Event()
        self.records = []
        self._thread = None

    def start(self):
        self._stop.clear()
        self.records = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            info = _nvidia_smi_query(self.device_index)
            if info is not None:
                self.records.append(info)
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def summary(self):
        if not self.records:
            return {}
        return {
            'gpu_util_avg': mean([r['gpu_util'] for r in self.records]),
            'gpu_util_max': max([r['gpu_util'] for r in self.records]),
            'mem_used_mb_avg': mean([r['mem_used_mb'] for r in self.records]),
            'mem_used_mb_max': max([r['mem_used_mb'] for r in self.records]),
        }


def build_model(cfg, syncbn=True):
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.background)
    BN = nn.SyncBatchNorm if hasattr(nn, 'SyncBatchNorm') and syncbn else nn.BatchNorm2d
    model = segmodel(cfg=cfg, criterion=criterion, norm_layer=BN, syncbn=syncbn)
    model.eval()
    return model


def measure_inference(model, val_loader, warmup=20, iters=100):
    jt.clean()
    times = []
    sampler = GPUSampler(device_index=0)
    # Warmup
    for i, batch in enumerate(val_loader):
        imgs = batch['data']
        modal = batch['modal_x']
        _ = model(imgs, modal)
        if i + 1 >= warmup:
            break
    jt.sync_all()
    # Timed
    sampler.start()
    n = 0
    for batch in val_loader:
        imgs = batch['data']
        modal = batch['modal_x']
        t0 = time.perf_counter()
        _ = model(imgs, modal)
        jt.sync_all()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        n += 1
        if n >= iters:
            break
    sampler.stop()
    gpu_stats = sampler.summary()
    return {
        'latency_ms_mean': float(np.mean(times)) if times else None,
        'latency_ms_p50': float(np.percentile(times, 50)) if times else None,
        'latency_ms_p95': float(np.percentile(times, 95)) if times else None,
        'throughput_imgs_per_s': (1000.0 * len(times) * val_loader.batch_size / sum(times)) if times else None,
        'gpu_stats': gpu_stats,
        'max_memory_used_mb': None  # Jittor memory API varies; leave None or extend with jt flags
    }


def measure_train_one_epoch(model, train_loader, optimizer):
    model.train()
    sampler = GPUSampler(device_index=0)
    start_time = time.time()
    loss_sum = 0.0
    steps = 0
    sampler.start()
    for batch in train_loader:
        imgs = batch['data']
        modal = batch['modal_x']
        gts = batch['label']
        loss = model(imgs, modal, gts)
        if isinstance(loss, (tuple, list)):
            loss = loss[-1] if len(loss) > 1 else loss[0]
        optimizer.step(loss)
        loss_sum += float(loss)
        steps += 1
    jt.sync_all()
    sampler.stop()
    elapsed = time.time() - start_time
    return {
        'epoch_time_s': elapsed,
        'train_loss_avg': loss_sum / max(1, steps),
        'gpu_stats': sampler.summary(),
        'max_memory_used_mb': None,
    }


def run_eval_metrics(model, val_loader, config):
    jt.clean()
    res = evaluate(model, val_loader, verbose=False, config=config)
    if isinstance(res, dict):
        return {'mIoU': float(res.get('mIoU', 0.0)), 'mAcc': float(res.get('mAcc', 0.0)), 'Overall_Acc': float(res.get('Overall_Acc', 0.0))}
    else:
        # SegmentationMetric
        results = res.get_results()
        return {'mIoU': float(results.get('mIoU', 0.0)), 'mAcc': float(results.get('mAcc', 0.0)), 'Overall_Acc': float(results.get('Overall_Acc', 0.0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--limit_train_iters', type=int, default=None)
    parser.add_argument('--output', type=str, default='output/benchmark_jittor.json')
    args = parser.parse_args()

    config = getattr(import_module(args.config), 'C')
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        config.niters_per_epoch = config.num_train_imgs // config.batch_size + 1

    jt.flags.use_cuda = 1

    engine = Engine()
    train_loader, _ = get_train_loader(engine, RGBXDataset, config)
    val_loader, _ = get_val_loader(engine, RGBXDataset, config, val_batch_size=config.batch_size)

    model = build_model(config, syncbn=False)

    # Optimizer
    optimizer = jt.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    results = {
        'framework': 'jittor',
        'model': config.backbone,
        'dataset': config.dataset_name,
        'batch_size': config.batch_size,
        'runs': args.runs,
        'inference': [],
        'train_epoch': [],
        'metrics': None,
    }

    for _ in range(args.runs):
        inf = measure_inference(model, val_loader, warmup=10, iters=50)
        results['inference'].append(inf)

    # Limit iters if requested
    if args.limit_train_iters is not None:
        original_loader = train_loader
        def limited_iter():
            it = iter(original_loader)
            for i in range(args.limit_train_iters):
                yield next(it)
        train_loader = limited_iter()
    tr = measure_train_one_epoch(model, train_loader, optimizer)
    results['train_epoch'].append(tr)

    metrics = run_eval_metrics(model, val_loader, config)
    results['metrics'] = metrics

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

