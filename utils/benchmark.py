"""
Benchmark script for DFormer Jittor models.
"""

import argparse
import time
import jittor as jt
from jittor import nn
from importlib import import_module
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.builder import EncoderDecoder


def benchmark_model(model, input_shape=(3, 480, 640), batch_size=1, warmup=10, runs=100):
    """Benchmark model inference speed and memory usage."""
    print(f"Benchmarking model with input shape: {input_shape}")
    
    # Create dummy inputs
    rgb = jt.random((batch_size, *input_shape))
    depth = jt.random((batch_size, 1, input_shape[1], input_shape[2]))
    
    model.eval()
    
    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        with jt.no_grad():
            _ = model(rgb, depth)
    
    # Sync and measure time
    jt.sync_all(True)
    
    print(f"Running {runs} iterations...")
    start_time = time.time()
    
    for _ in range(runs):
        with jt.no_grad():
            _ = model(rgb, depth)
    
    jt.sync_all(True)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / runs
    fps = 1.0 / avg_time
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    
    # Estimate parameters (rough count)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DFormer Jittor model")
    parser.add_argument("--backbone", default="DFormer-Base", help="Backbone model")
    parser.add_argument("--decoder", default="ham", help="Decoder type")
    parser.add_argument("--num_classes", type=int, default=40, help="Number of classes")
    parser.add_argument("--input_size", nargs=2, type=int, default=[480, 640], help="Input H W")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark iterations")
    
    args = parser.parse_args()
    
    # Build model
    print(f"Building model: {args.backbone} + {args.decoder}")
    model = EncoderDecoder.from_config(
        backbone=args.backbone,
        decoder=args.decoder,
        num_classes=args.num_classes,
        decoder_embed_dim=512
    )
    
    # Run benchmark
    benchmark_model(
        model,
        input_shape=(3, args.input_size[0], args.input_size[1]),
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs
    ) 