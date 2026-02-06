#!/bin/bash

# DFormer Jittor Inference Script
# Adapted from PyTorch version for Jittor framework

set -e  # Exit on any error

# Default configuration
GPUS=${GPUS:-1}
CONFIG=${CONFIG:-"local_configs.NYUDepthv2.DFormer_Large"}
CHECKPOINT=${CHECKPOINT:-"/caojiaolong/hjd/DFormer-Jittor/checkpoints/trained/NYUv2_DFormer_Large.pth"}
OUTPUT_DIR=${OUTPUT_DIR:-"/caojiaolong/hjd/DFormer-Jittor/output"}
VERBOSE=${VERBOSE:-false}
MULTI_SCALE=${MULTI_SCALE:-false}
FLIP=${FLIP:-false}
EVAL_ONLY=${EVAL_ONLY:-false}
SIMPLE_MODE=${SIMPLE_MODE:-false}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint|--continue_fpath)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output|--save_path)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --multi-scale)
            MULTI_SCALE=true
            shift
            ;;
        --flip)
            FLIP=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --simple)
            SIMPLE_MODE=true
            shift
            ;;
        --help|-h)
            echo "DFormer Jittor Inference Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config CONFIG           Model configuration (default: $CONFIG)"
            echo "  --checkpoint PATH         Path to checkpoint file"
            echo "  --output PATH             Output directory (default: $OUTPUT_DIR)"
            echo "  --gpus N                  Number of GPUs (default: $GPUS)"
            echo "  --verbose, -v             Enable verbose output"
            echo "  --multi-scale             Enable multi-scale inference"
            echo "  --flip                    Enable flip augmentation"
            echo "  --eval-only               Only run evaluation, no saving"
            echo "  --simple                  Use simplified inference (recommended for testing)"
            echo "  --help, -h                Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic inference"
            echo "  $0 --config local_configs.NYUDepthv2.DFormer_Small \\"
            echo "     --checkpoint /path/to/checkpoint.pkl"
            echo ""
            echo "  # Multi-scale inference with flip"
            echo "  $0 --config local_configs.NYUDepthv2.DFormer_Base \\"
            echo "     --checkpoint /path/to/checkpoint.pkl \\"
            echo "     --multi-scale --flip --verbose"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set CUDA device
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# Validate inputs
if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint path is required. Use --checkpoint to specify."
    echo "Use --help for usage information."
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "DFormer Jittor Inference Configuration:"
echo "==========================================="
echo "Config:       $CONFIG"
echo "Checkpoint:   $CHECKPOINT"
echo "Output:       $OUTPUT_DIR"
echo "GPUs:         $GPUS"
echo "CUDA Device:  $CUDA_VISIBLE_DEVICES"
echo "Verbose:      $VERBOSE"
echo "Multi-scale:  $MULTI_SCALE"
echo "Flip:         $FLIP"
echo "Eval only:    $EVAL_ONLY"
echo "Simple mode:  $SIMPLE_MODE"
echo "==========================================="

# Build command arguments
ARGS=(
    --config="$CONFIG"
    --continue_fpath="$CHECKPOINT"
    --save_path="$OUTPUT_DIR"
    --gpus="$GPUS"
)

if [[ "$VERBOSE" == "true" ]]; then
    ARGS+=(--verbose)
fi

if [[ "$MULTI_SCALE" == "true" ]]; then
    ARGS+=(--multi_scale)
    ARGS+=(--scales 0.5 0.75 1.0 1.25 1.5)
fi

if [[ "$FLIP" == "true" ]]; then
    ARGS+=(--flip)
fi

if [[ "$EVAL_ONLY" == "true" ]]; then
    ARGS+=(--eval_only)
fi

# Run inference
echo "Starting inference..."

if [[ "$SIMPLE_MODE" == "true" ]]; then
    echo "Using simplified inference mode..."
    python simple_infer.py --test-mode --use-real-model --output="$OUTPUT_DIR"
else
    python utils/infer.py "${ARGS[@]}"
fi

echo "Inference completed successfully!"

# Available configurations and checkpoints for reference:
echo ""
echo "Available model configurations:"
echo "================================"
echo ""
echo "NYUv2 DFormers:"
echo "  --config=local_configs.NYUDepthv2.DFormer_Large"
echo "  --config=local_configs.NYUDepthv2.DFormer_Base"
echo "  --config=local_configs.NYUDepthv2.DFormer_Small"
echo "  --config=local_configs.NYUDepthv2.DFormer_Tiny"
echo ""
echo "NYUv2 DFormerv2:"
echo "  --config=local_configs.NYUDepthv2.DFormerv2_L"
echo "  --config=local_configs.NYUDepthv2.DFormerv2_B"
echo "  --config=local_configs.NYUDepthv2.DFormerv2_S"
echo ""
echo "SUNRGBD DFormers:"
echo "  --config=local_configs.SUNRGBD.DFormer_Large"
echo "  --config=local_configs.SUNRGBD.DFormer_Base"
echo "  --config=local_configs.SUNRGBD.DFormer_Small"
echo "  --config=local_configs.SUNRGBD.DFormer_Tiny"
echo ""
echo "SUNRGBD DFormerv2:"
echo "  --config=local_configs.SUNRGBD.DFormerv2_L"
echo "  --config=local_configs.SUNRGBD.DFormerv2_B"
echo "  --config=local_configs.SUNRGBD.DFormerv2_S"
echo ""
echo "Example checkpoint paths:"
echo "  checkpoints/trained/NYUv2_DFormer_Large.pkl"
echo "  checkpoints/trained/DFormerv2_Small_NYU.pkl"
echo "  pretrained/DFormerv2_Small_pretrained.pth"
