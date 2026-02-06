# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate jittordet

export CUDA_VISIBLE_DEVICES="0"
GPUS=1
LOGFILE="eval_results.log"

# Check if jittor is available
python -c "import jittor; print('Jittor version:', jittor.__version__)" || {
    echo "Error: Jittor not found. Please check environment setup."
    exit 1
}

# Function to check if model file exists and run evaluation
run_eval() {
    local config=$1
    local model_path=$2
    local model_name=$3

    echo "========================================" | tee -a $LOGFILE
    echo "$(date): Checking model: $model_name" | tee -a $LOGFILE
    if [ -f "$model_path" ]; then
        echo "✓ Found model file: $model_path" | tee -a $LOGFILE
        echo "$(date): Starting evaluation for $model_name..." | tee -a $LOGFILE

        # Run evaluation with timeout and better error handling (using fast mode)
        timeout 1800 python utils/eval.py --config=$config --gpus=$GPUS --continue_fpath=$model_path --verbose 2>&1 | tee -a $LOGFILE
        local exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo "$(date): ✓ $model_name evaluation completed successfully" | tee -a $LOGFILE
        elif [ $exit_code -eq 124 ]; then
            echo "$(date): ⚠ $model_name evaluation timed out (1 hour limit)" | tee -a $LOGFILE
        else
            echo "$(date): ✗ $model_name evaluation failed with exit code $exit_code" | tee -a $LOGFILE
        fi
    else
        echo "✗ Model file not found: $model_path" | tee -a $LOGFILE
        echo "$(date): Skipping $model_name evaluation" | tee -a $LOGFILE

        # Check if there's a downloading file
        if [ -f "${model_path}.baiduyun.p.downloading" ]; then
            echo "  ℹ Note: Found downloading file ${model_path}.baiduyun.p.downloading" | tee -a $LOGFILE
            echo "  ℹ This model may still be downloading from Baidu Cloud" | tee -a $LOGFILE
        elif [ -f "${model_path}.downloading" ]; then
            echo "  ℹ Note: Found downloading file ${model_path}.downloading" | tee -a $LOGFILE
            echo "  ℹ This model may still be downloading" | tee -a $LOGFILE
        else
            echo "  ℹ Note: No downloading file found. Model may need to be downloaded manually." | tee -a $LOGFILE
        fi
    fi
    echo "========================================" | tee -a $LOGFILE
    echo "" | tee -a $LOGFILE
}

# Function to run evaluation with different strategies
run_eval_fast() {
    local config=$1
    local model_path=$2
    local model_name=$3

    echo "========================================" | tee -a $LOGFILE
    echo "$(date): Checking model: $model_name (Fast Mode)" | tee -a $LOGFILE
    if [ -f "$model_path" ]; then
        echo "✓ Found model file: $model_path" | tee -a $LOGFILE
        echo "$(date): Starting fast evaluation for $model_name..." | tee -a $LOGFILE

        # Run evaluation without multi-scale and sliding window for faster results
        timeout 1800 python utils/eval.py --config=$config --gpus=$GPUS --continue_fpath=$model_path --verbose 2>&1 | tee -a $LOGFILE
        local exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo "$(date): ✓ $model_name fast evaluation completed successfully" | tee -a $LOGFILE
        elif [ $exit_code -eq 124 ]; then
            echo "$(date): ⚠ $model_name evaluation timed out (30 min limit)" | tee -a $LOGFILE
        else
            echo "$(date): ✗ $model_name evaluation failed with exit code $exit_code" | tee -a $LOGFILE
        fi
    else
        echo "✗ Model file not found: $model_path" | tee -a $LOGFILE
        echo "$(date): Skipping $model_name evaluation" | tee -a $LOGFILE
    fi
    echo "========================================" | tee -a $LOGFILE
    echo "" | tee -a $LOGFILE
}

echo "==== NYUDepthv2 DFormer Models ====" > $LOGFILE
run_eval_fast "local_configs.NYUDepthv2.DFormer_Large" "checkpoints/trained/NYUv2_DFormer_Large.pth" "NYUv2_DFormer_Large"
run_eval_fast "local_configs.NYUDepthv2.DFormer_Base" "checkpoints/trained/NYUv2_DFormer_Base.pth" "NYUv2_DFormer_Base"
run_eval_fast "local_configs.NYUDepthv2.DFormer_Small" "checkpoints/trained/NYUv2_DFormer_Small.pth" "NYUv2_DFormer_Small"
run_eval_fast "local_configs.NYUDepthv2.DFormer_Tiny" "checkpoints/trained/NYUv2_DFormer_Tiny.pth" "NYUv2_DFormer_Tiny"
echo "==== NYUDepthv2 DFormerv2 Models ====" >> $LOGFILE
run_eval_fast "local_configs.NYUDepthv2.DFormerv2_S" "checkpoints/trained/DFormerv2_Small_NYU.pth" "DFormerv2_Small_NYU"
run_eval_fast "local_configs.NYUDepthv2.DFormerv2_B" "checkpoints/trained/DFormerv2_Base_NYU.pth" "DFormerv2_Base_NYU"
run_eval_fast "local_configs.NYUDepthv2.DFormerv2_L" "checkpoints/trained/DFormerv2_Large_NYU.pth" "DFormerv2_Large_NYU"

echo "==== SUNRGBD DFormer Models ====" >> $LOGFILE
run_eval_fast "local_configs.SUNRGBD.DFormer_Large" "checkpoints/trained/SUNRGBD_DFormer_Large.pth" "SUNRGBD_DFormer_Large"
run_eval_fast "local_configs.SUNRGBD.DFormer_Base" "checkpoints/trained/SUNRGBD_DFormer_Base.pth" "SUNRGBD_DFormer_Base"
run_eval_fast "local_configs.SUNRGBD.DFormer_Small" "checkpoints/trained/SUNRGBD_DFormer_Small.pth" "SUNRGBD_DFormer_Small"
run_eval_fast "local_configs.SUNRGBD.DFormer_Tiny" "checkpoints/trained/SUNRGBD_DFormer_Tiny.pth" "SUNRGBD_DFormer_Tiny"

echo "==== SUNRGBD DFormerv2 Models ====" >> $LOGFILE
run_eval_fast "local_configs.SUNRGBD.DFormerv2_S" "checkpoints/trained/DFormerv2_Small_SUNRGBD.pth" "DFormerv2_Small_SUNRGBD"
run_eval_fast "local_configs.SUNRGBD.DFormerv2_B" "checkpoints/trained/DFormerv2_Base_SUNRGBD.pth" "DFormerv2_Base_SUNRGBD"
run_eval_fast "local_configs.SUNRGBD.DFormerv2_L" "checkpoints/trained/DFormerv2_Large_SUNRGBD.pth" "DFormerv2_Large_SUNRGBD"

echo "==== Evaluation Summary ====" >> $LOGFILE
echo "All available models have been evaluated. Check the log above for detailed results." >> $LOGFILE
