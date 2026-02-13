#!/bin/bash
#SBATCH --job-name=motionr1_sft
#SBATCH --output=logs/sft_full_%j.out
#SBATCH --error=logs/sft_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=167:59:00
#SBATCH --partition=journey_gpu

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Navigate to project directory
cd /home/bi.ga/Workspace/vlmm-mcot

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dora_cuda

# Add workspace to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Disable debug mode for production
export DEBUG_MODE="false"
export WANDB_MODE="online"

# FULL TRAINING MODE
export QUICK_TEST="false"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Configuration - subset only (filtered to available videos)
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
EXP_NAME="sft_full_slurm_${SLURM_JOB_ID}"
OUT_DIR="outputs/${EXP_NAME}"
DATA_ROOT="/mnt/data/stgr"
SOURCE_DATASET="${DATA_ROOT}/json_data/STGR-SFT-subset.json"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-SFT-subset-motion-v3.json"

# Auto-augment data with motion tags if not already done
if [ ! -f "$DATASET_JSON" ]; then
    echo "=========================================="
    echo "Motion-augmented dataset not found!"
    echo "Running data augmentation first..."
    echo "=========================================="
    echo "Source: $SOURCE_DATASET"
    echo "Output: $DATASET_JSON"
    echo ""
    
    python scripts/augment_motion_data_simple.py \
        --input "$SOURCE_DATASET" \
        --output "$DATASET_JSON"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Data augmentation complete!"
        echo ""
    else
        echo "ERROR: Data augmentation failed!"
        exit 1
    fi
else
    echo "=========================================="
    echo "Using existing motion-augmented dataset"
    echo "=========================================="
    echo "Dataset: $DATASET_JSON"
    echo ""
fi

echo "=========================================="
echo "SFT Training Configuration"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON"
echo "Output: $OUT_DIR"
echo "GPUs: 4 (DeepSpeed ZeRO-2)"
echo "Expected Duration: ~6-8 hours"
echo "Time Limit: 7 days (167:59:00)"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Multi-GPU training with DeepSpeed ZeRO-2
srun torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    training/train_sft.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "$DATASET_JSON" \
    --deepspeed configs/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 10 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $EXP_NAME \
    --save_steps 500 \
    --max_grad_norm 5 \
    --save_only_model true

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job Complete!"
echo "=========================================="
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "Model saved to: $OUT_DIR"
echo "=========================================="

exit $EXIT_CODE
