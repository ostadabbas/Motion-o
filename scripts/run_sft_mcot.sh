#!/bin/bash
# SFT Training Script with Motion Chain of Thought (MCoT)
# Uses augmented dataset with <motion> tags

cd "$(dirname "$0")/.."

# Activate dora_cuda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dora_cuda

# Add workspace to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export DEBUG_MODE="true"
export WANDB_MODE="offline"

# Quick test mode: set to "true" to test with only 10 samples
export QUICK_TEST="true"
export MAX_SAMPLES="10"  # Number of samples to use in quick test mode

# Memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Configuration
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # HuggingFace model
EXP_NAME="sft_mcot_test"
OUT_DIR="outputs/${EXP_NAME}"

# Use Motion-augmented dataset (10 samples for quick test)
DATA_ROOT="/mnt/data/stgr"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-SFT-motion-test.json"

echo "=========================================="
echo "Starting SFT Training with MCoT"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON (10 motion-augmented samples)"
echo "Output: $OUT_DIR"
echo "GPUs: 4 (0,1,2,3)"
echo "DeepSpeed: configs/zero2.json"
echo "Motion Chain of Thought: ENABLED"
echo "=========================================="
echo ""

# Multi-GPU training with DeepSpeed ZeRO-2
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    training/train_sft.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "$DATASET_JSON" \
    --deepspeed configs/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $EXP_NAME \
    --save_steps 500 \
    --max_grad_norm 5 \
    --save_only_model true
