#!/bin/bash
# SFT Training Script - FULL DATASET (5,696 samples)
# Production training with optimized settings

cd "$(dirname "$0")/.."

# Activate dora_cuda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dora_cuda

# Add workspace to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export DEBUG_MODE="false"
export WANDB_MODE="online"  # Enable online tracking for full training

# FULL TRAINING MODE - Using all samples
export QUICK_TEST="false"

# Memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Configuration - subset only (filtered to available videos)
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # HuggingFace model
EXP_NAME="sft_full_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/${EXP_NAME}"
DATA_ROOT="/mnt/data/stgr"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-SFT-subset-motion-v3.json"

echo "=========================================="
echo "Starting FULL SFT Training (DeepSpeed ZeRO-2)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON"
echo "Output: $OUT_DIR"
echo "GPUs: 4 (0,1,2,3)"
echo "DeepSpeed: configs/zero2.json"
echo "Epochs: 1 (standard for video VLM SFT)"
echo "Estimated time: ~6-8 hours on 4xA100"
echo "=========================================="
echo ""

# Multi-GPU training with DeepSpeed ZeRO-2 (matching Open-o3-Video)
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

echo ""
echo "=========================================="
echo "SFT Training Complete!"
echo "Model saved to: $OUT_DIR"
echo "=========================================="
