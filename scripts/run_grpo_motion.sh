#!/bin/bash
# GRPO Training - With Motion Reward (Our Contribution)
# Using subset dataset with 5,819 available samples

cd "$(dirname "$0")/.."

# Activate dora_cuda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dora_cuda

# Add workspace to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export WANDB_MODE="offline"

# Quick test mode: set to "true" to test with only 10 samples
export QUICK_TEST="true"
export MAX_SAMPLES="10"  # Number of samples to use in quick test mode

# Memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Configuration
MODEL_PATH="outputs/sft_subset"  # Use SFT checkpoint from previous step
EXP_NAME="rl_motion_subset"
OUT_DIR="outputs/${EXP_NAME}"

# Use filtered subset dataset
DATA_ROOT="/mnt/data/stgr"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-RL-subset.json"

echo "=========================================="
echo "Starting Motion-Aware GRPO Training"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON (5,819 samples)"
echo "Output: $OUT_DIR"
echo "Rewards: Including motion_trajectory"
echo "=========================================="
echo ""

# Train WITH motion_trajectory reward (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12321" \
    training/train_grpo.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "$DATASET_JSON" \
    --deepspeed "configs/zero2.json" \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 true \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --attn_implementation eager \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name $EXP_NAME \
    --save_steps 500 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model true \
    --num_generations 2 \
    --reward_funcs ans_acc ans_tiou ans_viou thk_temporal_point thk_temporal_segment thk_spatial motion_trajectory format
    # ⬆️ LoRA enabled + num_generations=2 for memory efficiency
