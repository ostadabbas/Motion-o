#!/bin/bash
# Debug run: reuse QUICK_TEST (train_grpo.py .select(max_samples)), same dataset as full run.

cd "$(dirname "$0")/.."

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dora_cuda

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_MODE="offline"

export QUICK_TEST="true"
export MAX_SAMPLES="10"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Use same checkpoint as full run (sft_full_* = trained on <think>/answer format). Fallback: sft_subset, then any sft_*
MODEL_PATH="outputs/sft_full_slurm_639"

EXP_NAME="rl_motion_debug"
OUT_DIR="outputs/${EXP_NAME}"
DATA_ROOT="/mnt/data/stgr"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-RL-subset.json"

echo "=========================================="
echo "QUICK_TEST: 10 samples"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON"
echo "Output: $OUT_DIR"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12321 \
    training/train_grpo.py \
    --output_dir "$OUT_DIR" \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATASET_JSON" \
    --deepspeed configs/zero2.json \
    --use_peft true --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --max_prompt_length 16384 --max_completion_length 768 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 --lr_scheduler_type cosine --weight_decay 0.01 \
    --bf16 true --logging_steps 1 --gradient_checkpointing true \
    --attn_implementation eager --max_pixels 401408 \
    --num_train_epochs 1 --run_name "$EXP_NAME" --save_steps 500 \
    --beta 0.04 --max_grad_norm 5 --save_only_model true \
    --num_generations 2 \
    --reward_funcs ans_acc ans_tiou ans_viou thk_temporal_point thk_temporal_segment thk_spatial motion_trajectory format
