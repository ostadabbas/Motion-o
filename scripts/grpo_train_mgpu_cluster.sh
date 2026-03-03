#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --time=23:59:59
#SBATCH --job-name=motionr1_grpo
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err

set -euo pipefail

echo "=========================================="
echo "GRPO Training - 2x H200 (LoRA on merged SFT)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2"
echo "Start: $(date)"
echo "=========================================="

cd /projects/zura-storage/Workspace/vlmm-mcot
mkdir -p logs

# Activate environment
source /projects/zura-storage/Workspace/dora/env_grpo/bin/activate
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DECORD_EOF_RETRY_MAX=20480

# Wandb
export WANDB_MODE="online"

# ---- Change these as needed ----
NUM_GPUS=2
MODEL_PATH="outputs/sft_h200_4403849/merged"
EXP_NAME="grpo_h200_${NUM_GPUS}gpu_${SLURM_JOB_ID}"
OUT_DIR="outputs/${EXP_NAME}"
DATA_ROOT="/scratch/bai.xiang/Open-o3-Video"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-RL-filtered.json"
# ---------------------------------

echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON"
echo "Output: $OUT_DIR"
echo "Effective batch size: ${NUM_GPUS} x 1 x 4 = $((NUM_GPUS * 4))"
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=12321 \
    training/train_grpo.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "$DATASET_JSON" \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 4 \
    --max_prompt_length 16384 \
    --max_completion_length 1024 \
    --max_pixels 802816 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_only_model true \
    --report_to wandb \
    --run_name $EXP_NAME \
    --temperature 0.7 \
    --reward_funcs ans_acc ans_tiou ans_viou thk_temporal_point thk_temporal_segment thk_spatial motion_trajectory format

echo ""
echo "GRPO Complete! Model at: $OUT_DIR"
echo "End: $(date)"