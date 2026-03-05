#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --time=23:59:59
#SBATCH --job-name=open-o3_grpo_v3
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/open-o3_grpo_v3_%j.out
#SBATCH --error=logs/open-o3_grpo_v3_%j.err

set -euo pipefail

echo "=========================================="
echo "Open-o3 GRPO v3 — Dual-Chain Grounding"
echo "2x H200 141GB"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /projects/zura-storage/Workspace/vlmm-mcot
mkdir -p logs

source /projects/zura-storage/Workspace/dora/env_grpo/bin/activate
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DECORD_EOF_RETRY_MAX=20480
export WANDB_MODE="online"
export DEBUG_MOTION_REWARD=1
export DEBUG_GROUNDING_REWARD=1

# Config
MODEL_PATH="outputs/open-o3_motion_sft_4666166/merged"
EXP_NAME="open-o3_grpo_v3_${SLURM_JOB_ID}"
OUT_DIR="outputs/${EXP_NAME}"
DATA_ROOT="/scratch/bai.xiang/Open-o3-Video"
# DATASET_JSON="${DATA_ROOT}/json_data/STGR-RL-filtered-motion-densebbox.json"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-RL-dense-5k-mixed.json"
echo "Model:   $MODEL_PATH"
echo "Dataset: $DATASET_JSON"
echo "Output:  $OUT_DIR"

# Auto-resume
RESUME_ARG=""
if [ -d "$OUT_DIR" ]; then
    LATEST_CKPT=$(ls -d ${OUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
        echo "Resuming from: $LATEST_CKPT"
    fi
fi
echo ""

torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=12322 \
    training/train_grpo_v3.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "$DATASET_JSON" \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_generations 4 \
    --generation_batch_size 4 \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --max_pixels 401408 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation eager \
    --num_train_epochs 1 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --logging_steps 25 \
    --save_steps 200 \
    --save_only_model true \
    --report_to wandb \
    --run_name $EXP_NAME \
    --seed 42 \
    --gen_temperature 1.0 \
    $RESUME_ARG \
    --reward_funcs ans_acc ans_tiou ans_viou thk_temporal_point thk_temporal_segment thk_spatial motion_trajectory motion_grounding format

echo ""
echo "=========================================="
echo "Open-o3 GRPO v3 Dual-Chain Complete! Model at: $OUT_DIR"
echo "End: $(date)"
echo "=========================================="