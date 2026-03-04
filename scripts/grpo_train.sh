#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=23:59:59
#SBATCH --job-name=motionr1_grpo
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/grpo_motion_%j.out
#SBATCH --error=logs/grpo_motion_%j.err

set -euo pipefail

echo "=========================================="
echo "GRPO Motion Training - 1x H200"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd /projects/zura-storage/Workspace/vlmm-mcot
mkdir -p logs

# Activate environment
source /projects/zura-storage/Workspace/dora/env_dora/bin/activate
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wandb
export WANDB_MODE="online"

# Find SFT checkpoint - override with: SFT_CHECKPOINT=path sbatch ...
MODEL_PATH="${SFT_CHECKPOINT:-}"
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
    MODEL_PATH=$(ls -td outputs/sft_h200_* 2>/dev/null | head -1)
fi
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: No SFT checkpoint found. Set SFT_CHECKPOINT=path"
    exit 1
fi

EXP_NAME="grpo_motion_h200_${SLURM_JOB_ID}"
OUT_DIR="outputs/${EXP_NAME}"
DATA_ROOT="/scratch/bai.xiang/Open-o3-Video"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-RL-filtered.json"

echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON"
echo "Output: $OUT_DIR"
echo ""

# Single GPU - GRPO with LoRA
# Note: GRPO needs more memory than SFT due to multiple generations
# H200 (141GB) should handle this with gradient checkpointing + LoRA
python training/train_grpo.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "$DATASET_JSON" \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --bf16 true \
    --logging_steps 10 \
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
    --gen_temperature 0.7 \
    --reward_funcs ans_acc ans_tiou ans_viou thk_temporal_point thk_temporal_segment thk_spatial motion_trajectory format

EXIT_CODE=$?

echo ""
echo "GRPO Complete! Model at: $OUT_DIR"
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"
exit $EXIT_CODE