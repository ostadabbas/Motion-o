#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=23:59:59
#SBATCH --job-name=motionr1_grpo
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err

set -euo pipefail

echo "=========================================="
echo "GRPO Training - 1x H200 141GB (LoRA on merged SFT)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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

# Config
MODEL_PATH="outputs/sft_h200_4403849/merged"
EXP_NAME="grpo_h200_${SLURM_JOB_ID}"
OUT_DIR="outputs/${EXP_NAME}"
DATA_ROOT="/scratch/bai.xiang/Open-o3-Video"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-RL-filtered.json"

echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON"
echo "Output: $OUT_DIR"
echo "Effective batch size: 1 x 4 = 4"
echo ""

# H200 141GB memory budget (with LoRA, no separate ref model):
#   - Model (bf16):              ~14GB
#   - LoRA + optimizer states:   ~4GB
#   - Generation (8 completions): ~40GB
#   - Activations (grad ckpt):   ~25GB
#   - Pixel values + frames:     ~10GB
#   Total: ~93GB / 141GB available
#
# num_generations=8: much better advantage estimation
# More generations = more signal per sample = faster convergence

python training/train_grpo.py \
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
    --generation_batch_size 4 \
    --max_prompt_length 16384 \
    --max_completion_length 1024 \
    --max_pixels 802816 \
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
    --save_steps 500 \
    --save_only_model true \
    --report_to wandb \
    --run_name $EXP_NAME \
    --reward_funcs ans_acc ans_tiou ans_viou thk_temporal_point thk_temporal_segment thk_spatial motion_trajectory format

echo ""
echo "GRPO Complete! Model at: $OUT_DIR"
echo "End: $(date)"