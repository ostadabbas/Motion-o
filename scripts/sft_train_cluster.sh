#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=23:59:59
#SBATCH --job-name=motionr1_sft
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

set -euo pipefail

echo "=========================================="
echo "SFT Training - 1x H200"
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

# Video reader
export VIDEO_READER_BACKEND=decord

# Wandb
export WANDB_MODE="online"

# Config
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
EXP_NAME="sft_h200_${SLURM_JOB_ID}"
OUT_DIR="outputs/${EXP_NAME}"
DATA_ROOT="/scratch/bai.xiang/Open-o3-Video"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-SFT-filtered-motion.json"

echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON"
echo "Output: $OUT_DIR"
echo ""

# Single GPU - full fine-tuning (H200 has 141GB, no LoRA needed)
python training/train_sft.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "$DATASET_JSON" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
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
echo "SFT Complete! Model at: $OUT_DIR"
echo "End: $(date)"