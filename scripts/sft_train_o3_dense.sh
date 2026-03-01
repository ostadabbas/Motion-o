#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=23:59:59
#SBATCH --job-name=open-o3_motion_sft
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/open-o3_motion_sft_%j.out
#SBATCH --error=logs/open-o3_motion_sft_%j.err

set -euo pipefail

echo "=========================================="
echo "Open-o3 Motion SFT — Mixed Dataset"
echo "1x H200 (LoRA)"
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
MODEL_PATH="/projects/zura-storage/Workspace/vlmm-mcot/Open-o3-Video/Qwen2.5-VL-open-o3"
EXP_NAME="open-o3_motion_sft_${SLURM_JOB_ID}"
OUT_DIR="outputs/${EXP_NAME}"
DATA_ROOT="/scratch/bai.xiang/Open-o3-Video"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-SFT-motion-mixed.json"

echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON"
echo "Output: $OUT_DIR"
echo ""

# ── Validate dataset before training ──────────────────────────
echo "Validating dataset..."
python -c "
import json
with open('${DATASET_JSON}') as f:
    data = json.load(f)
motion_count = sum(1 for s in data if '<motion ' in s.get('reasoning_process', ''))
print(f'Total: {len(data)}, with motion: {motion_count}')
assert motion_count > 0, 'No motion tags found!'
print('Validation passed!')
"
echo ""

python training/train_sft_v2.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "$DATASET_JSON" \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --logging_steps 10 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $EXP_NAME \
    --save_steps 500 \
    --max_grad_norm 5 \
    --save_only_model true \
    --seed 42

echo ""
echo "Open-o3 Motion SFT Complete! Model at: $OUT_DIR"
echo "End: $(date)"