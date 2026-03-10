#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=23:59:59
#SBATCH --job-name=eval_motion_tags
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/eval_motion_tags_%j.out
#SBATCH --error=logs/eval_motion_tags_%j.err

set -euo pipefail

# ==========================================================================
# Motion Tag Evaluation Pipeline
#
# Evaluates a LoRA-trained model on the SFT motion dataset to measure
# per-attribute accuracy of <motion/> tags (direction, speed, scale).
#
# Usage:
#   sbatch scripts/eval_motion_tags.sh <checkpoint_path> [max_samples]
#
# Examples:
#   sbatch scripts/eval_motion_tags.sh outputs/grpo_dense_t07_4737145/checkpoint-800
#   sbatch scripts/eval_motion_tags.sh outputs/open-o3_grpo_v2_1234/checkpoint-400 200
#
# The script will:
#   1. Merge the LoRA adapter into the base model
#   2. Run motion-tag evaluation on the SFT dataset
# ==========================================================================

cd /projects/zura-storage/Workspace/vlmm-mcot
mkdir -p logs

source /projects/zura-storage/Workspace/dora/env_eval/bin/activate
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd):$(pwd)/evaluation"

# ---- Configuration ----
DATA_ROOT="/scratch/bai.xiang/Open-o3-Video"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-SFT-motion-mixed.json"
# -----------------------

CHECKPOINT_PATH="${1:?Usage: sbatch scripts/eval_motion_tags.sh <checkpoint_path> [max_samples]}"
MAX_SAMPLES="${2:-}"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}/adapter_config.json" ]; then
    echo "ERROR: No adapter_config.json in $CHECKPOINT_PATH — not a LoRA checkpoint?"
    exit 1
fi

# Derive experiment name
DIR_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")
CKPT_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/checkpoint-/ckpt/')
EXP_NAME="${DIR_NAME}_${CKPT_NAME}"
MERGED_DIR="${CHECKPOINT_PATH}/merged"

echo "=========================================="
echo "Motion Tag Evaluation Pipeline"
echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Checkpoint:  $CHECKPOINT_PATH"
echo "Exp name:    $EXP_NAME"
echo "Merged dir:  $MERGED_DIR"
echo "Dataset:     $DATASET_JSON"
echo "Start:       $(date)"
echo "=========================================="

# ---------- Step 1: Merge LoRA adapter (uses env_grpo for PEFT) ----------
if [ -f "${MERGED_DIR}/config.json" ]; then
    echo ""
    echo "[Step 1/2] Merged model already exists at $MERGED_DIR — skipping merge."
else
    echo ""
    echo "[Step 1/2] Merging LoRA adapter into base model..."

    source /projects/zura-storage/Workspace/dora/env_grpo/bin/activate

    BASE_MODEL=$(python -c "
import json, os
ac = json.load(open('${CHECKPOINT_PATH}/adapter_config.json'))
print(ac['base_model_name_or_path'])
")
    echo "  Base model: $BASE_MODEL"

    if [ ! -d "$BASE_MODEL" ]; then
        echo "ERROR: Base model not found at $BASE_MODEL"
        echo "You may need to update the base_model_name_or_path in adapter_config.json"
        exit 1
    fi

    python scripts/merge_lora.py \
        --base "$BASE_MODEL" \
        --adapter "$CHECKPOINT_PATH" \
        --output "$MERGED_DIR"

    echo "  Merge complete!"

    # Switch back to eval env for inference
    source /projects/zura-storage/Workspace/dora/env_eval/bin/activate
fi

# ---------- Step 2: Evaluate motion tags ----------
echo ""
echo "[Step 2/2] Running motion tag evaluation..."

mkdir -p evaluation/logs/motion_tags_logs

SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    SAMPLES_ARG="--max_samples $MAX_SAMPLES"
    echo "  Running on subset: $MAX_SAMPLES samples"
fi

OUTPUT_FILE="evaluation/logs/motion_tags_logs/${EXP_NAME}_motion_eval.json"

python scripts/eval_motion_tags.py \
    --model_path "$MERGED_DIR" \
    --dataset_json "$DATASET_JSON" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 4 \
    $SAMPLES_ARG 2>&1 | tee "evaluation/logs/motion_tags_logs/${EXP_NAME}_motion_eval.log"

echo ""
echo "=========================================="
echo "Motion Tag Evaluation Complete!"
echo "Results: $OUTPUT_FILE"
echo "Log:     evaluation/logs/motion_tags_logs/${EXP_NAME}_motion_eval.log"
echo "End:     $(date)"
echo "=========================================="
