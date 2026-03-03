#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=23:59:59
#SBATCH --job-name=eval_vstar
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/eval_vstar_%j.out
#SBATCH --error=logs/eval_vstar_%j.err

set -euo pipefail

# ==========================================================================
# V-STaR Evaluation Pipeline
#
# Usage:
#   sbatch scripts/eval_vstar.sh <checkpoint_path>
#
# Examples:
#   sbatch scripts/eval_vstar.sh outputs/grpo_dense_t07_4737145/checkpoint-800
#   sbatch scripts/eval_vstar.sh outputs/open-o3_grpo_v2_1234/checkpoint-400
#
# The script will:
#   1. Merge the LoRA adapter into the base model
#   2. Run V-STaR inference (multi-GPU)
#   3. Run LLM-as-judge scoring
# ==========================================================================

cd /projects/zura-storage/Workspace/vlmm-mcot
mkdir -p logs

# Use env_eval for vLLM-based evaluation (separate from env_grpo training env)
source /projects/zura-storage/Workspace/dora/env_eval/bin/activate
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd):$(pwd)/evaluation"

# ---- Configuration (update these paths for your cluster) ----
VSTAR_VIDEO_FOLDER="/scratch/bai.xiang/eval_benchmarks/V-STaR/videos"
VSTAR_ANNO_FILE="/scratch/bai.xiang/eval_benchmarks/V-STaR/V_STaR_test.json"
LLM_PATH="Qwen/Qwen2.5-72B-Instruct"
NUM_EVAL_GPUS=1
# ---------------------------------------------------------------

CHECKPOINT_PATH="${1:?Usage: sbatch scripts/eval_vstar.sh <checkpoint_path>}"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}/adapter_config.json" ]; then
    echo "ERROR: No adapter_config.json in $CHECKPOINT_PATH — not a LoRA checkpoint?"
    exit 1
fi

# Derive experiment name from checkpoint path
# e.g. outputs/grpo_dense_t07_4737145/checkpoint-800 -> grpo_dense_t07_4737145_ckpt800
DIR_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")
CKPT_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/checkpoint-/ckpt/')
EXP_NAME="${DIR_NAME}_${CKPT_NAME}"
MERGED_DIR="${CHECKPOINT_PATH}/merged"

echo "=========================================="
echo "V-STaR Evaluation Pipeline"
echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Checkpoint:  $CHECKPOINT_PATH"
echo "Exp name:    $EXP_NAME"
echo "Merged dir:  $MERGED_DIR"
echo "Eval GPUs:   $NUM_EVAL_GPUS"
echo "Start:       $(date)"
echo "=========================================="

# ---------- Step 1: Merge LoRA adapter (uses env_grpo for PEFT) ----------
if [ -f "${MERGED_DIR}/config.json" ]; then
    echo ""
    echo "[Step 1/3] Merged model already exists at $MERGED_DIR — skipping merge."
else
    echo ""
    echo "[Step 1/3] Merging LoRA adapter into base model..."

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

# ---------- Step 2: V-STaR inference ----------
echo ""
echo "[Step 2/3] Running V-STaR inference with $NUM_EVAL_GPUS GPUs..."

mkdir -p ./evaluation/logs/vstar_logs

GPU_IDS=$(seq -s, 0 $((NUM_EVAL_GPUS - 1)))

MODEL_KWARGS="./evaluation/config/vstar.yaml"
NUM_GPUS=$NUM_EVAL_GPUS CUDA_VISIBLE_DEVICES=$GPU_IDS python ./evaluation/test/test_vstar_multi_images.py \
    --video_folder "$VSTAR_VIDEO_FOLDER" \
    --anno_file "$VSTAR_ANNO_FILE" \
    --result_file "./evaluation/logs/vstar_logs/${EXP_NAME}_vstar.json" \
    --model_path "$MERGED_DIR" \
    --model_kwargs $MODEL_KWARGS \
    --think_mode 2>&1 | tee "./evaluation/logs/vstar_logs/test_${EXP_NAME}_vstar.log"

echo "  V-STaR inference complete!"

# ---------- Step 3: LLM-as-judge scoring ----------
echo ""
echo "[Step 3/3] Running LLM-as-judge scoring..."

CUDA_VISIBLE_DEVICES=$GPU_IDS python ./evaluation/test/eval_vstar.py \
    --result_file "./evaluation/logs/vstar_logs/${EXP_NAME}_vstar.json" \
    --model_path "$LLM_PATH" 2>&1 | tee "./evaluation/logs/vstar_logs/eval_${EXP_NAME}_vstar.log"

echo ""
echo "=========================================="
echo "V-STaR Evaluation Complete!"
echo "Results: ./evaluation/logs/vstar_logs/${EXP_NAME}_vstar.json"
echo "Scores:  ./evaluation/logs/vstar_logs/eval_${EXP_NAME}_vstar.log"
echo "End:     $(date)"
echo "=========================================="
