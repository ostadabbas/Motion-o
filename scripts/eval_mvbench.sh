#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=23:59:59
#SBATCH --job-name=eval_mvbench
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/eval_mvbench_%j.out
#SBATCH --error=logs/eval_mvbench_%j.err

set -euo pipefail

# ==========================================================================
# MVBench Evaluation Pipeline
# https://huggingface.co/datasets/OpenGVLab/MVBench
#
# Usage:
#   sbatch scripts/eval_mvbench.sh <checkpoint_path> [video_dir]
#
# Examples:
#   sbatch scripts/eval_mvbench.sh outputs/grpo_dense_t07_4737145/checkpoint-800
#   sbatch scripts/eval_mvbench.sh outputs/.../checkpoint-800 /scratch/bai.xiang/MVBench/video
#
# Prerequisites:
#   - MVBench videos under video_dir. Layout: video_dir/<subset_name>/<filename>
#     or video_dir/<filename>. Download from HF dataset repo or clone and get
#     video/ tree; 320 NTU RGB+D videos need manual download (see dataset card).
# ==========================================================================

cd /projects/zura-storage/Workspace/vlmm-mcot
mkdir -p logs

source /projects/zura-storage/Workspace/dora/env_eval/bin/activate
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd):$(pwd)/evaluation"

# Default video root (override with second argument or env MVBENCH_VIDEO_DIR)
MVBENCH_VIDEO_DIR="${MVBENCH_VIDEO_DIR:-/scratch/bai.xiang/eval_benchmarks/MVBench/video}"

CHECKPOINT_PATH="${1:?Usage: sbatch scripts/eval_mvbench.sh <checkpoint_path> [video_dir]}"
VIDEO_DIR="${2:-$MVBENCH_VIDEO_DIR}"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$VIDEO_DIR" ]; then
    echo "ERROR: Video dir not found: $VIDEO_DIR"
    echo "Download MVBench videos (see https://huggingface.co/datasets/OpenGVLab/MVBench) and set video_dir."
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}/adapter_config.json" ]; then
    echo "ERROR: No adapter_config.json in $CHECKPOINT_PATH — not a LoRA checkpoint?"
    exit 1
fi

DIR_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")
CKPT_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/checkpoint-/ckpt/')
EXP_NAME="${DIR_NAME}_${CKPT_NAME}"
MERGED_DIR="${CHECKPOINT_PATH}/merged"

echo "=========================================="
echo "MVBench Evaluation Pipeline"
echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Checkpoint:  $CHECKPOINT_PATH"
echo "Merged dir:  $MERGED_DIR"
echo "Video dir:   $VIDEO_DIR"
echo "Start:       $(date)"
echo "=========================================="

# ---------- Step 1: Merge LoRA adapter ----------
if [ -f "${MERGED_DIR}/config.json" ]; then
    echo ""
    echo "[Step 1/2] Merged model already exists at $MERGED_DIR — skipping merge."
else
    echo ""
    echo "[Step 1/2] Merging LoRA adapter into base model..."
    source /projects/zura-storage/Workspace/dora/env_grpo/bin/activate

    BASE_MODEL=$(python -c "
import json
ac = json.load(open('${CHECKPOINT_PATH}/adapter_config.json'))
print(ac['base_model_name_or_path'])
")
    echo "  Base model: $BASE_MODEL"
    if [ ! -d "$BASE_MODEL" ]; then
        echo "ERROR: Base model not found at $BASE_MODEL"
        exit 1
    fi
    python scripts/merge_lora.py \
        --base "$BASE_MODEL" \
        --adapter "$CHECKPOINT_PATH" \
        --output "$MERGED_DIR"
    echo "  Merge complete!"
    source /projects/zura-storage/Workspace/dora/env_eval/bin/activate
fi

# ---------- Step 2: Run MVBench evaluation ----------
echo ""
echo "[Step 2/2] Running MVBench evaluation..."
mkdir -p evaluation/logs/mvbench_logs
OUTPUT_FILE="evaluation/logs/mvbench_logs/${EXP_NAME}_mvbench.json"

python scripts/eval_mvbench.py \
    --model_path "$MERGED_DIR" \
    --video_dir "$VIDEO_DIR" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 4 \
    2>&1 | tee "evaluation/logs/mvbench_logs/${EXP_NAME}_mvbench.log"

echo ""
echo "=========================================="
echo "MVBench Evaluation Complete!"
echo "Results: $OUTPUT_FILE"
echo "End:     $(date)"
echo "=========================================="
