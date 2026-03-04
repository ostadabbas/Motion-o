#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=23:59:59
#SBATCH --job-name=eval_all
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/eval_all_%j.out
#SBATCH --error=logs/eval_all_%j.err

set -uo pipefail

# ==========================================================================
# Full Evaluation Pipeline: V-STaR + Video-MME + VideoMMMU + WorldSense
#
# Usage:
#   sbatch scripts/eval_all.sh <checkpoint_path> [max_samples]
#
# Examples:
#   sbatch scripts/eval_all.sh outputs/grpo_dense_t07_4737145/checkpoint-800 100   # quick
#   sbatch scripts/eval_all.sh outputs/grpo_dense_t07_4737145/checkpoint-800       # full
# ==========================================================================

cd /projects/zura-storage/Workspace/vlmm-mcot
mkdir -p logs

CHECKPOINT_PATH="${1:?Usage: sbatch scripts/eval_all.sh <checkpoint_path> [max_samples]}"
MAX_SAMPLES="${2:-}"

# ---- Dataset paths ----
DATA_ROOT="/scratch/bai.xiang/eval_benchmarks"
VSTAR_VIDEO_FOLDER="${DATA_ROOT}/V-STaR/videos"
VSTAR_ANNO_FILE="${DATA_ROOT}/V-STaR/V_STaR_test.json"
VMME_DATA_DIR="${DATA_ROOT}/Video-MME"
VMMMU_DATA_DIR="${DATA_ROOT}/VideoMMMU"
WS_DATA_DIR="${DATA_ROOT}/WorldSense"

NUM_EVAL_GPUS=1
GPU_IDS=0

# ---- Derive experiment name ----
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"; exit 1
fi

DIR_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")
CKPT_NAME=$(basename "$CHECKPOINT_PATH" | sed 's/checkpoint-/ckpt/')
EXP_NAME="${DIR_NAME}_${CKPT_NAME}"
MERGED_DIR="${CHECKPOINT_PATH}/merged"

SUBSET_MSG=""
if [ -n "$MAX_SAMPLES" ]; then
    SUBSET_MSG=" (subset: ${MAX_SAMPLES} samples per benchmark)"
fi

echo "=========================================="
echo "Full Evaluation Pipeline${SUBSET_MSG}"
echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID:-interactive}"
echo "Checkpoint:  $CHECKPOINT_PATH"
echo "Exp name:    $EXP_NAME"
echo "Merged dir:  $MERGED_DIR"
echo "Start:       $(date)"
echo "=========================================="

# ================================================================
# Step 1: Merge LoRA adapter (env_grpo)
# ================================================================
if [ -f "${MERGED_DIR}/config.json" ]; then
    echo ""
    echo "[Step 1/5] Merged model already exists — skipping."
else
    echo ""
    echo "[Step 1/5] Merging LoRA adapter..."
    source /projects/zura-storage/Workspace/dora/env_grpo/bin/activate

    BASE_MODEL=$(python -c "
import json
ac = json.load(open('${CHECKPOINT_PATH}/adapter_config.json'))
print(ac['base_model_name_or_path'])
")
    echo "  Base model: $BASE_MODEL"

    if [ ! -d "$BASE_MODEL" ]; then
        echo "ERROR: Base model not found at $BASE_MODEL"; exit 1
    fi

    python scripts/merge_lora.py \
        --base "$BASE_MODEL" \
        --adapter "$CHECKPOINT_PATH" \
        --output "$MERGED_DIR"
    echo "  Merge complete!"
fi

# ================================================================
# Switch to eval env for all benchmarks
# ================================================================
source /projects/zura-storage/Workspace/dora/env_eval/bin/activate
export PYTHONPATH="$(pwd):$(pwd)/evaluation"

# ================================================================
# Step 2: V-STaR
# ================================================================
echo ""
echo "[Step 2/5] V-STaR evaluation..."
mkdir -p ./evaluation/logs/vstar_logs

SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

NUM_GPUS=$NUM_EVAL_GPUS CUDA_VISIBLE_DEVICES=$GPU_IDS python ./evaluation/test/test_vstar_multi_images.py \
    --video_folder "$VSTAR_VIDEO_FOLDER" \
    --anno_file "$VSTAR_ANNO_FILE" \
    --result_file "./evaluation/logs/vstar_logs/${EXP_NAME}_vstar.json" \
    --model_path "$MERGED_DIR" \
    --model_kwargs ./evaluation/config/vstar.yaml \
    --think_mode $SAMPLES_ARG 2>&1 | tee "./evaluation/logs/vstar_logs/test_${EXP_NAME}_vstar.log" || echo "  V-STaR FAILED — continuing"

echo "  V-STaR done!"

# ================================================================
# Step 3: Video-MME
# ================================================================
echo ""
echo "[Step 3/5] Video-MME evaluation..."
mkdir -p ./evaluation/logs/videomme_logs

NUM_GPUS=$NUM_EVAL_GPUS CUDA_VISIBLE_DEVICES=$GPU_IDS python ./evaluation/test/test_videomme.py \
    --exp_name "${EXP_NAME}_mme" \
    --data_dir "$VMME_DATA_DIR" \
    --model_path "$MERGED_DIR" \
    --model_kwargs ./evaluation/config/video_mme.yaml \
    --N 1 \
    --vote 'majority_voting' \
    --think_mode 2>&1 | tee "./evaluation/logs/videomme_logs/${EXP_NAME}_mme.log" || echo "  Video-MME FAILED — continuing"

echo "  Video-MME done!"

# ================================================================
# Step 4: VideoMMMU
# ================================================================
echo ""
echo "[Step 4/5] VideoMMMU evaluation..."
mkdir -p ./evaluation/logs/videommmu_logs

NUM_GPUS=$NUM_EVAL_GPUS CUDA_VISIBLE_DEVICES=$GPU_IDS python ./evaluation/test/test_videommmu.py \
    --exp_name "${EXP_NAME}_videommmu" \
    --data_dir "$VMMMU_DATA_DIR" \
    --model_path "$MERGED_DIR" \
    --model_kwargs ./evaluation/config/video_mmmu.yaml \
    --N 1 \
    --vote 'majority_voting' \
    --think_mode 2>&1 | tee "./evaluation/logs/videommmu_logs/${EXP_NAME}_videommmu.log" || echo "  VideoMMMU FAILED — continuing"

echo "  VideoMMMU done!"

# ================================================================
# Step 5: WorldSense
# ================================================================
echo ""
echo "[Step 5/5] WorldSense evaluation..."
mkdir -p ./evaluation/logs/world_logs

NUM_GPUS=$NUM_EVAL_GPUS CUDA_VISIBLE_DEVICES=$GPU_IDS python ./evaluation/test/test_worldsense.py \
    --exp_name "${EXP_NAME}_wds" \
    --data_dir "$WS_DATA_DIR" \
    --model_path "$MERGED_DIR" \
    --model_kwargs ./evaluation/config/world_sense.yaml \
    --N 1 \
    --vote 'majority_voting' \
    --think_mode 2>&1 | tee "./evaluation/logs/world_logs/${EXP_NAME}_wds.log" || echo "  WorldSense FAILED — continuing"

echo "  WorldSense done!"

# ================================================================
# Summary
# ================================================================
echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to:"
echo "  V-STaR:     ./evaluation/logs/vstar_logs/${EXP_NAME}_vstar.json"
echo "  Video-MME:  ./evaluation/logs/videomme_logs/${EXP_NAME}_mme.log"
echo "  VideoMMMU:  ./evaluation/logs/videommmu_logs/${EXP_NAME}_videommmu.log"
echo "  WorldSense: ./evaluation/logs/world_logs/${EXP_NAME}_wds.log"
echo ""
echo "Note: V-STaR LLM-as-judge scoring skipped (needs 72B model)."
echo "      Run separately with 4+ GPUs if needed."
echo "End: $(date)"
echo "=========================================="
