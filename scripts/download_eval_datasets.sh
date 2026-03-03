#!/bin/bash
#SBATCH --job-name=download_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=short
#SBATCH --output=logs/download_eval_%j.out
#SBATCH --error=logs/download_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@northeastern.edu

set -euo pipefail

cd /projects/zura-storage/Workspace/vlmm-mcot
mkdir -p logs

source /projects/zura-storage/Workspace/dora/env_grpo/bin/activate

DOWNLOAD_DIR="${1:-/scratch/bai.xiang/eval_benchmarks}"

echo "=========================================="
echo "Downloading evaluation benchmarks"
echo "Target: $DOWNLOAD_DIR"
echo "=========================================="

mkdir -p "$DOWNLOAD_DIR"

# ---------- 1. V-STaR ----------
echo ""
echo "[1/4] V-STaR (V-STaR-Bench/V-STaR)"
echo "  Videos + annotations for spatio-temporal reasoning"
VSTAR_DIR="$DOWNLOAD_DIR/V-STaR"
if [ -d "$VSTAR_DIR" ] && [ -f "$VSTAR_DIR/V_STaR_test.json" ]; then
    echo "  Already exists at $VSTAR_DIR — skipping (delete to re-download)"
else
    cd "$DOWNLOAD_DIR"
    GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/datasets/V-STaR-Bench/V-STaR
    echo "  Done: $VSTAR_DIR"
fi

# ---------- 2. Video-MME ----------
echo ""
echo "[2/4] Video-MME (lmms-lab/Video-MME)"
echo "  900 videos for multi-modal evaluation"
VMME_DIR="$DOWNLOAD_DIR/Video-MME"
if [ -d "$VMME_DIR" ]; then
    echo "  Already exists at $VMME_DIR — skipping"
else
    cd "$DOWNLOAD_DIR"
    GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/datasets/lmms-lab/Video-MME
    echo "  Done: $VMME_DIR"
fi

# ---------- 3. VideoMMMU ----------
echo ""
echo "[3/4] VideoMMMU (lmms-lab/VideoMMMU)"
echo "  Multi-discipline professional video QA"
VMMMU_DIR="$DOWNLOAD_DIR/VideoMMMU"
if [ -d "$VMMMU_DIR" ]; then
    echo "  Already exists at $VMMMU_DIR — skipping"
else
    cd "$DOWNLOAD_DIR"
    GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/datasets/lmms-lab/VideoMMMU
    echo "  Done: $VMMMU_DIR"
fi

# ---------- 4. WorldSense ----------
echo ""
echo "[4/4] WorldSense (honglyhly/WorldSense)"
echo "  Omnimodal video understanding benchmark"
WS_DIR="$DOWNLOAD_DIR/WorldSense"
if [ -d "$WS_DIR" ]; then
    echo "  Already exists at $WS_DIR — skipping"
else
    cd "$DOWNLOAD_DIR"
    GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/datasets/honglyhly/WorldSense
    echo "  Done: $WS_DIR"
fi

echo ""
echo "=========================================="
echo "All downloads complete!"
echo ""
echo "Dataset locations:"
echo "  V-STaR:     $DOWNLOAD_DIR/V-STaR"
echo "  Video-MME:  $DOWNLOAD_DIR/Video-MME"
echo "  VideoMMMU:  $DOWNLOAD_DIR/VideoMMMU"
echo "  WorldSense: $DOWNLOAD_DIR/WorldSense"
echo ""
echo "Disk usage:"
du -sh "$DOWNLOAD_DIR"/* 2>/dev/null || true
echo "=========================================="
