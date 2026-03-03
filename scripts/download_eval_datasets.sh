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
pip install -q huggingface_hub[cli]

DOWNLOAD_DIR="${1:-/scratch/bai.xiang/eval_benchmarks}"

echo "=========================================="
echo "Downloading evaluation benchmarks"
echo "Target: $DOWNLOAD_DIR"
echo "=========================================="

mkdir -p "$DOWNLOAD_DIR"

# ---------- 1. V-STaR ----------
echo ""
echo "[1/4] V-STaR (V-STaR-Bench/V-STaR)"
VSTAR_DIR="$DOWNLOAD_DIR/V-STaR"
if [ -d "$VSTAR_DIR" ] && [ -f "$VSTAR_DIR/V_STaR_test.json" ]; then
    echo "  Already exists — skipping"
else
    huggingface-cli download V-STaR-Bench/V-STaR \
        --repo-type dataset \
        --local-dir "$VSTAR_DIR"
    echo "  Done: $VSTAR_DIR"
fi

# ---------- 2. Video-MME ----------
echo ""
echo "[2/4] Video-MME (lmms-lab/Video-MME)"
VMME_DIR="$DOWNLOAD_DIR/Video-MME"
if [ -d "$VMME_DIR" ]; then
    echo "  Already exists — skipping"
else
    huggingface-cli download lmms-lab/Video-MME \
        --repo-type dataset \
        --local-dir "$VMME_DIR"
    echo "  Done: $VMME_DIR"
fi

# ---------- 3. VideoMMMU ----------
echo ""
echo "[3/4] VideoMMMU (lmms-lab/VideoMMMU)"
VMMMU_DIR="$DOWNLOAD_DIR/VideoMMMU"
if [ -d "$VMMMU_DIR" ]; then
    echo "  Already exists — skipping"
else
    huggingface-cli download lmms-lab/VideoMMMU \
        --repo-type dataset \
        --local-dir "$VMMMU_DIR"
    echo "  Done: $VMMMU_DIR"
fi

# ---------- 4. WorldSense ----------
echo ""
echo "[4/4] WorldSense (honglyhly/WorldSense)"
WS_DIR="$DOWNLOAD_DIR/WorldSense"
if [ -d "$WS_DIR" ]; then
    echo "  Already exists — skipping"
else
    huggingface-cli download honglyhly/WorldSense \
        --repo-type dataset \
        --local-dir "$WS_DIR"
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
