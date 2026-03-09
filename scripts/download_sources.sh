#!/bin/bash
#SBATCH --job-name=download_sources
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=short
#SBATCH --output=logs/download_sources_%j.out
#SBATCH --error=logs/download_sources_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@northeastern.edu

set -euo pipefail

echo "=========================================="
echo "Downloading missing video sources"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

source /projects/zura-storage/Workspace/dora/env_grpo/bin/activate
cd /scratch/bai.xiang/Open-o3-Video/videos/videor1
mkdir -p logs

# Small ones first (quick diversity wins)
for dir in PerceptionTest CLEVRER NeXT-QA STAR; do
    echo ""
    echo "=== Downloading $dir === $(date)"
    huggingface-cli download Video-R1/Video-R1-data \
        --repo-type dataset \
        --include "$dir/**" \
        --local-dir .
    echo "=== $dir done === $(date)"
done

# Big one last
echo ""
echo "=== Downloading LLaVA-Video-178K === $(date)"
huggingface-cli download Video-R1/Video-R1-data \
    --repo-type dataset \
    --include "LLaVA-Video-178K/**" \
    --local-dir .
echo "=== LLaVA-Video-178K done === $(date)"

# Verify what we got
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
for dir in PerceptionTest CLEVRER NeXT-QA STAR LLaVA-Video-178K; do
    count=$(find "$dir" -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.avi" \) 2>/dev/null | wc -l)
    echo "  $dir: $count video files"
done

echo ""
echo "Download complete! $(date)"