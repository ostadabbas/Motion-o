#!/bin/bash
#SBATCH --job-name=extract_videos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=short
#SBATCH --output=logs/extract_videos_%j.out
#SBATCH --error=logs/extract_videos_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@northeastern.edu

set -euo pipefail

echo "=========================================="
echo "Extracting video zip files"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

ROOT="/scratch/bai.xiang/Open-o3-Video/videos/videor1"

for dir in PerceptionTest CLEVRER NeXT-QA STAR LLaVA-Video-178K; do
    echo ""
    echo "=== Extracting $dir === $(date)"
    cd "$ROOT/$dir"
    for z in *.zip; do
        [ -f "$z" ] || continue
        echo "  Unzipping $z..."
        unzip -n -q "$z"
    done
    echo "=== $dir done === $(date)"
done

echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
cd "$ROOT"
for dir in PerceptionTest CLEVRER NeXT-QA STAR LLaVA-Video-178K; do
    count=$(find "$dir" -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.avi" -o -name "*.webm" \) 2>/dev/null | wc -l)
    echo "  $dir: $count video files"
done

echo ""
echo "Extraction complete! $(date)"