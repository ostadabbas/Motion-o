#!/bin/bash
#SBATCH --job-name=extract_espresso
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=short
#SBATCH --output=logs/extract_espresso_%j.out
#SBATCH --error=logs/extract_espresso_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@northeastern.edu

set -euo pipefail

echo "=========================================="
echo "Extracting VideoEspresso split zips"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

cd /scratch/bai.xiang/Open-o3-Video/sources/VideoEspresso_train_video

# Step 1: Combine split zips into one
echo "Combining split zips..."
zip -s 0 VideoEspresso_train_video.zip --out combined.zip
echo "Combined zip size: $(du -h combined.zip | cut -f1)"

# Step 2: Extract
echo "Extracting..."
unzip -o combined.zip -d extracted/
echo "Extraction done"

# Step 3: Find where the videos actually are
echo ""
echo "=== Looking for video directories ==="
find extracted/ -maxdepth 6 -type d | head -30

echo ""
echo "=== Looking for video files ==="
find extracted/ -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.avi" \) | head -10
video_count=$(find extracted/ -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.avi" \) | wc -l)
echo "Total video files: $video_count"

# Step 4: Check for expected directories
echo ""
echo "=== Checking for expected dirs ==="
for dir in Youcook2 Moviechat CUVA XD-Violence; do
    found=$(find extracted/ -type d -name "$dir" 2>/dev/null | head -1)
    echo "$dir: ${found:-NOT FOUND}"
done

echo ""
echo "Complete! $(date)"