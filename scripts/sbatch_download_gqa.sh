#!/bin/bash
#SBATCH --job-name=download_gqa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --output=download_gqa_%j.out
#SBATCH --error=download_gqa_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@northeastern.edu

set -e
echo "=== Started at $(date) ==="

DATA_ROOT="/scratch/bai.xiang/Open-o3-Video"
GQA_DIR="${DATA_ROOT}/videos/gqa"

mkdir -p "$GQA_DIR"
cd "$GQA_DIR"

# Download GQA images (20.3 GB)
if [ -f "images.zip" ] || [ -d "images" ]; then
    echo "GQA images already downloaded or extracted, checking..."
else
    echo "Downloading GQA images (20.3 GB)..."
    wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
    echo "Download complete"
fi

# Extract
if [ -d "images" ] && [ "$(ls images/*.jpg 2>/dev/null | head -1)" ]; then
    echo "Already extracted, skipping"
else
    echo "Extracting images.zip..."
    unzip -o -q images.zip
    echo "Extraction complete"
fi

# The JSON references image paths like "2331819.jpg" directly
# GQA extracts to images/ directory, so we need the files at gqa/2331819.jpg
# Move images up one level
if [ -d "images" ]; then
    echo "Moving images from images/ to gqa/ root..."
    mv images/*.jpg . 2>/dev/null || true
    rmdir images 2>/dev/null || true
    echo "Done"
fi

# Cleanup
if [ -f "images.zip" ]; then
    echo "Removing images.zip to save space..."
    rm images.zip
fi

echo ""
echo "=== Finished at $(date) ==="
echo "GQA images at: $GQA_DIR"
echo "Sample files:"
ls "$GQA_DIR" | head -10
echo "Total images: $(ls "$GQA_DIR"/*.jpg 2>/dev/null | wc -l)"