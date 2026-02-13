#!/bin/bash
#SBATCH --job-name=download_o3video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=short
#SBATCH --output=download_%j.out
#SBATCH --error=download_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@northeastern.edu

echo "=== Download started at $(date) ==="
echo "Running on node: $(hostname)"

# Activate environment
source /projects/zura-storage/Workspace/dora/env_dora/bin/activate

# Make sure huggingface_hub is available
pip install --user huggingface_hub --quiet

# Navigate to project directory
cd /projects/zura-storage/Workspace/vlmm-mcot

# Run the download script
python scripts/download_dataset.py --output-dir /scratch/bai.xiang/Open-o3-Video

echo "=== Download finished at $(date) ==="