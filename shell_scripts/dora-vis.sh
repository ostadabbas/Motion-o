#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=07:59:59
#SBATCH --job-name=dora
#SBATCH --mem=64GB
#SBATCH --ntasks=8
#SBATCH --output=dora-vis.%j.out
#SBATCH --error=dora-vis.%j.err

cd /projects/XXXX-2-storage/Workspace/dora
source env_dora/bin/activate
cd dora
python scripts/visualize.py --mcq-dataset mcq_dataset_updated_spatial_audit.json --videos-dir /scratch/XXXX-6.XXXX-7/dora_mp4/ --labels-dir /scratch/XXXX-6.XXXX-7/filtered_labels/ --output-dir ./visualize/comp --frames-per-category 100 --ckpt-pth ./outputs/train_q3/checkpoint-250/