#!/bin/bash
#SBATCH -J MyJob                            # Job name
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH --cpus-per-task 4
#SBATCH -o output_%j.txt                    # Standard output file
#SBATCH -e error_%j.txt                     # Standard error file
#SBATCH --mail-user=$USER@XXXX-4.XXXX-5  # Email
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --partition=short
#SBATCH --mail-type=ALL                     # Type of email notifications

cd /projects/XXXX-2-storage/Workspace/dora
source env_dora/bin/activate
cd dora/scripts
python generate_grpo_dataset.py \
    --labels-dir /projects/XXXX-1/dora/filtered_labels \
    --videos-dir /projects/XXXX-1/dora/mp4/mp4 \
    --srt-dir /projects/XXXX-1/dora/srt \
    --output-dir /projects/XXXX-1/dora/grpo_dataset_updated \
    --seasons 1 2 3 4 5 6 7 8 \
    --use-full-video 0.3   \
    --full-video-fps 0.2    \
    --full-video-max-frames 20   \
    --max-collages 2 \
    --num-workers 4