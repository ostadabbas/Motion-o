#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=23:59:59
#SBATCH --job-name=dora
#SBATCH --mem=64GB
#SBATCH --ntasks=8
#SBATCH --output=dora-train.%j.out
#SBATCH --error=dora-train.%j.err

cd /projects/XXXX-2-storage/Workspace/dora
source env_dora/bin/activate
cd dora
python scripts/train_finetune.py /scratch/XXXX-6.XXXX-7/grpo_dataset_updatedv2 --output-dir ./outputs/train_sft --use-lora