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
python scripts/train_grpo_dora_vl_clean_q3.py /projects/XXXX-1/dora/grpo_dataset_updatedv2 --output-dir ./outputs/train_q3 --use-frames --use-lora --num-generations 8 --max-steps 5000 --batch-size 8 --gradient-accumulation-steps 4 --learning-rate 1e-4 --max-prompt-length 512 --max-response-length 256 --save-steps 5 --dataloader-num-workers 4 --kl-beta 0.01 --reward-weights 2.0