#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=11:59:59
#SBATCH --job-name=dora
#SBATCH --mem=64GB
#SBATCH --ntasks=8
#SBATCH --output=visual-only.%j.out
#SBATCH --error=visual-only.%j.err

cd /projects/XXXX-2-storage/Workspace/dora
source env_dora/bin/activate
cd dora
python scripts/train_grpo_dora_vl_clean.py /scratch/XXXX-6.XXXX-7/grpo_dataset_updatedv2 --output-dir ./outputs/visual-only --use-lora --use-frames --visual-only --num-generations 8 --max-steps 5000 --batch-size 16 --gradient-accumulation-steps 4 --learning-rate 1e-4 --max-prompt-length 512 --max-response-length 256 --save-steps 50 --dataloader-num-workers 4 --kl-beta 0.01 --reward-weights 2.0