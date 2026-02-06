#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=7:59:59
#SBATCH --job-name=dora_eval
#SBATCH --mem=64GB
#SBATCH --ntasks=8
#SBATCH --output=dora-eval.%j.out
#SBATCH --error=dora-eval.%j.err

cd /projects/XXXX-2-storage/Workspace/dora
source env_dora/bin/activate
cd dora
python scripts/eval_mcq.py