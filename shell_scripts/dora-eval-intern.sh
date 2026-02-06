#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --job-name=dora-eval-intern-2
#SBATCH --mem=64GB
#SBATCH --ntasks=8
#SBATCH --output=dora-eval-intern-2.%j.out
#SBATCH --error=dora-eval-intern-2.%j.err

cd /projects/XXXX-2-storage/Workspace/dora
source env_intern/bin/activate
cd dora
python scripts/eval_mcq_intern.py