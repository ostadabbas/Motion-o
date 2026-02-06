#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --job-name=dora-eval-gemini-t
#SBATCH --mem=64GB
#SBATCH --ntasks=8
#SBATCH --output=dora-eval-gemini-t.%j.out
#SBATCH --error=dora-eval-gemini-t.%j.err

cd /projects/XXXX-2-storage/Workspace/dora
source env_dora/bin/activate
cd dora
python scripts/eval_mcq_gemini-thinking.py
