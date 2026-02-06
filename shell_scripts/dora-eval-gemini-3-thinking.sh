#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=07:59:00
#SBATCH --job-name=dora-eval-gemini-3-thinking
#SBATCH --mem=64GB
#SBATCH --ntasks=8
#SBATCH --output=dora-eval-gemini-3-thinking.%j.out
#SBATCH --error=dora-eval-gemini-3-thinking.%j.err

cd /projects/XXXX-2-storage/Workspace/dora
source env_intern/bin/activate
cd dora
python scripts/eval_mcq_gemini-3.py --thinking