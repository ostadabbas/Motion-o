#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --job-name=dora-eval-v-llava
#SBATCH --mem=64GB
#SBATCH --ntasks=8
#SBATCH --output=dora-eval-v-llava.%j.out
#SBATCH --error=dora-eval-v-llava.%j.err

cd /projects/XXXX-2-storage/Workspace/dora
source env_llava/bin/activate
cd dora
python scripts/eval_mcq_v_llava.py