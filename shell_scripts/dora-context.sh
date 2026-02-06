#!/bin/bash
#SBATCH -J MyJob                            # Job name
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 8                               # Number of tasks
#SBATCH -o output_%j.txt                    # Standard output file
#SBATCH -e error_%j.txt                     # Standard error file
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --partition=short

cd /projects/XXXX-2-storage/Workspace/dora
source env_dora/bin/activate
cd dora
python scripts/add_context_dora_vqa.py