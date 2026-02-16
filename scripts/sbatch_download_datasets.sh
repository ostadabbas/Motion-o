#!/bin/bash
#SBATCH --job-name=download_sources
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=short
#SBATCH --output=download_sources_%j.out
#SBATCH --error=download_sources_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@northeastern.edu

echo "=== Download started at $(date) ==="
echo "Running on node: $(hostname)"

# Activate environment
source /projects/zura-storage/Workspace/dora/env_dora/bin/activate

# Navigate to project directory
cd /projects/zura-storage/Workspace/vlmm-mcot

DATA_ROOT="/scratch/bai.xiang/Open-o3-Video"

# ============================================================
# 1. TimeR1-Dataset -> videos/timerft
# ============================================================
TIMERFT_DIR="${DATA_ROOT}/videos/timerft"
if [ -d "$TIMERFT_DIR" ] && [ "$(ls -A $TIMERFT_DIR 2>/dev/null)" ]; then
    echo "=== timerft already exists, skipping ==="
else
    echo "=== Downloading Boshenxx/TimeR1-Dataset -> videos/timerft ==="
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Boshenxx/TimeR1-Dataset',
    repo_type='dataset',
    local_dir='${DATA_ROOT}/sources/TimeR1-Dataset',
)
print('TimeR1-Dataset download complete')
"
    echo "=== TimeR1-Dataset downloaded ==="
fi

# ============================================================
# 2. TreeVGR-RL-37K -> videos/treevgr
# ============================================================
TREEVGR_DIR="${DATA_ROOT}/videos/treevgr"
if [ -d "$TREEVGR_DIR" ] && [ "$(ls -A $TREEVGR_DIR 2>/dev/null)" ]; then
    echo "=== treevgr already exists, skipping ==="
else
    echo "=== Downloading HaochenWang/TreeVGR-RL-37K -> videos/treevgr ==="
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='HaochenWang/TreeVGR-RL-37K',
    repo_type='dataset',
    local_dir='${DATA_ROOT}/sources/TreeVGR-RL-37K',
)
print('TreeVGR-RL-37K download complete')
"
    echo "=== TreeVGR-RL-37K downloaded ==="
fi

# ============================================================
# 3. Video-R1-data -> videos/videor1
# ============================================================
VIDEOR1_DIR="${DATA_ROOT}/videos/videor1"
if [ -d "$VIDEOR1_DIR" ] && [ "$(ls -A $VIDEOR1_DIR 2>/dev/null)" ]; then
    echo "=== videor1 already exists, skipping ==="
else
    echo "=== Downloading Video-R1/Video-R1-data -> videos/videor1 ==="
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Video-R1/Video-R1-data',
    repo_type='dataset',
    local_dir='${DATA_ROOT}/sources/Video-R1-data',
)
print('Video-R1-data download complete')
"
    echo "=== Video-R1-data downloaded ==="
fi

echo ""
echo "=== All downloads finished at $(date) ==="
echo ""
echo "Downloaded source datasets to: ${DATA_ROOT}/sources/"
echo ""
echo "IMPORTANT: You will need to organize/symlink the downloaded media"
echo "into the expected directory structure:"
echo "  ${DATA_ROOT}/videos/timerft    <- from TimeR1-Dataset"
echo "  ${DATA_ROOT}/videos/treevgr    <- from TreeVGR-RL-37K (images/)"
echo "  ${DATA_ROOT}/videos/videor1    <- from Video-R1-data"
echo ""
echo "Check what was downloaded:"
echo "  ls -la ${DATA_ROOT}/sources/"
echo "  tree -L 2 ${DATA_ROOT}/sources/"