#!/bin/bash
#SBATCH --job-name=dl_videoespresso
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=short
#SBATCH --output=dl_videoespresso_%j.out
#SBATCH --error=dl_videoespresso_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@northeastern.edu

set -euo pipefail

echo "=== VideoEspresso download started at $(date) ==="
echo "Running on node: $(hostname)"

source /projects/zura-storage/Workspace/dora/env_dora/bin/activate

DATA_ROOT="/scratch/bai.xiang/Open-o3-Video"
SRC_DIR="${DATA_ROOT}/sources/VideoEspresso_train_video"
VIDEOS_DIR="${DATA_ROOT}/videos"

# ============================================================
# Step 1: Download from HuggingFace (~206 GB, 48 zip parts)
# ============================================================
echo "=== Step 1: Downloading hshjerry0315/VideoEspresso_train_video ==="

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='hshjerry0315/VideoEspresso_train_video',
    repo_type='dataset',
    local_dir='${SRC_DIR}',
    resume_download=True,
)
print('Download complete')
"

echo "=== Download finished at $(date) ==="
ls -lh ${SRC_DIR}/VideoEspresso_train_video.z* | wc -l
echo "zip parts downloaded"

# ============================================================
# Step 2: Combine split zip and extract
# ============================================================
echo "=== Step 2: Combining and extracting split zip ==="
cd ${SRC_DIR}

# zip -s 0 combines split zip into single zip
zip -s 0 VideoEspresso_train_video.zip --out ${SRC_DIR}/combined.zip
echo "=== Combined at $(date) ==="

# Extract
unzip -o ${SRC_DIR}/combined.zip -d ${SRC_DIR}/extracted
echo "=== Extracted at $(date) ==="

# Clean up combined zip to save space
rm -f ${SRC_DIR}/combined.zip

# Show what we got
echo "=== Extracted contents: ==="
ls ${SRC_DIR}/extracted/
find ${SRC_DIR}/extracted -maxdepth 3 -type d | head -30

# ============================================================
# Step 3: Link into videos/
# ============================================================
echo "=== Step 3: Setting up symlinks ==="

for subdir in Moviechat Youcook2 CUVA XD-Violence; do
    found=$(find ${SRC_DIR}/extracted -maxdepth 3 -type d -name "$subdir" 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        # Remove old symlink/dir
        rm -rf "${VIDEOS_DIR}/${subdir}" 2>/dev/null || true
        ln -sfn "$found" "${VIDEOS_DIR}/${subdir}"
        echo "Linked: ${subdir} -> $found"
        echo "  Files: $(find -L ${VIDEOS_DIR}/${subdir} -name '*.mp4' | wc -l) mp4s"
    else
        echo "WARNING: ${subdir} not found in extracted files"
    fi
done

# ============================================================
# Step 4: Verify
# ============================================================
echo "=== Step 4: Verification ==="
python3 -c "
import json, os
R = '${DATA_ROOT}'
with open(f'{R}/json_data/STGR-RL.json') as f:
    data = json.load(f)
found = missing = 0
for item in data:
    if item.get('source') != 'videoespresso_train_video':
        continue
    vp = item.get('video_path','')
    if os.path.exists(f'{R}/videos/{vp}'):
        found += 1
    else:
        missing += 1
        if missing <= 3:
            print(f'  Still missing: {vp}')
total = found + missing
print(f'videoespresso: {found}/{total} ({found/total*100:.1f}%)')
"

echo "=== All done at $(date) ==="