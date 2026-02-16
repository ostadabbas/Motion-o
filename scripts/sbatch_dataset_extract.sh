#!/bin/bash
#SBATCH --job-name=organize_extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=short
#SBATCH --output=organize_%j.out
#SBATCH --error=organize_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@northeastern.edu

set -e
echo "=== Started at $(date) ==="

DATA_ROOT="/scratch/bai.xiang/Open-o3-Video"
SOURCES="${DATA_ROOT}/sources"
VIDEOS="${DATA_ROOT}/videos"

mkdir -p "$VIDEOS"

# ============================================================
# 1. TimeR1 -> videos/timerft
#    TimeRFT_data.zip extracts to timerft_data/*.mp4
# ============================================================
if [ -d "$VIDEOS/timerft" ] && [ "$(ls $VIDEOS/timerft/*.mp4 2>/dev/null | head -1)" ]; then
    echo "=== videos/timerft already exists, skipping ==="
else
    echo "=== Extracting TimeRFT_data.zip -> videos/timerft ==="
    cd "$VIDEOS"
    unzip -o "${SOURCES}/TimeR1-Dataset/TimeRFT_data.zip"
    # Rename timerft_data -> timerft
    if [ -d "timerft_data" ]; then
        mv timerft_data timerft
    fi
    echo "=== TimeR1 done ==="
fi

# ============================================================
# 2. TreeVGR -> videos/treevgr
#    images.tar.gz extracts to images/*.jpg
# ============================================================
if [ -d "$VIDEOS/treevgr" ] && [ "$(ls $VIDEOS/treevgr/images/*.jpg 2>/dev/null | head -1)" ]; then
    echo "=== videos/treevgr already exists, skipping ==="
else
    echo "=== Extracting TreeVGR images.tar.gz -> videos/treevgr ==="
    mkdir -p "$VIDEOS/treevgr"
    cd "$VIDEOS/treevgr"
    tar -xzf "${SOURCES}/TreeVGR-RL-37K/images.tar.gz"
    echo "=== TreeVGR done ==="
fi

# ============================================================
# 3. Video-R1 -> symlink subdirectories into videos/
#    The STGR JSONs reference paths like:
#      LLaVA-Video-178K/...
#      Moviechat/...
#      Youcook2/...
#      GroundedVLLM/...
#      CLEVRER/...
#      NeXT-QA/...
#      PerceptionTest/...
#      STAR/...
#      activitynet/...  (note: lowercase in path)
#      coin/...
#      DiDeMo/...
#      querYD/...
#      qvhighlights/...
# ============================================================
echo "=== Extracting Video-R1 zip files ==="

# Extract all zip files in Video-R1-data subdirectories
for zipdir in CLEVRER LLaVA-Video-178K NeXT-QA PerceptionTest STAR; do
    target="${SOURCES}/Video-R1-data/${zipdir}"
    if [ -d "$target" ]; then
        echo "--- Extracting zips in ${zipdir} ---"
        cd "$target"
        for z in *.zip; do
            if [ -f "$z" ]; then
                echo "  Extracting $z ..."
                unzip -o -q "$z"
            fi
        done
    fi
done

echo "=== Creating symlinks for Video-R1 directories ==="

# Directories that need to be accessible under videos/
# These come from Video-R1-data and are referenced in the STGR JSONs
LINK_DIRS=(
    "CLEVRER"
    "LLaVA-Video-178K"
    "NeXT-QA"
    "PerceptionTest"
    "STAR"
    "Moviechat"
    "Youcook2"
    "GroundedVLLM"
    "activitynet"
    "coin"
    "DiDeMo"
    "querYD"
    "qvhighlights"
    "CUVA"
    "XD-Violence"
)

for dir in "${LINK_DIRS[@]}"; do
    src="${SOURCES}/Video-R1-data/${dir}"
    dst="${VIDEOS}/${dir}"
    if [ -e "$dst" ]; then
        echo "  $dir already exists in videos/, skipping"
    elif [ -d "$src" ]; then
        echo "  Symlinking $dir"
        ln -s "$src" "$dst"
    else
        echo "  WARNING: $src not found, skipping"
    fi
done

# ============================================================
# 4. Summary
# ============================================================
echo ""
echo "=== Finished at $(date) ==="
echo ""
echo "Directory structure:"
ls -la "$VIDEOS/"
echo ""
echo "Checking key directories:"
for d in timerft treevgr stgr LLaVA-Video-178K GroundedVLLM activitynet coin DiDeMo querYD qvhighlights videoespresso gqa; do
    p="$VIDEOS/$d"
    if [ -e "$p" ]; then
        echo "  ✓ $d"
    else
        echo "  ✗ $d MISSING"
    fi
done