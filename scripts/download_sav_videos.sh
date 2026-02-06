#!/bin/bash
# Download all SA-V training videos using authenticated links

OUTPUT_DIR="/mnt/data/plm_stc/raw"
SA_V_DIR="$OUTPUT_DIR/sa-v"

mkdir -p "$SA_V_DIR"
cd "$OUTPUT_DIR"

echo "========================================================================"
echo "Downloading All SA-V Training Videos (~448GB)"
echo "========================================================================"
echo

# Read tab-separated file (skip header line)
tail -n +2 /home/bi.ga/Workspace/vlmm-mcot/dataset_download_links.txt | while IFS=$'\t' read -r filename url; do
    # Skip non-training tar files (only download sav_NNN.tar)
    if [[ ! "$filename" =~ ^sav_[0-9]+\.tar$ ]]; then
        echo "Skipping: $filename"
        continue
    fi
    
    echo "----------------------------------------"
    echo "Downloading: $filename (~8GB)"
    echo "----------------------------------------"
    
    wget -c -O "$filename" "$url" || {
        echo "ERROR: Failed to download $filename"
        echo "Links may have expired - refresh from Meta's portal"
        continue
    }
    
    echo "Extracting $filename..."
    tar -xf "$filename" -C sa-v/
    
    # Move files from nested sav_train/sav_XXX/ to sa-v/ directly
    if [ -d "sa-v/sav_train" ]; then
        find sa-v/sav_train -type f -exec mv {} sa-v/ \;
        rm -rf sa-v/sav_train
    fi
    
    echo "✓ $filename complete"
    rm "$filename"  # Delete tar to save space (~8GB freed)
    
done

echo
echo "========================================================================"
echo "COMPLETE!"
echo "========================================================================"
echo "Videos: $(ls $SA_V_DIR/*.mp4 2>/dev/null | wc -l)"
echo "Manual masklets: $(ls $SA_V_DIR/*_manual.json 2>/dev/null | wc -l)"