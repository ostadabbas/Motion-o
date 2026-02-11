#!/bin/bash
# Regenerate motion-augmented dataset with multi-frame filtering

echo "=========================================="
echo "Regenerating Motion-Augmented Dataset"
echo "=========================================="
echo ""

INPUT="/mnt/data/stgr/json_data/STGR-SFT-subset.json"
OUTPUT="/mnt/data/stgr/json_data/STGR-SFT-subset-motion-v2.json"
OLD_OUTPUT="/mnt/data/stgr/json_data/STGR-SFT-subset-motion.json"

echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo ""

# Backup old version
if [ -f "$OUTPUT" ]; then
    echo "Backing up existing output to ${OUTPUT}.bak"
    cp "$OUTPUT" "${OUTPUT}.bak"
fi

# Run augmentation with fixed script
echo "Running augmentation (multi-frame filtering enabled)..."
python scripts/augment_motion_data_simple.py \
    --input "$INPUT" \
    --output "$OUTPUT"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Dataset regenerated successfully!"
    echo "=========================================="
    echo ""
    
    # Compare statistics
    echo "Comparing old vs new:"
    echo ""
    
    python3 << 'EOF'
import json
import re

def count_motion_tags(data):
    total_samples = len(data)
    samples_with_motion = 0
    total_motion_tags = 0
    stationary_tags = 0
    actual_motion_tags = 0
    single_frame_tags = 0
    
    for sample in data:
        motion_tags = re.findall(r'<motion>([^<]+)</motion>', sample.get('reasoning_process', ''))
        if motion_tags:
            samples_with_motion += 1
            total_motion_tags += len(motion_tags)
            for tag in motion_tags:
                if 'single frame' in tag:
                    single_frame_tags += 1
                elif 'stationary' in tag:
                    stationary_tags += 1
                else:
                    actual_motion_tags += 1
    
    return {
        'total_samples': total_samples,
        'samples_with_motion': samples_with_motion,
        'total_motion_tags': total_motion_tags,
        'stationary_tags': stationary_tags,
        'actual_motion_tags': actual_motion_tags,
        'single_frame_tags': single_frame_tags
    }

# Old version
try:
    with open('/mnt/data/stgr/json_data/STGR-SFT-subset-motion.json') as f:
        old_data = json.load(f)
    old_stats = count_motion_tags(old_data)
    
    print("OLD Dataset (with single-frame noise):")
    print(f"  Total samples: {old_stats['total_samples']}")
    print(f"  Samples with motion tags: {old_stats['samples_with_motion']}")
    print(f"  Total motion tags: {old_stats['total_motion_tags']}")
    print(f"    - Actual motion: {old_stats['actual_motion_tags']} ({100*old_stats['actual_motion_tags']/max(1,old_stats['total_motion_tags']):.1f}%)")
    print(f"    - Stationary: {old_stats['stationary_tags']} ({100*old_stats['stationary_tags']/max(1,old_stats['total_motion_tags']):.1f}%)")
    print(f"    - Single frame: {old_stats['single_frame_tags']} ({100*old_stats['single_frame_tags']/max(1,old_stats['total_motion_tags']):.1f}%)")
    print()
except:
    print("OLD dataset not found (skipping comparison)")
    print()

# New version
with open('/mnt/data/stgr/json_data/STGR-SFT-subset-motion-v2.json') as f:
    new_data = json.load(f)
new_stats = count_motion_tags(new_data)

print("NEW Dataset (multi-frame only):")
print(f"  Total samples: {new_stats['total_samples']}")
print(f"  Samples with motion tags: {new_stats['samples_with_motion']}")
print(f"  Total motion tags: {new_stats['total_motion_tags']}")
print(f"    - Actual motion: {new_stats['actual_motion_tags']} ({100*new_stats['actual_motion_tags']/max(1,new_stats['total_motion_tags']):.1f}%)")
print(f"    - Stationary: {new_stats['stationary_tags']} ({100*new_stats['stationary_tags']/max(1,new_stats['total_motion_tags']):.1f}%)")
print(f"    - Single frame: {new_stats['single_frame_tags']} ({100*new_stats['single_frame_tags']/max(1,new_stats['total_motion_tags']):.1f}%)")

if new_stats['single_frame_tags'] == 0:
    print()
    print("✅ SUCCESS: No more 'single frame' noise!")
EOF

else
    echo "❌ Augmentation failed"
    exit 1
fi
