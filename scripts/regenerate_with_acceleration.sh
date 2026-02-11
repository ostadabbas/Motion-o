#!/bin/bash
# Regenerate motion dataset with acceleration instead of smooth/jerky/erratic

echo "=========================================="
echo "Regenerating Dataset with Acceleration"
echo "=========================================="
echo ""

INPUT="/mnt/data/stgr/json_data/STGR-SFT-subset.json"
OUTPUT="/mnt/data/stgr/json_data/STGR-SFT-subset-motion-v3.json"

echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo ""
echo "Changes from v2:"
echo "  • Added: acceleration (±X.XXX units/s²)"
echo "  • Removed: smooth/jerky/erratic descriptors"
echo "  • Better: Numerical physics-based metrics"
echo ""

# Backup v2
if [ -f "/mnt/data/stgr/json_data/STGR-SFT-subset-motion-v2.json" ]; then
    echo "Keeping v2 as backup..."
fi

# Run augmentation with new acceleration-based script
echo "Running augmentation..."
python scripts/augment_motion_data_simple.py \
    --input "$INPUT" \
    --output "$OUTPUT"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Dataset regenerated with acceleration!"
    echo "=========================================="
    echo ""
    
    # Show examples
    python3 << 'EOF'
import json
import re

with open('/mnt/data/stgr/json_data/STGR-SFT-subset-motion-v3.json') as f:
    data = json.load(f)

print("📊 Dataset Statistics:")
print(f"   Total samples: {len(data)}")

# Count motion tags
samples_with_motion = 0
total_motion_tags = 0
accel_positive = 0
accel_negative = 0
accel_constant = 0

for sample in data:
    motion_tags = re.findall(r'<motion>([^<]+)</motion>', sample.get('reasoning_process', ''))
    if motion_tags:
        samples_with_motion += 1
        total_motion_tags += len(motion_tags)
        
        for tag in motion_tags:
            accel_match = re.search(r'accel:\s*([+-]?[\d.]+)', tag)
            if accel_match:
                accel = float(accel_match.group(1))
                if accel > 0.01:
                    accel_positive += 1
                elif accel < -0.01:
                    accel_negative += 1
                else:
                    accel_constant += 1

print(f"   Samples with motion: {samples_with_motion}")
print(f"   Total motion tags: {total_motion_tags}")
print(f"\n   Acceleration distribution:")
print(f"     Accelerating (+): {accel_positive} ({100*accel_positive/max(1,total_motion_tags):.1f}%)")
print(f"     Decelerating (-): {accel_negative} ({100*accel_negative/max(1,total_motion_tags):.1f}%)")
print(f"     Constant (≈0): {accel_constant} ({100*accel_constant/max(1,total_motion_tags):.1f}%)")

print(f"\n📝 Example motion tags:")

# Find sample with acceleration
for sample in data:
    motion_tags = re.findall(r'<motion>([^<]+)</motion>', sample.get('reasoning_process', ''))
    for tag in motion_tags:
        if 'accel:' in tag and '+' in tag:
            accel_match = re.search(r'accel:\s*\+(\d+\.\d+)', tag)
            if accel_match and float(accel_match.group(1)) > 0.1:
                print(f"\n   Accelerating:")
                print(f"   <motion>{tag}</motion>")
                break
    else:
        continue
    break

# Find constant velocity
for sample in data[:100]:
    motion_tags = re.findall(r'<motion>([^<]+)</motion>', sample.get('reasoning_process', ''))
    for tag in motion_tags:
        if 'accel:' in tag:
            accel_match = re.search(r'accel:\s*([+-]?\d+\.\d+)', tag)
            if accel_match and abs(float(accel_match.group(1))) < 0.01:
                print(f"\n   Constant velocity:")
                print(f"   <motion>{tag}</motion>")
                break
    else:
        continue
    break

print(f"\n{'='*70}")
print("✅ Motion tags now include:")
print("   • Direction (textual)")
print("   • Speed (numerical, units/s)")
print("   • Acceleration (numerical, ±units/s²)")
print("{'='*70}")

EOF

else
    echo "❌ Augmentation failed"
    exit 1
fi
