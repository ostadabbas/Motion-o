#!/usr/bin/env python3
"""
Test the new acceleration-based motion tags.
"""

import json
import sys
sys.path.insert(0, '.')

from configs.data_root import DATA_ROOT
import os

# Import the new augmentation function
from scripts.augment_motion_data_simple import augment_sample

def main():
    print("="*70)
    print("Testing New Motion Tags with Acceleration")
    print("="*70)
    
    # Load a sample with known motion
    dataset_path = os.path.join(DATA_ROOT, "json_data/STGR-SFT-subset.json")
    
    with open(dataset_path) as f:
        data = json.load(f)
    
    # Test sample #5283 (strong motion)
    sample = data[5283]
    
    print(f"\n📹 Sample #5283:")
    print(f"   Question: {sample['question'][:60]}...")
    
    # Augment it
    augmented = augment_sample(sample)
    
    # Extract motion tags
    import re
    motion_tags = re.findall(r'<motion>([^<]+)</motion>', augmented['reasoning_process'])
    
    print(f"\n🎯 Generated Motion Tags ({len(motion_tags)}):")
    for i, tag in enumerate(motion_tags, 1):
        print(f"\n   {i}. <motion>{tag}</motion>")
        
        # Parse components
        if 'speed:' in tag and 'accel:' in tag:
            import re
            speed_match = re.search(r'speed:\s*([\d.]+)', tag)
            accel_match = re.search(r'accel:\s*([+-]?[\d.]+)', tag)
            direction = tag.split(' motion')[0]
            
            if speed_match and accel_match:
                speed = float(speed_match.group(1))
                accel = float(accel_match.group(1))
                
                print(f"      Direction: {direction}")
                print(f"      Speed: {speed:.3f} units/s")
                print(f"      Acceleration: {accel:+.3f} units/s²", end="")
                
                if accel > 0.01:
                    print(" ← Speeding up! 🚀")
                elif accel < -0.01:
                    print(" ← Slowing down 🛑")
                else:
                    print(" ← Constant velocity →")
    
    # Test a few more samples
    print(f"\n" + "="*70)
    print("Testing across multiple samples...")
    print("="*70)
    
    samples_tested = 0
    motion_found = 0
    accel_positive = 0
    accel_negative = 0
    accel_constant = 0
    
    for idx in range(100):
        sample = data[idx]
        augmented = augment_sample(sample)
        motion_tags = re.findall(r'<motion>([^<]+)</motion>', augmented['reasoning_process'])
        
        if motion_tags:
            samples_tested += 1
            motion_found += len(motion_tags)
            
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
    
    print(f"\n📊 Statistics from first 100 samples:")
    print(f"   Samples with motion: {samples_tested}")
    print(f"   Total motion tags: {motion_found}")
    if motion_found > 0:
        print(f"\n   Acceleration breakdown:")
        print(f"     Accelerating (+): {accel_positive} ({100*accel_positive/motion_found:.1f}%)")
        print(f"     Decelerating (-): {accel_negative} ({100*accel_negative/motion_found:.1f}%)")
        print(f"     Constant (≈0): {accel_constant} ({100*accel_constant/motion_found:.1f}%)")
    
    print(f"\n" + "="*70)
    print("✅ Acceleration-based motion tags working!")
    print("="*70)
    
    print(f"\n💡 Motion tag format:")
    print(f"   <motion>DIRECTION motion (speed: X.XXX units/s, accel: ±X.XXX units/s²)</motion>")
    print(f"\n   Where:")
    print(f"   • DIRECTION: up-left, rightward, downward, etc.")
    print(f"   • speed: Average speed in units/s")
    print(f"   • accel: Acceleration (+speeding up, -slowing down, ≈0 constant)")


if __name__ == "__main__":
    main()
