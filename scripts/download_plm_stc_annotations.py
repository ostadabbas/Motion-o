#!/usr/bin/env python3
"""
Download PLM-STC RDCap annotations from HuggingFace.

Downloads the RDCap subset (dense captions) which has 117K samples
and is the best format match for multi-step evidence chains.
"""

from datasets import load_dataset
from pathlib import Path
import json
import sys

def main():
    output_dir = Path("/mnt/data/plm_stc/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Downloading PLM-STC RDCap annotations from HuggingFace")
    print("="*70)
    print()
    
    try:
        # Download RDCap (dense captions)
        print("Loading dataset 'facebook/PLM-Video-Human', subset 'rdcap'...")
        rdcap = load_dataset('facebook/PLM-Video-Human', 'rdcap', split='train')
        print(f"✓ Downloaded successfully!")
        print()
        
        # Analyze dataset
        print("Dataset Statistics:")
        print(f"  Total samples: {len(rdcap)}")
        
        # Extract unique video IDs
        video_ids = set(rdcap['video'])
        print(f"  Unique videos: {len(video_ids)}")
        print()
        
        # Show sample
        print("Sample annotation:")
        sample = rdcap[0]
        print(f"  Video: {sample['video']}")
        print(f"  Masklet ID: {sample['masklet_id']}")
        print(f"  Total frames: {sample['total_frames']}")
        if 'dense_captions' in sample:
            print(f"  Dense captions: {len(sample['dense_captions'])} entries")
            print(f"  First caption: {sample['dense_captions'][0] if sample['dense_captions'] else 'N/A'}")
        print()
        
        # Save annotations
        print(f"Saving annotations to {output_dir / 'rdcap'}...")
        rdcap.save_to_disk(str(output_dir / 'rdcap'))
        print("✓ Saved!")
        print()
        
        # Save video ID list for selective download
        video_list_path = output_dir / 'video_ids_needed.txt'
        print(f"Saving video ID list to {video_list_path}...")
        with open(video_list_path, 'w') as f:
            for vid in sorted(video_ids):
                f.write(f"{vid}\n")
        print(f"✓ Saved {len(video_ids)} video IDs!")
        print()
        
        # Save first 100 for testing
        test_video_list_path = output_dir / 'video_ids_test_100.txt'
        print(f"Saving first 100 video IDs to {test_video_list_path}...")
        with open(test_video_list_path, 'w') as f:
            for vid in sorted(video_ids)[:100]:
                f.write(f"{vid}\n")
        print("✓ Saved!")
        print()
        
        print("="*70)
        print("SUCCESS!")
        print("="*70)
        print()
        print("Next steps:")
        print(f"  1. Download SA-V videos listed in: {video_list_path}")
        print(f"     For quick testing, use first 100: {test_video_list_path}")
        print(f"  2. Place videos in: /mnt/data/plm_stc/raw/sa-v/")
        print(f"  3. Run conversion script to format data")
        print()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
