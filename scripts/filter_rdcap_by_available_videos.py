#!/usr/bin/env python3
"""
Filter RDCap dataset to only include samples for videos we actually downloaded.
"""

import argparse
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Filter RDCap to available videos"
    )
    parser.add_argument(
        "--input-rdcap",
        type=str,
        default="/mnt/data/plm_stc/raw/rdcap",
        help="Path to RDCap dataset"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="/mnt/data/plm_stc/raw/sa-v",
        help="Directory containing SA-V videos"
    )
    parser.add_argument(
        "--output-rdcap",
        type=str,
        default="/mnt/data/plm_stc/raw/rdcap_filtered",
        help="Output path for filtered dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit filtered samples"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Filtering RDCap to Available Videos")
    print("="*70)
    print()
    
    # Load RDCap
    print(f"Loading RDCap from {args.input_rdcap}...")
    rdcap = load_from_disk(args.input_rdcap)
    print(f"✓ Loaded {len(rdcap)} samples")
    print()
    
    # Get available videos
    video_dir = Path(args.video_dir)
    available_videos = set()
    for video_file in video_dir.glob("*.mp4"):
        available_videos.add(video_file.name)
    
    print(f"Available videos: {len(available_videos)}")
    print(f"  Sample videos: {list(sorted(available_videos))[:5]}")
    print()
    
    # Filter samples
    print("Filtering samples...")
    filtered_indices = []
    
    for idx in tqdm(range(len(rdcap)), desc="Scanning"):
        video_name = rdcap[idx]['video']
        if video_name in available_videos:
            filtered_indices.append(idx)
            if args.limit and len(filtered_indices) >= args.limit:
                break
    
    print()
    print(f"Matching samples: {len(filtered_indices)}")
    print()
    
    if len(filtered_indices) == 0:
        print("ERROR: No matching samples found!")
        print()
        print("RDCap references:")
        print(f"  {rdcap[0]['video']}")
        print(f"  {rdcap[1]['video'] if len(rdcap) > 1 else 'N/A'}")
        print()
        print("Available videos:")
        print(f"  {list(sorted(available_videos))[:5]}")
        return
    
    # Create filtered dataset
    print(f"Creating filtered dataset with {len(filtered_indices)} samples...")
    filtered_rdcap = rdcap.select(filtered_indices)
    
    # Save
    output_path = Path(args.output_rdcap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path}...")
    filtered_rdcap.save_to_disk(str(output_path))
    print("✓ Saved!")
    print()
    
    # Show samples
    print("Sample filtered entries:")
    for i in range(min(3, len(filtered_rdcap))):
        sample = filtered_rdcap[i]
        print(f"  {i+1}. {sample['video']} (masklet {sample['masklet_id']}, {len(sample['dense_captions'])} captions)")
    print()
    
    print("="*70)
    print("SUCCESS!")
    print("="*70)
    print()
    print(f"Filtered dataset: {output_path}")
    print(f"Samples: {len(filtered_rdcap)}")
    print()
    print("Next step:")
    print(f"  python scripts/convert_plm_stc_to_format.py \\")
    print(f"      --input-annotations {output_path} \\")
    print(f"      --input-videos {args.video_dir} \\")
    print(f"      --output-dir /mnt/data/plm_stc/formatted_test \\")
    print(f"      --limit 100")
    print()


if __name__ == "__main__":
    main()
