#!/usr/bin/env python3
"""
Filter STGR dataset to only include samples with available videos.

This creates a subset JSON with only samples whose video files exist on disk.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def filter_by_available_videos(json_path: str, data_root: str, output_path: str):
    """
    Filter dataset to only samples with existing video files.
    
    Args:
        json_path: Path to STGR-SFT.json or STGR-RL.json
        data_root: Root directory of STGR dataset
        output_path: Path to save filtered JSON
    """
    print("=" * 70)
    print("Filtering Dataset by Available Videos")
    print("=" * 70)
    print(f"Input: {json_path}")
    print(f"Data root: {data_root}")
    print(f"Output: {output_path}")
    print()
    
    # Load JSON
    with open(json_path) as f:
        data = json.load(f)
    
    print(f"Original samples: {len(data)}")
    
    # Filter by video existence
    filtered_data = []
    missing_sources = {}
    
    for item in tqdm(data, desc="Checking videos"):
        video_path = item.get('video_path', '')
        
        if not video_path:
            continue
        
        # Try different path constructions
        possible_paths = [
            os.path.join(data_root, video_path),
            os.path.join(data_root, 'videos', video_path),
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                found = True
                break
        
        if found:
            filtered_data.append(item)
        else:
            source = item.get('source', 'unknown')
            missing_sources[source] = missing_sources.get(source, 0) + 1
    
    print()
    print(f"Filtered samples: {len(filtered_data)} ({len(filtered_data)/len(data)*100:.1f}%)")
    print(f"Missing samples: {len(data) - len(filtered_data)}")
    print()
    
    if missing_sources:
        print("Missing videos by source:")
        for source, count in sorted(missing_sources.items(), key=lambda x: -x[1]):
            print(f"  - {source}: {count} samples")
        print()
    
    # Save filtered dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"✅ Saved filtered dataset to: {output_path}")
    print()
    print("Next steps:")
    print(f"1. Use this filtered JSON in training: {output_path}")
    print(f"2. Update training script to use: --dataset_name {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Filter STGR by available videos")
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to STGR-SFT.json or STGR-RL.json"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data/stgr",
        help="Root directory of STGR dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for filtered JSON"
    )
    
    args = parser.parse_args()
    filter_by_available_videos(args.json, args.data_root, args.output)


if __name__ == "__main__":
    main()
