#!/usr/bin/env python3
"""
Filter STGR dataset to samples with available videos.
Handles the actual extracted directory structure from stgr.zip.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm


def find_video_file(video_path_from_json, data_root):
    """
    Try multiple path patterns to locate the actual video file.
    
    Args:
        video_path_from_json: The video_path from JSON (e.g., "sav_054365.mp4")
        data_root: Root directory (/mnt/data/stgr)
    
    Returns:
        Full path if found, None otherwise
    """
    videos_dir = Path(data_root) / "videos"
    
    # Pattern 1: Direct sav_*.mp4 files (PLM)
    if video_path_from_json.startswith('sav_') and video_path_from_json.endswith('.mp4'):
        possible_paths = [
            videos_dir / "stgr" / "plm" / "videos" / video_path_from_json,
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
    
    # Pattern 2: Coin/didemo/activitynet videos in temporal_grounding
    if '/' in video_path_from_json:
        parts = Path(video_path_from_json).parts
        # Try: videos/stgr/temporal_grounding/videos/{source}/videos/{filename}
        if len(parts) >= 2:
            source = parts[0]  # e.g., "coin", "activitynet"
            filename = parts[-1]
            
            possible_paths = [
                videos_dir / "stgr" / "temporal_grounding" / "videos" / source / "videos" / filename,
                videos_dir / "stgr" / "temporal_grounding" / source / "videos" / filename,
                videos_dir / "stgr" / "temporal_grounding" / "videos" / filename,
            ]
            for path in possible_paths:
                if path.exists():
                    return str(path)
    
    # Pattern 3: Direct path under videos/
    direct_path = videos_dir / video_path_from_json
    if direct_path.exists():
        return str(direct_path)
    
    return None


def filter_dataset(json_path, data_root, output_path):
    """Filter dataset to only samples with existing videos."""
    
    print("=" * 70)
    print("Filtering STGR Dataset by Available Videos")
    print("=" * 70)
    print(f"Input: {json_path}")
    print(f"Data root: {data_root}")
    print(f"Output: {output_path}\n")
    
    # Load JSON
    with open(json_path) as f:
        data = json.load(f)
    
    print(f"Original samples: {len(data):,}")
    
    # Filter by video existence
    filtered_data = []
    found_by_source = {}
    missing_by_source = {}
    
    for item in tqdm(data, desc="Checking videos"):
        video_path = item.get('video_path', '')
        source = item.get('source', 'unknown')
        
        if not video_path:
            missing_by_source[source] = missing_by_source.get(source, 0) + 1
            continue
        
        actual_path = find_video_file(video_path, data_root)
        
        if actual_path:
            # Update the video_path to the actual full path for easier loading
            item['video_path_full'] = actual_path
            filtered_data.append(item)
            found_by_source[source] = found_by_source.get(source, 0) + 1
        else:
            missing_by_source[source] = missing_by_source.get(source, 0) + 1
    
    print()
    print(f"✅ Filtered samples: {len(filtered_data):,} ({len(filtered_data)/len(data)*100:.1f}%)")
    print(f"❌ Missing samples: {len(data) - len(filtered_data):,}\n")
    
    if found_by_source:
        print("Found videos by source:")
        for source, count in sorted(found_by_source.items(), key=lambda x: -x[1]):
            print(f"  ✅ {source}: {count:,} samples")
        print()
    
    if missing_by_source:
        print("Missing videos by source:")
        for source, count in sorted(missing_by_source.items(), key=lambda x: -x[1])[:10]:
            print(f"  ❌ {source}: {count:,} samples")
        if len(missing_by_source) > 10:
            print(f"  ... and {len(missing_by_source) - 10} more sources")
        print()
    
    # Save filtered dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"✅ Saved filtered dataset: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Input JSON (STGR-SFT.json or STGR-RL.json)")
    parser.add_argument("--data-root", default="/mnt/data/stgr", help="STGR data root")
    parser.add_argument("--output", required=True, help="Output filtered JSON path")
    args = parser.parse_args()
    
    filter_dataset(args.json, args.data_root, args.output)
