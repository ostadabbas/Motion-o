#!/usr/bin/env python3
"""
Convert old label files to unified format (multiple videos in one file).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

def convert_labels_to_unified(label_files: List[str], output_path: str):
    """Convert multiple label files to unified format."""
    unified_data = {"videos": []}
    
    print(f"Converting {len(label_files)} label files to unified format...")
    
    for label_file in label_files:
        label_path = Path(label_file)
        if not label_path.exists():
            print(f"⚠ Warning: {label_file} not found, skipping")
            continue
        
        print(f"  Processing {label_path.name}...")
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle old format
        if "video_path" in data and "segments" in data:
            unified_data["videos"].append({
                "video_path": data["video_path"],
                "segments": data["segments"]
            })
            print(f"    ✓ Added {len(data['segments'])} segments")
        # Already unified format
        elif "videos" in data:
            unified_data["videos"].extend(data["videos"])
            total = sum(len(v.get("segments", [])) for v in data["videos"])
            print(f"    ✓ Added {total} segments from {len(data['videos'])} videos")
        else:
            print(f"    ⚠ Warning: Invalid format, skipping")
    
    # Save unified format
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, indent=2, ensure_ascii=False)
    
    total_videos = len(unified_data["videos"])
    total_segments = sum(len(v.get("segments", [])) for v in unified_data["videos"])
    print(f"\n✓ Saved unified labels: {total_segments} segments across {total_videos} videos")
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert label files to unified format"
    )
    parser.add_argument(
        "label_files",
        nargs="+",
        help="Label files to convert"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="labels/video_labels_unified.json",
        help="Output path for unified labels"
    )
    
    args = parser.parse_args()
    convert_labels_to_unified(args.label_files, args.output)


if __name__ == "__main__":
    main()

