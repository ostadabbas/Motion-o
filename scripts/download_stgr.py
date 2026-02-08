#!/usr/bin/env python3
"""
Download STGR dataset (annotations and videos).

STGR Dataset components:
1. JSON annotations (STGR-SFT.json and STGR-RL.json) - from HuggingFace
2. Video files from multiple sources (GQA, TimeRFT, TVG, PLM, etc.)

Dataset paper: https://huggingface.co/datasets/facebook/PLM-Video-Human
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def download_json_annotations(output_dir: str):
    """
    Download STGR JSON annotations from HuggingFace.
    
    Args:
        output_dir: Directory to save JSON files
    """
    from datasets import load_dataset
    
    output_path = Path(output_dir) / "json_data"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Downloading STGR Annotations from HuggingFace")
    print("=" * 70)
    
    # Note: The actual STGR dataset is not directly available on HuggingFace
    # Open-o3 Video provides it separately
    # This is a placeholder - you need to get the actual download link from:
    # https://github.com/marinero4972/Open-o3-Video
    
    print("\n⚠️  IMPORTANT:")
    print("The STGR dataset is not directly available on HuggingFace.")
    print("You need to:")
    print("1. Visit: https://github.com/marinero4972/Open-o3-Video")
    print("2. Follow their data preparation instructions")
    print("3. Download STGR-SFT.json and STGR-RL.json")
    print("4. Place them in:", output_path.absolute())
    print()
    print("Expected files:")
    print(f"  - {output_path}/STGR-SFT.json  (~30k samples for SFT)")
    print(f"  - {output_path}/STGR-RL.json   (~36k samples for RL)")
    print()


def check_dataset_structure(data_root: str):
    """
    Check if STGR dataset is properly downloaded and structured.
    
    Args:
        data_root: Root directory of STGR dataset
    """
    data_root = Path(data_root)
    
    print("=" * 70)
    print("Checking STGR Dataset Structure")
    print("=" * 70)
    print()
    
    required_files = {
        "json_data/STGR-SFT.json": "SFT annotations (~30k samples)",
        "json_data/STGR-RL.json": "RL annotations (~36k samples)",
    }
    
    required_video_dirs = {
        "videos/gqa": "GQA videos",
        "videos/stgr/plm": "PLM videos",
        "videos/stgr/temporal_grounding": "Temporal grounding videos",
        "videos/timerft": "TimeRFT videos",
        "videos/treevgr": "TreeVGR videos",
        "videos/tvg_r1": "TVG-R1 videos",
        "videos/videoespresso": "VideoEspresso videos",
        "videos/videor1": "VideoR1 videos",
    }
    
    # Check JSON files
    print("JSON Annotations:")
    for file_path, description in required_files.items():
        full_path = data_root / file_path
        if full_path.exists():
            size = full_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✅ {file_path} ({size:.1f} MB) - {description}")
        else:
            print(f"  ❌ {file_path} - MISSING - {description}")
    print()
    
    # Check video directories
    print("Video Directories:")
    total_videos = 0
    for dir_path, description in required_video_dirs.items():
        full_path = data_root / dir_path
        if full_path.exists():
            video_count = len(list(full_path.glob("*.mp4")))
            total_videos += video_count
            print(f"  ✅ {dir_path} ({video_count} videos) - {description}")
        else:
            print(f"  ❌ {dir_path} - MISSING - {description}")
    
    print()
    print(f"Total videos found: {total_videos}")
    print()
    
    if total_videos == 0:
        print("⚠️  WARNING: No video files found!")
        print("You need to download the video files separately.")
        print("See: https://github.com/marinero4972/Open-o3-Video for instructions")
    
    return total_videos > 0


def main():
    parser = argparse.ArgumentParser(description="Download STGR dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/data/stgr",
        help="Output directory for STGR dataset"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if dataset is properly structured (no download)"
    )
    
    args = parser.parse_args()
    
    if args.check_only:
        check_dataset_structure(args.output_dir)
    else:
        print("=" * 70)
        print("STGR Dataset Download")
        print("=" * 70)
        print()
        print(f"Output directory: {args.output_dir}")
        print()
        
        download_json_annotations(args.output_dir)
        
        print()
        print("=" * 70)
        print("Next Steps:")
        print("=" * 70)
        print()
        print("1. Follow instructions above to download JSON annotations")
        print("2. Download video files from Open-o3 Video data sources")
        print("3. Run: python scripts/download_stgr.py --check-only --output-dir", args.output_dir)
        print("4. Update configs/data_root.py with your data path")
        print()


if __name__ == "__main__":
    main()
