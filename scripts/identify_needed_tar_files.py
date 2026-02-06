#!/usr/bin/env python3
"""
Identify which SA-V tar files are needed for PLM-STC videos.

Helps you download only the tar files you actually need, instead of all 442GB.
"""

import argparse
from pathlib import Path
from collections import defaultdict


def video_id_to_tar_number(video_id: str) -> int:
    """
    Determine which tar file contains a given video.
    
    SA-V videos are numbered and distributed:
    - sav_000.tar: videos 0-999
    - sav_001.tar: videos 1000-1999
    - ...
    
    Args:
        video_id: Video filename (e.g., 'sav_017599.mp4' or 'sav_017599')
    
    Returns:
        Tar file number (e.g., 17 for sav_017.tar)
    """
    # Extract numeric part
    video_id = video_id.replace('.mp4', '').replace('sav_', '')
    
    try:
        video_num = int(video_id)
        tar_num = video_num // 1000  # Each tar has 1000 videos
        return tar_num
    except ValueError:
        return -1


def main():
    parser = argparse.ArgumentParser(
        description="Identify SA-V tar files needed for PLM-STC dataset"
    )
    parser.add_argument(
        "--video-list",
        type=str,
        default="/mnt/data/plm_stc/raw/video_ids_test_100.txt",
        help="Path to video IDs list"
    )
    parser.add_argument(
        "--show-videos",
        action="store_true",
        help="Show which videos are in each tar file"
    )
    
    args = parser.parse_args()
    
    video_list_path = Path(args.video_list)
    
    if not video_list_path.exists():
        print(f"ERROR: Video list not found: {video_list_path}")
        print()
        print("Run this first:")
        print("  python scripts/download_plm_stc_annotations.py")
        return
    
    # Read video IDs
    with open(video_list_path, 'r') as f:
        video_ids = [line.strip() for line in f if line.strip()]
    
    print("="*70)
    print("SA-V TAR FILE REQUIREMENTS ANALYSIS")
    print("="*70)
    print(f"Video list: {video_list_path}")
    print(f"Total videos: {len(video_ids)}")
    print()
    
    # Group videos by tar file
    tar_to_videos = defaultdict(list)
    
    for video_id in video_ids:
        tar_num = video_id_to_tar_number(video_id)
        if tar_num >= 0:
            tar_to_videos[tar_num].append(video_id)
    
    # Sort by tar number
    sorted_tars = sorted(tar_to_videos.keys())
    
    print(f"Required tar files: {len(sorted_tars)}")
    print()
    
    # Calculate total download size (each tar is ~8GB)
    total_size_gb = len(sorted_tars) * 8
    
    print("Download Summary:")
    print(f"  Tar files needed: {len(sorted_tars)}")
    print(f"  Estimated size: ~{total_size_gb}GB")
    print()
    
    print("Required tar files:")
    for tar_num in sorted_tars:
        videos = tar_to_videos[tar_num]
        tar_name = f"sav_{tar_num:03d}.tar"
        print(f"  {tar_name:20s} ({len(videos):4d} videos needed)")
        
        if args.show_videos:
            for vid in videos[:5]:  # Show first 5
                print(f"      - {vid}")
            if len(videos) > 5:
                print(f"      ... and {len(videos) - 5} more")
    
    print()
    print("="*70)
    print("DOWNLOAD COMMANDS")
    print("="*70)
    print()
    
    # Generate download commands
    print("# Create download directory")
    print("mkdir -p /mnt/data/plm_stc/raw/")
    print("cd /mnt/data/plm_stc/raw/")
    print()
    
    print("# Download required tar files")
    base_url = "https://dl.fbaipublicfiles.com/segment_anything_video/sav_train/"
    
    for tar_num in sorted_tars:
        tar_name = f"sav_{tar_num:03d}.tar"
        print(f"wget {base_url}{tar_name}")
    
    print()
    print("# Or download all at once with aria2c:")
    print("cat > sa_v_download_list.txt << EOF")
    for tar_num in sorted_tars:
        tar_name = f"sav_{tar_num:03d}.tar"
        print(f"{base_url}{tar_name}")
    print("EOF")
    print()
    print("aria2c -i sa_v_download_list.txt -x 4 -j 3")
    print()
    
    print("="*70)
    print("EXTRACTION")
    print("="*70)
    print()
    
    print("# Extract all downloaded tar files")
    print("mkdir -p sa-v")
    for tar_num in sorted_tars:
        tar_name = f"sav_{tar_num:03d}.tar"
        print(f"tar -xvf {tar_name} -C sa-v/")
    
    print()
    print("# Verify extraction")
    print("ls sa-v/*.mp4 | wc -l")
    print()
    
    print("="*70)
    print("RECOMMENDATION")
    print("="*70)
    print()
    
    if total_size_gb > 50:
        print(f"⚠️  Total size is large ({total_size_gb}GB)")
        print()
        print("For quick testing, consider:")
        print("  1. Download only the first tar file (sav_000.tar)")
        print("  2. Filter RDCap to use only videos from that range")
        print()
        print("Command:")
        print("  python scripts/filter_rdcap_by_video_range.py \\")
        print("      --min-video 0 --max-video 999 --limit 100")
    else:
        print(f"✓ Manageable download size: {total_size_gb}GB")
        print(f"  This is {len(sorted_tars)} tar files")
        print()
        print("Proceed with download commands above.")
    
    print()


if __name__ == "__main__":
    main()
