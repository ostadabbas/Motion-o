#!/usr/bin/env python3
"""
Download STGR dataset from HuggingFace.

Dataset: marinero4972/Open-o3-Video (~48.7 GB)
Includes: JSON annotations + video files
"""

import os
import json
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, login


def download_dataset(output_dir: str, json_only: bool = False, token: str = None):
    """
    Download STGR dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save dataset (e.g., /mnt/data/stgr)
        json_only: If True, only download JSON files (fast, ~few hundred MB)
        token: HuggingFace token (optional, for private repos)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Downloading STGR Dataset from HuggingFace")
    print("=" * 70)
    print(f"Repository: marinero4972/Open-o3-Video")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Mode: {'JSON only' if json_only else 'Full dataset (48.7 GB)'}")
    print("=" * 70)
    print()
    
    # Login if token provided
    if token:
        print("Logging in to HuggingFace...")
        login(token=token)
        print("✅ Logged in successfully")
        print()
    
    try:
        if json_only:
            print("Downloading JSON annotations only...")
            print("(This is fast - just a few hundred MB)")
            print()
            snapshot_download(
                repo_id="marinero4972/Open-o3-Video",
                repo_type="dataset",
                local_dir=str(output_path),
                allow_patterns=["json_data/*"],
                resume_download=True,
            )
        else:
            print("Downloading full dataset (48.7 GB)...")
            print("(This may take 30 mins - 2 hours depending on your connection)")
            print()
            snapshot_download(
                repo_id="marinero4972/Open-o3-Video",
                repo_type="dataset",
                local_dir=str(output_path),
                resume_download=True,
            )
        
        print()
        print("=" * 70)
        print("✅ Download Complete!")
        print("=" * 70)
        print()
        
        # Verify what was downloaded
        verify_download(output_path)
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ Download Failed!")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        print("Possible solutions:")
        print("1. Check your internet connection")
        print("2. Login to HuggingFace: huggingface-cli login")
        print("3. Verify repo access: https://huggingface.co/datasets/marinero4972/Open-o3-Video")
        print("4. Try again with --json-only first to test")
        print()
        raise


def verify_download(data_root: Path):
    """Verify what was downloaded and show statistics."""
    print("Verifying downloaded files...")
    print()
    
    # Check JSON files
    json_dir = data_root / "json_data"
    if json_dir.exists():
        print("📄 JSON Annotations:")
        for json_file in ["STGR-SFT.json", "STGR-RL.json"]:
            json_path = json_dir / json_file
            if json_path.exists():
                size = json_path.stat().st_size / (1024 * 1024)  # MB
                with open(json_path) as f:
                    data = json.load(f)
                    print(f"  ✅ {json_file}: {len(data)} samples ({size:.1f} MB)")
            else:
                print(f"  ❌ {json_file}: MISSING")
    else:
        print("  ❌ json_data/ directory not found")
    
    print()
    
    # Check video directories
    videos_dir = data_root / "videos"
    if videos_dir.exists():
        print("🎥 Video Directories:")
        video_subdirs = [
            "gqa", "stgr/plm", "stgr/temporal_grounding",
            "timerft", "treevgr", "tvg_r1", "videoespresso", "videor1"
        ]
        
        total_videos = 0
        for subdir in video_subdirs:
            subdir_path = videos_dir / subdir
            if subdir_path.exists():
                video_count = len(list(subdir_path.rglob("*.mp4")))
                total_videos += video_count
                if video_count > 0:
                    print(f"  ✅ {subdir}: {video_count} videos")
                else:
                    print(f"  ⚠️  {subdir}: exists but no .mp4 files found")
            else:
                print(f"  ❌ {subdir}: NOT FOUND")
        
        print()
        print(f"  Total videos: {total_videos}")
    else:
        print("  ⚠️  videos/ directory not found (did you use --json-only?)")
    
    print()
    print("Next steps:")
    print(f"1. Update configs/data_root.py to point to: {data_root.absolute()}")
    print(f"2. Verify data loading: python -c \"import json; print(json.load(open('{json_dir}/STGR-RL.json'))[:1])\"")
    print("3. Start training: bash scripts/run_sft.sh")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download STGR dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download JSON only (fast, for inspection)
  python scripts/download_dataset.py --output-dir /mnt/data/stgr --json-only
  
  # Download full dataset (48.7 GB)
  python scripts/download_dataset.py --output-dir /mnt/data/stgr
  
  # With HuggingFace token
  python scripts/download_dataset.py --output-dir /mnt/data/stgr --token hf_...
        """
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for STGR dataset (e.g., /mnt/data/stgr)"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only download JSON annotations (fast, ~few hundred MB)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, for private repos)"
    )
    
    args = parser.parse_args()
    
    download_dataset(args.output_dir, args.json_only, args.token)


if __name__ == "__main__":
    main()
