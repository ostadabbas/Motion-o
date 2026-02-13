#!/usr/bin/env python3
"""
Download STGR dataset from HuggingFace.

Dataset: marinero4972/Open-o3-Video (~48.7 GB)
Includes: JSON annotations + video files

Recent change (2026-02): HuggingFace repo now packages some
videos as zip files (e.g., videos/stgr.zip, videos/videoespresso/kfs.zip).
This script now automatically extracts any .zip / .tar(.gz) archives
after download so you get the expected directory layout.
"""

import os
import json
import argparse
import zipfile
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download, login


def _is_already_downloaded(output_path: Path, json_only: bool) -> bool:
    """Return True if the dataset appears to be already present (skip download)."""
    json_dir = output_path / "json_data"
    if json_only:
        return (json_dir / "STGR-SFT.json").exists() and (json_dir / "STGR-RL.json").exists()
    return json_dir.exists() and (output_path / "videos").exists()


def download_dataset(output_dir: str, json_only: bool = False, token: str = None):
    """
    Download STGR dataset from HuggingFace.
    
    If the dataset is already present (json_data/ and videos/ for full,
    or both JSON files for --json-only), skips download and only runs
    extraction and verification.
    
    Args:
        output_dir: Directory to save dataset (e.g., /mnt/data/stgr)
        json_only: If True, only download JSON files (fast, ~few hundred MB)
        token: HuggingFace token (optional, for private repos)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STGR Dataset from HuggingFace")
    print("=" * 70)
    print(f"Repository: marinero4972/Open-o3-Video")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Mode: {'JSON only' if json_only else 'Full dataset (48.7 GB)'}")
    print("=" * 70)
    print()
    
    already = _is_already_downloaded(output_path, json_only)
    
    if already:
        print("✅ Data already present — skipping download.")
        print("   Extracting archives (if any) and verifying...")
        print()
    else:
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
    
    # Always run extraction (no-op if no archives) and verification
    extract_archives(output_path)
    verify_download(output_path)


def extract_archives(data_root: Path) -> None:
    """
    Extract any .zip / .tar / .tar.gz archives under data_root.

    This is mainly for cases where the HuggingFace dataset stores
    videos as zip files (e.g., videos/stgr.zip).
    """
    print("=" * 70)
    print("Checking for archive files to extract...")
    print("=" * 70)
    print()

    archives: list[tuple[str, Path]] = []

    # Collect archives
    for p in data_root.rglob("*.zip"):
        archives.append(("zip", p))
    for p in data_root.rglob("*.tar.gz"):
        archives.append(("tar.gz", p))
    for p in data_root.rglob("*.tar"):
        # Avoid double-counting .tar.gz which we already added
        if not str(p).endswith(".tar.gz"):
            archives.append(("tar", p))

    if not archives:
        print("✅ No archive files found (dataset already extracted).")
        print()
        return

    print(f"Found {len(archives)} archive file(s):")
    for kind, p in archives:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  - {p.relative_to(data_root)} ({kind}, {size_mb:.1f} MB)")
    print()

    # Extract in-place next to each archive
    for kind, p in archives:
        print(f"Extracting {p.relative_to(data_root)} ...")
        try:
            target_dir = p.parent
            if kind == "zip":
                with zipfile.ZipFile(p, "r") as zf:
                    zf.extractall(target_dir)
            elif kind == "tar.gz":
                with tarfile.open(p, "r:gz") as tf:
                    tf.extractall(target_dir)
            elif kind == "tar":
                with tarfile.open(p, "r") as tf:
                    tf.extractall(target_dir)
            print("  ✅ Done")
            # If you want to save space, uncomment the next two lines:
            # p.unlink()
            # print("  🗑️  Removed archive file")
        except Exception as e:
            print(f"  ❌ Failed to extract {p}: {e}")
            print("  Continuing...")

    print()
    print("✅ Archive extraction pass finished.")
    print()


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
