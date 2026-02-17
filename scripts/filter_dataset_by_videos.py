#!/usr/bin/env python3
"""
Filter STGR dataset to only include samples with available videos/images.

Uses same Open-o3-Video layout and path resolution as filter_stgr_available.py.
  DATA_ROOT = /scratch/bai.xiang/Open-o3-Video
  video_path / image_path relative to DATA_ROOT/videos/
"""

import json
import argparse
import os
import sys
from pathlib import Path

# Allow importing filter_stgr_available when run as script from repo root
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from filter_stgr_available import (
    DEFAULT_DATA_ROOT,
    find_video_file,
    find_image_file,
    filter_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Filter STGR JSON to samples with available videos/images (Open-o3-Video layout)"
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to STGR-SFT.json or STGR-RL.json",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Open-o3-Video data root",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for filtered JSON",
    )
    args = parser.parse_args()
    filter_dataset(args.json, args.data_root, args.output)
    print("Next steps:")
    print(f"  1. Use this filtered JSON in training: --dataset_name {args.output}")
    print(f"  2. Training expects video_root = {args.data_root}/videos/")


if __name__ == "__main__":
    main()
