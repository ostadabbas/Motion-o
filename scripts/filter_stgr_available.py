#!/usr/bin/env python3
"""
Filter STGR dataset to samples with available videos/images.

Uses Open-o3-Video directory layout:
  DATA_ROOT = /scratch/bai.xiang/Open-o3-Video
  video_path / image_path are relative to DATA_ROOT/videos/
  GQA: image_path is bare filename (e.g. 2331819.jpg) → file at videos/gqa/2331819.jpg
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

# Default data root (Open-o3-Video layout)
DEFAULT_DATA_ROOT = "/scratch/bai.xiang/Open-o3-Video"


def find_video_file(video_path_from_json: str, data_root: str) -> str | None:
    """
    Resolve video_path from JSON to an existing file under data_root/videos/.

    JSON video_path is relative to DATA_ROOT/videos/. Tries:
      1. videos_dir / video_path (primary: Open-o3 layout)
      2. PLM: stgr/plm/videos/sav_*.mp4
      3. Temporal grounding: stgr/temporal_grounding/videos/{source}/videos/{filename}
    """
    videos_dir = Path(data_root) / "videos"
    path_str = video_path_from_json.strip()
    if not path_str:
        return None

    # 1. Primary: path as stored (relative to videos/)
    direct = videos_dir / path_str
    if direct.exists():
        return str(direct)

    # 2. PLM: sav_*.mp4 under stgr/plm/videos/
    if path_str.startswith("sav_") and path_str.endswith(".mp4"):
        p = videos_dir / "stgr" / "plm" / "videos" / path_str
        if p.exists():
            return str(p)

    # 3. Temporal grounding: e.g. coin/..., activitynet/... → stgr/temporal_grounding/videos/{source}/videos/{filename}
    if "/" in path_str:
        parts = Path(path_str).parts
        filename = parts[-1]
        source = parts[0] if parts else ""
        for candidate in [
            videos_dir / "stgr" / "temporal_grounding" / "videos" / source / "videos" / filename,
            videos_dir / "stgr" / "temporal_grounding" / source / "videos" / filename,
            videos_dir / "stgr" / "temporal_grounding" / "videos" / filename,
        ]:
            if candidate.exists():
                return str(candidate)

    return None


def find_image_file(image_path_from_json: str, data_root: str) -> str | None:
    """
    Resolve image_path from JSON. GQA uses bare filename (e.g. 2331819.jpg)
    and file lives at videos/gqa/<filename>.
    """
    videos_dir = Path(data_root) / "videos"
    path_str = image_path_from_json.strip()
    if not path_str:
        return None

    # 1. Direct under videos/ (e.g. treevgr, or full path)
    direct = videos_dir / path_str
    if direct.exists():
        return str(direct)

    # 2. GQA: no prefix in JSON, file in videos/gqa/
    if "/" not in path_str:
        gqa_path = videos_dir / "gqa" / path_str
        if gqa_path.exists():
            return str(gqa_path)

    return None


def filter_dataset(json_path: str, data_root: str, output_path: str) -> None:
    """Filter dataset to only samples with existing video or image files."""
    print("=" * 70)
    print("Filtering STGR Dataset by Available Videos/Images")
    print("=" * 70)
    print(f"Input: {json_path}")
    print(f"Data root: {data_root}")
    print(f"Output: {output_path}\n")

    with open(json_path) as f:
        data = json.load(f)

    print(f"Original samples: {len(data):,}")

    filtered_data = []
    found_by_source = {}
    missing_by_source = {}

    for item in tqdm(data, desc="Checking files"):
        source = item.get("source", "unknown")
        video_path = item.get("video_path", "").strip()
        image_path = item.get("image_path", "").strip()

        # Video sample
        if video_path:
            actual = find_video_file(video_path, data_root)
            if actual:
                item["video_path_full"] = actual
                filtered_data.append(item)
                found_by_source[source] = found_by_source.get(source, 0) + 1
            else:
                missing_by_source[source] = missing_by_source.get(source, 0) + 1
            continue

        # Image-only sample (e.g. GQA)
        if image_path:
            actual = find_image_file(image_path, data_root)
            if actual:
                item["image_path_full"] = actual
                filtered_data.append(item)
                found_by_source[source] = found_by_source.get(source, 0) + 1
            else:
                missing_by_source[source] = missing_by_source.get(source, 0) + 1
            continue

        missing_by_source[source] = missing_by_source.get(source, 0) + 1

    print()
    print(f"✅ Filtered samples: {len(filtered_data):,} ({len(filtered_data)/len(data)*100:.1f}%)")
    print(f"❌ Missing samples: {len(data) - len(filtered_data):,}\n")

    if found_by_source:
        print("Found by source:")
        for src, count in sorted(found_by_source.items(), key=lambda x: -x[1]):
            print(f"  ✅ {src}: {count:,} samples")
        print()

    if missing_by_source:
        print("Missing by source:")
        for src, count in sorted(missing_by_source.items(), key=lambda x: -x[1])[:15]:
            print(f"  ❌ {src}: {count:,} samples")
        if len(missing_by_source) > 15:
            print(f"  ... and {len(missing_by_source) - 15} more sources")
        print()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"✅ Saved filtered dataset: {out}")
    print(f"   Size: {out.stat().st_size / 1024 / 1024:.1f} MB")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter STGR JSON to samples with available media")
    parser.add_argument("--json", required=True, help="Input JSON (STGR-SFT.json or STGR-RL.json)")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Open-o3-Video data root")
    parser.add_argument("--output", required=True, help="Output filtered JSON path")
    args = parser.parse_args()

    filter_dataset(args.json, args.data_root, args.output)
