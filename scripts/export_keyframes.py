#!/usr/bin/env python3
"""
Export keyframe images for PLM dataset

This script:
1) Reads the JSON file containing entries with:
   - video_path_full (preferred)
   - video_path (fallback)
   - key_frames: [{idx, time, path}, ...]
2) Uses each key_frames[].time timestamp to seek the source video
3) Saves the frame image to OUTPUT_ROOT / key_frames[].path.

Notes:
- Update OUTPUT_ROOT below (or pass --output-root) before running.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2


# update this to where we want to store the keyframes in the cluster
OUTPUT_ROOT = Path("/path/to/save/keyframes")
DEFAULT_VIDEOS_ROOT = Path("/scratch/bai.xiang/Open-o3-Video/videos/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export keyframes from PLM subset JSON using key_frames timestamps."
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to input JSON (e.g. PLM subset JSON).",
    )
    parser.add_argument(
        "--videos-root",
        default=str(DEFAULT_VIDEOS_ROOT),
        help=(
            "Root folder containing source videos. "
            "Used when video_path_full is missing or not directly usable."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="",
        help=(
            "Where to save keyframes. If omitted, uses OUTPUT_ROOT variable in this script."
        ),
    )
    parser.add_argument(
        "--source-filter",
        default="STR_plm_rdcap",
        help=(
            "Filter by source. Defaults to STR_plm_rdcap."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frame files. By default existing files are skipped.",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list, got: {type(data).__name__}")
    return [x for x in data if isinstance(x, dict)]


def resolve_video_path(entry: Dict[str, Any], videos_root: Path) -> Optional[Path]:
    """
    Resolve video path using video_path_full primarily, then video_path.
    """
    vp = entry.get("video_path")
    vpf = entry.get("video_path_full")

    # 1) Prefer video_path_full from JSON.
    if isinstance(vpf, str) and vpf:
        p = Path(vpf)
        # Absolute path exactly as stored in JSON.
        if p.is_absolute() and p.exists():
            return p
        # If it is not directly usable, try videos_root + basename.
        if videos_root:
            candidate = videos_root / p.name
            if candidate.exists():
                return candidate

    # 2) Fallback to video_path.
    if isinstance(vp, str) and vp:
        p = Path(vp)
        if p.is_absolute() and p.exists():
            return p
        if videos_root:
            candidate = videos_root / vp
            if candidate.exists():
                return candidate

    return None


def collect_requests(
    entries: List[Dict[str, Any]],
    videos_root: Path,
    source_filter: str,
) -> Tuple[Dict[Path, List[Tuple[float, Path]]], int, int]:
    """
    Build: video_file -> [(time_sec, relative_output_path), ...]
    Returns: requests_map, skipped_entries, missing_video_entries
    """
    requests: Dict[Path, List[Tuple[float, Path]]] = defaultdict(list)
    skipped_entries = 0
    missing_video_entries = 0

    for e in entries:
        if source_filter and e.get("source") != source_filter:
            continue

        kfs = e.get("key_frames")
        if not isinstance(kfs, list) or not kfs:
            skipped_entries += 1
            continue

        video_file = resolve_video_path(e, videos_root)
        if video_file is None:
            missing_video_entries += 1
            continue

        for kf in kfs:
            if not isinstance(kf, dict):
                continue
            t = kf.get("time")
            rel = kf.get("path")
            if not isinstance(t, (int, float)):
                continue
            if not isinstance(rel, str) or not rel.strip():
                continue
            requests[video_file].append((float(t), Path(rel)))

    return requests, skipped_entries, missing_video_entries


def export_for_video(
    video_file: Path,
    items: List[Tuple[float, Path]],
    output_root: Path,
    overwrite: bool,
) -> Tuple[int, int]:
    """
    Returns: (written_count, failed_count)
    """
    written = 0
    failed = 0

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return 0, len(items)

    # Optional: sort by time for reproducibility
    items_sorted = sorted(items, key=lambda x: x[0])

    for t_sec, rel_out in items_sorted:
        out_file = output_root / rel_out
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists() and not overwrite:
            continue

        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            failed += 1
            continue

        ok_write = cv2.imwrite(str(out_file), frame)
        if ok_write:
            written += 1
        else:
            failed += 1

    cap.release()
    return written, failed


def main() -> None:
    args = parse_args()

    json_path = Path(args.json).expanduser().resolve()
    videos_root = Path(args.videos_root).expanduser().resolve() if args.videos_root else Path()
    output_root = (
        Path(args.output_root).expanduser().resolve() if args.output_root else OUTPUT_ROOT.expanduser()
    )

    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    if not output_root:
        raise ValueError("Output root is empty. Set OUTPUT_ROOT or pass --output-root.")

    entries = load_json(json_path)
    requests, skipped_entries, missing_video_entries = collect_requests(
        entries=entries,
        videos_root=videos_root,
        source_filter=args.source_filter.strip(),
    )

    total_written = 0
    total_failed = 0
    total_requests = sum(len(v) for v in requests.values())

    for video_file, items in requests.items():
        written, failed = export_for_video(
            video_file=video_file,
            items=items,
            output_root=output_root,
            overwrite=args.overwrite,
        )
        total_written += written
        total_failed += failed
        print(f"[{video_file.name}] requests={len(items)} written={written} failed={failed}")

    print("\nDone.")
    print(f"json: {json_path}")
    print(f"output_root: {output_root}")
    print(f"source_filter: {args.source_filter.strip()}")
    print(f"videos_seen: {len(requests)}")
    print(f"requests_total: {total_requests}")
    print(f"written_total: {total_written}")
    print(f"failed_total: {total_failed}")
    print(f"skipped_entries_no_keyframes: {skipped_entries}")
    print(f"entries_missing_video: {missing_video_entries}")


if __name__ == "__main__":
    main()