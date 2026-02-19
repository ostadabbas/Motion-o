#!/usr/bin/env python3
"""
Motion Chain of Thought (MCoT) v2 — Discrete Motion Primitives.

Replaces free-text continuous speed/acceleration with a structured tuple
of discrete, categorically-binned motion attributes computed from bbox
centroids and areas. Tags ALL objects (single-frame = stationary).

Output format:
  <motion obj="person" dir="E" speed="slow" scale="approaching"/>
  <motion obj="cup" dir="STAT" speed="stationary" scale="stable"/>

Attributes:
  dir   — 8 compass bins + STAT  (9 classes)
  speed — stationary/slow/moderate/fast  (4 classes)
  scale — approaching/stable/receding  (3 classes)
  shape — linear/curved/oscillating/stationary  (4 classes, ≥3 frames)
  phase — accelerating/constant/decelerating/reversing  (4 classes, ≥3 frames)
"""

import json
import argparse
import re
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def centroid(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: List[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_diagonal(bbox: List[float]) -> float:
    x1, y1, x2, y2 = bbox
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ---------------------------------------------------------------------------
# Direction — 8 compass bins + STAT
# ---------------------------------------------------------------------------

# Bin edges at multiples of 45°, centered so 0° → E
COMPASS_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def angle_to_compass(angle_rad: float) -> str:
    """Map angle (radians, math convention: 0=right, CCW positive) to compass.

    Note: in image coordinates y increases downward, so we negate dy before
    computing atan2 so that "upward on screen" maps to N.
    """
    deg = math.degrees(angle_rad) % 360
    idx = int((deg + 22.5) / 45.0) % 8
    return COMPASS_LABELS[idx]


def compute_direction(centroids: List[Tuple[float, float]],
                      magnitudes: List[float],
                      stat_threshold: float) -> str:
    """Magnitude-weighted dominant direction from centroid displacements.

    Each inter-frame displacement votes for its compass bin, weighted by its
    magnitude. This makes large real movements dominate over small jitter.
    """
    if len(centroids) < 2:
        return "STAT"

    bins = defaultdict(float)
    total_mag = 0.0

    for i in range(len(centroids) - 1):
        dx = centroids[i + 1][0] - centroids[i][0]
        # Negate dy so screen-upward → positive → maps to N
        dy = -(centroids[i + 1][1] - centroids[i][1])
        mag = magnitudes[i] if i < len(magnitudes) else math.sqrt(dx**2 + dy**2)
        if mag < 1e-9:
            continue
        angle = math.atan2(dy, dx)
        label = angle_to_compass(angle)
        bins[label] += mag
        total_mag += mag

    if total_mag < stat_threshold:
        return "STAT"

    return max(bins, key=bins.get)


# ---------------------------------------------------------------------------
# Speed — 4 ordinal bins
# ---------------------------------------------------------------------------

# Thresholds in normalised-displacement-per-second units.
# These are fractions of the object's own bbox diagonal per second.
# Tune after running --stats on your dataset.
SPEED_BINS = [
    (0.02, "stationary"),   # < 2% of own diagonal / sec
    (0.10, "slow"),         # 2–10%
    (0.30, "moderate"),     # 10–30%
    (None, "fast"),         # > 30%
]


def compute_speed_bin(norm_speed: float) -> str:
    for threshold, label in SPEED_BINS:
        if threshold is not None and norm_speed < threshold:
            return label
    return "fast"


# ---------------------------------------------------------------------------
# Scale change — 3 classes
# ---------------------------------------------------------------------------

SCALE_LOG_THRESHOLD = 0.15  # ±~16 % area change


def compute_scale(bboxes: List[List[float]]) -> str:
    a_first = bbox_area(bboxes[0])
    a_last = bbox_area(bboxes[-1])
    if a_first < 1e-9 or a_last < 1e-9:
        return "stable"
    log_ratio = math.log(a_last / a_first)
    if log_ratio > SCALE_LOG_THRESHOLD:
        return "approaching"
    elif log_ratio < -SCALE_LOG_THRESHOLD:
        return "receding"
    return "stable"


# ---------------------------------------------------------------------------
# Shape — 4 classes (needs ≥ 3 frames)
# ---------------------------------------------------------------------------

def compute_shape(centroids: List[Tuple[float, float]]) -> str:
    if len(centroids) < 3:
        return "linear"

    # Net displacement vs total path length  (sinuosity)
    net_dx = centroids[-1][0] - centroids[0][0]
    net_dy = centroids[-1][1] - centroids[0][1]
    net_disp = math.sqrt(net_dx**2 + net_dy**2)

    path_len = 0.0
    for i in range(len(centroids) - 1):
        dx = centroids[i + 1][0] - centroids[i][0]
        dy = centroids[i + 1][1] - centroids[i][1]
        path_len += math.sqrt(dx**2 + dy**2)

    if path_len < 1e-9:
        return "stationary"

    sinuosity = net_disp / path_len  # 1.0 = perfectly straight

    # Check for direction reversals (oscillation)
    reversals = 0
    for i in range(len(centroids) - 2):
        dx1 = centroids[i + 1][0] - centroids[i][0]
        dy1 = centroids[i + 1][1] - centroids[i][1]
        dx2 = centroids[i + 2][0] - centroids[i + 1][0]
        dy2 = centroids[i + 2][1] - centroids[i + 1][1]
        dot = dx1 * dx2 + dy1 * dy2
        if dot < 0:
            reversals += 1

    n_segments = len(centroids) - 1
    reversal_ratio = reversals / max(1, n_segments - 1)

    if reversal_ratio > 0.4:
        return "oscillating"
    if sinuosity > 0.85:
        return "linear"
    return "curved"


# ---------------------------------------------------------------------------
# Phase — 4 classes (needs ≥ 3 frames)
# ---------------------------------------------------------------------------

def compute_phase(magnitudes: List[float],
                  timestamps: List[float]) -> str:
    if len(magnitudes) < 2:
        return "constant"

    # Compute inter-frame speeds
    speeds = []
    for i in range(len(magnitudes)):
        if i < len(timestamps) - 1:
            dt = timestamps[i + 1] - timestamps[i]
            speeds.append(magnitudes[i] / dt if dt > 0 else 0.0)

    if len(speeds) < 2:
        return "constant"

    # Check monotonicity with tolerance
    diffs = [speeds[i + 1] - speeds[i] for i in range(len(speeds) - 1)]
    pos = sum(1 for d in diffs if d > 0.005)
    neg = sum(1 for d in diffs if d < -0.005)
    n = len(diffs)

    # Check for direction reversal in displacement vectors
    # (distinct from speed change — this means the object turned around)
    if any(d < -0.5 * max(speeds) for d in diffs) and any(d > 0.5 * max(speeds) for d in diffs):
        return "reversing"

    if pos > 0.6 * n and neg <= 0.2 * n:
        return "accelerating"
    if neg > 0.6 * n and pos <= 0.2 * n:
        return "decelerating"
    return "constant"


# ---------------------------------------------------------------------------
# Main descriptor — assembles all attributes
# ---------------------------------------------------------------------------

def compute_motion_descriptor(bboxes: List[List[float]],
                              timestamps: List[float]) -> Dict[str, str]:
    """Compute discrete motion descriptor from bbox trajectory.

    Returns dict with keys: dir, speed, scale, shape, phase.
    Works for any number of frames (≥1).
    """
    n = len(bboxes)

    # Single frame → everything stationary
    if n < 2:
        return {
            "dir": "STAT",
            "speed": "stationary",
            "scale": "stable",
            "shape": "stationary",
            "phase": "constant",
        }

    cents = [centroid(b) for b in bboxes]

    # Per-segment magnitudes (used for weighting + speed)
    mags = []
    for i in range(n - 1):
        dx = cents[i + 1][0] - cents[i][0]
        dy = cents[i + 1][1] - cents[i][1]
        mags.append(math.sqrt(dx**2 + dy**2))

    # Normalise by object's own average bbox diagonal (noise-invariant)
    avg_diag = sum(bbox_diagonal(b) for b in bboxes) / n
    stat_threshold = 0.02 * avg_diag  # 2 % of own size

    # Normalised speed: avg displacement per second / avg diagonal
    total_dist = sum(mags)
    total_time = timestamps[-1] - timestamps[0]
    if total_time > 0 and avg_diag > 1e-9:
        norm_speed = (total_dist / total_time) / avg_diag
    else:
        norm_speed = 0.0

    d = compute_direction(cents, mags, stat_threshold)
    sp = compute_speed_bin(norm_speed)
    sc = compute_scale(bboxes)

    # Enforce consistency: dir and speed must agree on stationarity
    if d == "STAT":
        sp = "stationary"
    elif sp == "stationary":
        d = "STAT"

    # Shape and phase need ≥ 3 frames
    sh = compute_shape(cents) if n >= 3 else ("stationary" if d == "STAT" else "linear")
    ph = compute_phase(mags, timestamps) if n >= 3 else "constant"

    return {"dir": d, "speed": sp, "scale": sc, "shape": sh, "phase": ph}


def format_motion_tag(obj_name: str, desc: Dict[str, str],
                      include_extended: bool = False) -> str:
    """Format descriptor as XML-style self-closing tag.

    Core (always): dir, speed, scale
    Extended (≥3 frames & flag): shape, phase
    """
    parts = [
        f'obj="{obj_name}"',
        f'dir="{desc["dir"]}"',
        f'speed="{desc["speed"]}"',
        f'scale="{desc["scale"]}"',
    ]
    if include_extended and desc.get("shape") not in (None, "linear", "stationary"):
        parts.append(f'shape="{desc["shape"]}"')
    if include_extended and desc.get("phase") not in (None, "constant"):
        parts.append(f'phase="{desc["phase"]}"')

    return "<motion " + " ".join(parts) + "/>"


# ---------------------------------------------------------------------------
# Dataset wiring (group, locate, insert)
# ---------------------------------------------------------------------------

def group_boxes_by_object(key_items: Dict,
                          key_frames: List[Dict]
                          ) -> Dict[str, List[Tuple[List[float], float]]]:
    """Group bounding boxes by object name across timestamps."""
    tracked = defaultdict(list)
    idx_to_time = {str(f['idx']): f['time'] for f in key_frames}

    for frame_idx, objects in key_items.items():
        ts = idx_to_time.get(frame_idx)
        if ts is None:
            continue
        for obj_name, bboxes in objects.items():
            if bboxes and len(bboxes) > 0:
                tracked[obj_name].append((bboxes[0], ts))

    for obj_name in tracked:
        tracked[obj_name].sort(key=lambda x: x[1])

    return dict(tracked)


def find_last_object_mention(text: str, obj_name: str) -> Optional[int]:
    """Position after the last <obj>name</obj>...<t>...</t>s span."""
    pat = rf"<obj>{re.escape(obj_name)}</obj>.*?<t>[\d.]+</t>s"
    matches = list(re.finditer(pat, text, re.DOTALL))
    return matches[-1].end() if matches else None


def insert_motion_tag(text: str, obj_name: str, tag: str) -> str:
    """Insert motion tag after the last temporal-spatial mention."""
    pos = find_last_object_mention(text, obj_name)
    if pos is None:
        return text
    return text[:pos] + tag + text[pos:]


# ---------------------------------------------------------------------------
# Per-sample augmentation
# ---------------------------------------------------------------------------

ELIGIBLE_TASKS = {
    "temporal-spatial free-form QA",
    "General video QA Free-form",
    "General video QA MCQ",
}


def compute_gt_motion(sample: Dict,
                      tag_single_frame: bool = True) -> Optional[Dict[str, Dict[str, str]]]:
    """Compute GT motion descriptors from key_items + key_frames.

    Works on ANY sample that has key_items and key_frames — including RL data
    that has no reasoning_process. Returns a dict keyed by object name with
    motion descriptor dicts as values.

    Returns:
        { "person": {"dir": "S", "speed": "slow", "scale": "receding"}, ... }
        or None if no objects can be tracked.
    """
    key_items = sample.get('key_items', {})
    key_frames = sample.get('key_frames', [])

    if not key_items or not key_frames:
        return None

    tracked = group_boxes_by_object(key_items, key_frames)
    if not tracked:
        return None

    gt_motion = {}
    for obj_name, traj in tracked.items():
        bboxes = [b for b, _ in traj]
        timestamps = [t for _, t in traj]

        if len(bboxes) < 2 and not tag_single_frame:
            continue

        desc = compute_motion_descriptor(bboxes, timestamps)
        gt_motion[obj_name] = desc

    return gt_motion if gt_motion else None


def augment_sample(sample: Dict,
                   include_extended: bool = False,
                   tag_single_frame: bool = True,
                   add_gt_motion: bool = True) -> Dict:
    """Augment a single sample with discrete motion tags.

    Args:
        sample: One entry from the SFT or RL JSON.
        include_extended: If True, include shape/phase when non-trivial.
        tag_single_frame: If True, tag single-frame objects as stationary.
                          This is CRITICAL for solving the sparsity problem
                          (coverage goes from ~26 % → ~100 %).
        add_gt_motion: If True, always add precomputed gt_motion field
                       (works for RL data even without reasoning_process).
    """
    out = sample.copy()

    # Always compute and store GT motion descriptors if possible
    # This is used by motion_trajectory_reward during GRPO
    if add_gt_motion and sample.get('task', '') in ELIGIBLE_TASKS:
        gt_motion = compute_gt_motion(sample, tag_single_frame)
        if gt_motion:
            out['gt_motion'] = gt_motion

    # If no reasoning_process, we can't insert tags (RL data) — return with gt_motion
    reasoning = sample.get('reasoning_process', '')
    if not reasoning:
        return out

    if sample.get('task', '') not in ELIGIBLE_TASKS:
        return out

    key_items = sample.get('key_items', {})
    key_frames = sample.get('key_frames', [])

    if not key_items or not key_frames:
        return out

    tracked = group_boxes_by_object(key_items, key_frames)
    if not tracked:
        return out

    augmented = reasoning

    for obj_name, traj in tracked.items():
        bboxes = [b for b, _ in traj]
        timestamps = [t for _, t in traj]

        if len(bboxes) < 2 and not tag_single_frame:
            continue

        desc = compute_motion_descriptor(bboxes, timestamps)
        tag = format_motion_tag(obj_name, desc, include_extended)
        augmented = insert_motion_tag(augmented, obj_name, tag)

    out['reasoning_process'] = augmented
    return out


# ---------------------------------------------------------------------------
# Dataset-level statistics (for tuning bin thresholds)
# ---------------------------------------------------------------------------

def collect_speed_distribution(data: List[Dict]) -> List[float]:
    """Scan dataset and return all normalised speeds for threshold tuning."""
    speeds = []
    for sample in data:
        if sample.get('task', '') not in ELIGIBLE_TASKS:
            continue
        key_items = sample.get('key_items', {})
        key_frames = sample.get('key_frames', [])
        if not key_items or not key_frames:
            continue

        tracked = group_boxes_by_object(key_items, key_frames)
        for _, traj in tracked.items():
            bboxes = [b for b, _ in traj]
            timestamps = [t for _, t in traj]
            if len(bboxes) < 2:
                continue
            n = len(bboxes)
            cents = [centroid(b) for b in bboxes]
            mags = []
            for i in range(n - 1):
                dx = cents[i + 1][0] - cents[i][0]
                dy = cents[i + 1][1] - cents[i][1]
                mags.append(math.sqrt(dx**2 + dy**2))
            avg_diag = sum(bbox_diagonal(b) for b in bboxes) / n
            total_dist = sum(mags)
            total_time = timestamps[-1] - timestamps[0]
            if total_time > 0 and avg_diag > 1e-9:
                speeds.append((total_dist / total_time) / avg_diag)
    return speeds


def print_stats(speeds: List[float]) -> None:
    """Print percentile-based speed distribution for threshold tuning."""
    if not speeds:
        print("  No multi-frame trajectories found.")
        return
    speeds.sort()
    n = len(speeds)
    percentiles = [5, 10, 25, 33, 50, 66, 75, 90, 95]
    print(f"  Trajectories: {n}")
    for p in percentiles:
        idx = min(int(n * p / 100), n - 1)
        print(f"    P{p:02d}: {speeds[idx]:.4f}")
    print(f"    Min: {speeds[0]:.6f}  Max: {speeds[-1]:.4f}")
    print()
    # Suggest bins
    p33 = speeds[min(int(n * 0.33), n - 1)]
    p66 = speeds[min(int(n * 0.66), n - 1)]
    print(f"  Suggested SPEED_BINS (percentile-based):")
    print(f"    stationary: < {p33 * 0.15:.4f}")
    print(f"    slow:       < {p33:.4f}  (P33)")
    print(f"    moderate:   < {p66:.4f}  (P66)")
    print(f"    fast:       ≥ {p66:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Augment STGR dataset with discrete motion primitive tags (v2)")
    parser.add_argument('--input', type=str,
                        default='/mnt/data/stgr/json_data/STGR-SFT.json')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--samples', type=int, default=None,
                        help="Process only first N samples")
    parser.add_argument('--output-suffix', type=str, default='-motion-v2')
    parser.add_argument('--inspect', type=int, default=0,
                        help="Print first N augmented samples")
    parser.add_argument('--extended', action='store_true',
                        help="Include shape/phase attributes when non-trivial")
    parser.add_argument('--no-single-frame', action='store_true',
                        help="Skip single-frame objects (not recommended)")
    parser.add_argument('--stats', action='store_true',
                        help="Print speed distribution for threshold tuning, then exit")
    args = parser.parse_args()

    # --- Load ---
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    total = len(data)
    print(f"  Loaded {total} samples.")

    # --- Stats mode ---
    if args.stats:
        print("\nCollecting speed distribution...")
        speeds = collect_speed_distribution(data)
        print_stats(speeds)
        return

    # --- Augment ---
    n = min(args.samples, total) if args.samples else total
    print(f"Processing {n} samples...")

    augmented_data = []
    counts = {"augmented": 0, "multi_frame": 0, "single_only": 0, "skipped": 0, "gt_motion_added": 0}

    for i, sample in enumerate(data[:n]):
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n} ...")

        out = augment_sample(
            sample,
            include_extended=args.extended,
            tag_single_frame=not args.no_single_frame,
        )
        augmented_data.append(out)

        # Track gt_motion field additions (works for RL data too)
        if 'gt_motion' in out and 'gt_motion' not in sample:
            counts["gt_motion_added"] += 1

        if out.get('reasoning_process', '') != sample.get('reasoning_process', ''):
            counts["augmented"] += 1
            if "<motion" in out.get('reasoning_process', '') and 'dir="STAT"' not in out.get('reasoning_process', ''):
                counts["multi_frame"] += 1
            else:
                counts["single_only"] += 1
        else:
            counts["skipped"] += 1

    print(f"\nAugmentation complete!")
    print(f"  Total:        {n}")
    print(f"  Augmented:    {counts['augmented']}  "
          f"({100*counts['augmented']/n:.1f}%)")
    print(f"    with motion:  {counts['multi_frame']}")
    print(f"    static only:  {counts['single_only']}")
    print(f"  GT motion added: {counts['gt_motion_added']}  "
          f"({100*counts['gt_motion_added']/n:.1f}%)")
    print(f"  Unchanged:    {counts['skipped']}")

    # --- Tag census ---
    tag_counts = defaultdict(int)
    for sample in augmented_data:
        r = sample.get('reasoning_process', '')
        for attr in ['dir', 'speed', 'scale', 'shape', 'phase']:
            for m in re.finditer(rf'{attr}="([^"]+)"', r):
                tag_counts[f"{attr}={m.group(1)}"] += 1
    if tag_counts:
        print("\nTag distribution:")
        for k in sorted(tag_counts, key=lambda x: (x.split('=')[0], x)):
            print(f"    {k:30s}  {tag_counts[k]:>6d}")

    # --- Save ---
    if args.output is None:
        p = Path(args.input)
        out_path = p.parent / (p.stem + args.output_suffix + p.suffix)
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {out_path}...")
    with open(out_path, 'w') as f:
        json.dump(augmented_data, f, indent=2)
    print("Done!")

    # --- Inspect ---
    if args.inspect > 0:
        print(f"\n{'='*80}")
        print(f"Inspecting first {args.inspect} augmented samples:")
        print('=' * 80)
        for i, s in enumerate(augmented_data[:args.inspect]):
            print(f"\n--- Sample {i+1} ---")
            print(f"ID:   {s.get('id', 'N/A')}")
            print(f"Task: {s.get('task', 'N/A')}")
            print(f"Q:    {s.get('question', '')[:120]}...")
            r = s.get('reasoning_process', '')
            # Highlight motion tags
            highlighted = re.sub(
                r'(<motion [^/]+/>)',
                lambda m: f"\n  >>> {m.group(1)} <<<\n",
                r,
            )
            print(highlighted[:1200])
            if len(highlighted) > 1200:
                print("...")
            print()


if __name__ == '__main__':
    main()