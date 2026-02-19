"""
Motion-aware trajectory reward for video reasoning.

Integrates with Open-o3 Video's modular reward system to add trajectory-level
motion evaluation (direction, speed, smoothness).

v2: Matches discrete motion primitive bins (dir, speed, scale) with ±1 adjacent
    bin partial credit instead of free-text direction word matching.
"""

import re
import json
import math
import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# No complex motion metrics needed - using discrete bin matching!

# Lightweight debug toggle for GRPO runs.
# Set DEBUG_MOTION_REWARD=1 in the environment to enable.
DEBUG_MOTION_REWARD = os.environ.get("DEBUG_MOTION_REWARD", "0") == "1"
_DEBUG_MOTION_MAX_LOGS = int(os.environ.get("DEBUG_MOTION_REWARD_MAX", "10"))
_debug_motion_logs = 0


# ============================================================================
# Discrete bin definitions (must match augment_motion_v2.py)
# ============================================================================

# Direction: 8 compass + STAT.  Ordered circularly for adjacency scoring.
COMPASS_ORDER = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
COMPASS_SET = set(COMPASS_ORDER) | {"STAT"}

# Speed: ordinal ranking for ±1 partial credit
SPEED_ORDER = ["stationary", "slow", "moderate", "fast"]
SPEED_SET = set(SPEED_ORDER)

# Scale: ordinal ranking for ±1 partial credit
SCALE_ORDER = ["receding", "stable", "approaching"]
SCALE_SET = set(SCALE_ORDER)

# Reward weights for each attribute
W_DIR = 0.40
W_SPEED = 0.30
W_SCALE = 0.30


# ============================================================================
# Adjacency-aware scoring helpers
# ============================================================================

def direction_score(pred: str, gt: str) -> float:
    """Score direction match with ±1 compass bin partial credit.
    
    Exact match → 1.0
    Adjacent bin (±45°) → 0.5
    STAT vs STAT → 1.0
    STAT vs anything else → 0.0
    Otherwise → 0.0
    """
    if pred == gt:
        return 1.0
    if pred == "STAT" or gt == "STAT":
        return 0.0
    # Circular adjacency on compass
    try:
        pi = COMPASS_ORDER.index(pred)
        gi = COMPASS_ORDER.index(gt)
    except ValueError:
        return 0.0
    diff = min(abs(pi - gi), 8 - abs(pi - gi))  # circular distance
    if diff == 1:
        return 0.5
    return 0.0


def ordinal_score(pred: str, gt: str, order: list) -> float:
    """Score ordinal attribute match with ±1 partial credit.
    
    Exact match → 1.0
    Off by 1 rank → 0.5
    Otherwise → 0.0
    """
    if pred == gt:
        return 1.0
    try:
        pi = order.index(pred)
        gi = order.index(gt)
    except ValueError:
        return 0.0
    if abs(pi - gi) == 1:
        return 0.5
    return 0.0


# ============================================================================
# Parsing helpers
# ============================================================================

def normalize_obj_name(name: str) -> str:
    """Normalize object names for identity matching."""
    if not name:
        return ""
    name = name.lower().strip()
    #keep alphanumerics and spaces only to reduce punctuation mismatch
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def parse_temporal_spatial_claims(think_content: str):
    """
    Parse think content to extract temporal-spatial claims.
    Format: <obj>name</obj><box>[x1,y1,x2,y2]</box>at<t>time</t>s
    
    Args:
        think_content: Text content between <think></think> tags
        
    Returns:
        List of dicts with keys: id, object_name, timestamp, bboxes
    """
    pattern = r"<obj>(.*?)</obj>((?:<box>\[.*?\]</box>)+)at<t>(.*?)</t>s"
    parsed_claims = []
    count = 0

    for match in re.finditer(pattern, think_content, re.DOTALL):
        try:
            object_name = match.group(1).strip()
            all_boxes_str = match.group(2)
            timestamp_str = match.group(3).strip()
            timestamp = float(timestamp_str)
            
            individual_box_strs = re.findall(r'\[.*?\]', all_boxes_str)
            bboxes = [json.loads(b_str) for b_str in individual_box_strs]
            
            parsed_claims.append({
                "id": count,
                "object_name": object_name,
                "timestamp": timestamp,
                "bboxes": bboxes
            })
            count += 1

        except (json.JSONDecodeError, ValueError, IndexError) as e:
            continue
            
    return parsed_claims


def parse_v2_motion_tags(think_content: str) -> Dict[str, Dict[str, str]]:
    """Parse all v2 self-closing motion tags, keyed by normalised object name.
    
    Format: <motion obj="person" dir="S" speed="slow" scale="receding"/>
    
    If multiple tags exist for the same object, the LAST one wins
    (matches augment_motion_v2.py insertion semantics).
    
    Returns:
        { "person": {"dir": "S", "speed": "slow", "scale": "receding"}, ... }
    """
    result = {}
    for m in re.finditer(r'<motion\s+([^/]*?)/>', think_content):
        attrs_str = m.group(1)
        obj_m = re.search(r'obj="([^"]*)"', attrs_str)
        dir_m = re.search(r'dir="([^"]*)"', attrs_str)
        speed_m = re.search(r'speed="([^"]*)"', attrs_str)
        scale_m = re.search(r'scale="([^"]*)"', attrs_str)
        if obj_m and dir_m and speed_m and scale_m:
            key = normalize_obj_name(obj_m.group(1))
            result[key] = {
                "dir": dir_m.group(1),
                "speed": speed_m.group(1),
                "scale": scale_m.group(1),
            }
    return result


def parse_v1_motion_tags(think_content: str) -> Optional[str]:
    """Legacy: parse last v1 paired <motion>text</motion> tag for direction only.
    
    Returns direction string or None.
    """
    tags = re.findall(r'<motion>([^<]+)</motion>', think_content)
    if not tags:
        return None
    text = tags[-1].lower().strip()
    if "stationary" in text:
        return "STAT"
    # Map v1 direction words to compass
    v1_to_compass = {
        "rightward": "E", "leftward": "W", "upward": "N", "downward": "S",
        "up-right": "NE", "up-left": "NW", "down-right": "SE", "down-left": "SW",
    }
    for word, compass in v1_to_compass.items():
        if word in text:
            return compass
    return None


def convert_coord_format(bbox, image_size):
    """
    Convert normalized bbox [0,1] to pixel coordinates.
    
    Args:
        bbox: [nx_min, ny_min, nx_max, ny_max] in [0, 1]
        image_size: (width, height) tuple
        
    Returns:
        [x_min, y_min, x_max, y_max] in pixels
    """
    nx_min, ny_min, nx_max, ny_max = bbox
    width, height = image_size
    x_min = nx_min * width
    y_min = ny_min * height
    x_max = nx_max * width
    y_max = ny_max * height
    return [x_min, y_min, x_max, y_max]


# ============================================================================
# GT motion computation from bboxes (mirrors augment_motion_v2.py logic)
# ============================================================================

def compute_gt_direction_from_bboxes(bbox1, bbox2) -> str:
    """Compute compass direction from two pixel-space bboxes.
    
    Uses same convention as augment_motion_v2.py:
    - screen-up (dy < 0) → N
    - screen-right (dx > 0) → E
    """
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    
    dx = cx2 - cx1
    # Negate dy so screen-upward → positive → maps to N (matches augment_motion_v2)
    dy = -(cy2 - cy1)
    
    mag = math.sqrt(dx**2 + dy**2)
    
    # Threshold for "stationary" — use fraction of avg bbox diagonal
    diag1 = math.sqrt((bbox1[2]-bbox1[0])**2 + (bbox1[3]-bbox1[1])**2)
    diag2 = math.sqrt((bbox2[2]-bbox2[0])**2 + (bbox2[3]-bbox2[1])**2)
    avg_diag = (diag1 + diag2) / 2
    threshold = 0.02 * avg_diag  # 2% of object size
    
    if mag < max(threshold, 3.0):  # at least 3 pixels
        return "STAT"
    
    angle = math.atan2(dy, dx)
    deg = math.degrees(angle) % 360
    idx = int((deg + 22.5) / 45.0) % 8
    return COMPASS_ORDER[idx]


def compute_gt_speed_bin(bbox1, bbox2, dt, image_size) -> str:
    """Compute speed bin from two pixel-space bboxes and time delta.
    
    Normalises by object's own bbox diagonal, same as augment_motion_v2.py.
    """
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    
    disp = math.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
    
    diag1 = math.sqrt((bbox1[2]-bbox1[0])**2 + (bbox1[3]-bbox1[1])**2)
    diag2 = math.sqrt((bbox2[2]-bbox2[0])**2 + (bbox2[3]-bbox2[1])**2)
    avg_diag = (diag1 + diag2) / 2
    
    if dt <= 0 or avg_diag < 1e-9:
        return "stationary"
    
    norm_speed = (disp / dt) / avg_diag
    
    # Same thresholds as augment_motion_v2.py SPEED_BINS
    if norm_speed < 0.02:
        return "stationary"
    elif norm_speed < 0.10:
        return "slow"
    elif norm_speed < 0.30:
        return "moderate"
    else:
        return "fast"


def compute_gt_scale_bin(bbox1, bbox2) -> str:
    """Compute scale bin from two pixel-space bboxes."""
    a1 = max(0.0, bbox1[2]-bbox1[0]) * max(0.0, bbox1[3]-bbox1[1])
    a2 = max(0.0, bbox2[2]-bbox2[0]) * max(0.0, bbox2[3]-bbox2[1])
    if a1 < 1e-9 or a2 < 1e-9:
        return "stable"
    log_ratio = math.log(a2 / a1)
    # Same threshold as augment_motion_v2.py
    if log_ratio > 0.15:
        return "approaching"
    elif log_ratio < -0.15:
        return "receding"
    return "stable"


# ============================================================================
# Main reward function
# ============================================================================

def motion_trajectory_reward(completions, **kwargs):
    """
    Motion-aware trajectory reward function.
    
    v2: Compares discrete motion bins (dir, speed, scale) from model output
    against GT bins computed from bbox trajectories. Uses ±1 adjacent bin
    partial credit for direction (circular compass) and ordinal attributes.
    
    Simpler approach inspired by Open-o3-Video's thk_spatial_reward:
    1. Match EACH predicted observation to closest GT frame (within threshold)
    2. If >=2 predictions match to DIFFERENT GT frames, compute trajectory reward
    3. Score each motion attribute with partial credit
    
    Args:
        completions: List of model completions (list of dicts with 'content')
        **kwargs: Additional arguments including:
            - task: List of task types
            - key_items: GT spatial annotations (can be sparse!)
            - key_frames: GT temporal keyframes  
            - image_size: Image dimensions
            
    Returns:
        List of rewards in [0, 1] for each completion
    """
    motion_rewards = []
    idx = 0
    
    for completion in completions:
        think_match = re.search(r"<think>(.*?)</think>", completion[0]["content"], re.DOTALL)
        
        if not think_match:
            motion_rewards.append(0.0)
            idx += 1
            continue
        
        task = kwargs.get('task', [''])[0] if isinstance(kwargs.get('task'), list) else kwargs.get('task', '')
        
        # Only for temporal-spatial tasks
        if task not in ["temporal-spatial free-form QA", "General video QA Free-form", "General video QA MCQ"]:
            motion_rewards.append(0.0)
            idx += 1
            continue
        
        think_content = think_match.group(1)
        parsed_claims = parse_temporal_spatial_claims(think_content)
        
        # Need at least 2 observations for trajectory
        if not parsed_claims or len(parsed_claims) < 2:
            motion_rewards.append(0.0)
            idx += 1
            continue
        
        try:
            gt_items = kwargs.get("key_items", [None])[idx] if idx < len(kwargs.get("key_items", [])) else None
            gt_frames = kwargs.get("key_frames", [None])[idx] if idx < len(kwargs.get("key_frames", [])) else None
            image_size = kwargs.get("image_size", [(640, 480)])[idx] if idx < len(kwargs.get("image_size", [])) else (640, 480)
            
            if gt_items is None:
                motion_rewards.append(0.0)
                idx += 1
                continue
            
            # Get available GT times (from key_items, not just key_frames!)
            available_gt_frame_indices = set(gt_items.keys())
            gt_frame_times = {}
            
            # Map frame indices to times (some may not be in key_frames)
            if gt_frames:
                for frame in gt_frames:
                    gt_frame_times[str(frame['idx'])] = frame['time']
            
            # For frames in key_items but not in key_frames, estimate time
            for frame_idx in available_gt_frame_indices:
                if frame_idx not in gt_frame_times:
                    # Estimate based on frame index (assume ~30fps)
                    gt_frame_times[frame_idx] = int(frame_idx) / 30.0
            
            # Match each prediction to closest GT frame (like thk_spatial_reward)
            #enforce object identity consistency when calculating traj reward
            matched_predictions = []
            threshold = 2.0  # seconds - relaxed threshold
            
            for claim in parsed_claims:
                pred_time = claim['timestamp']
                pred_bbox = claim['bboxes'][0] if claim['bboxes'] else None
                pred_obj_name = claim.get("object_name", "")
                pred_obj_name_norm = normalize_obj_name(pred_obj_name)
                
                if pred_bbox is None:
                    continue
                
                # Convert to pixel coordinates if normalized
                if all(0 <= c <= 1 for c in pred_bbox):
                    pred_bbox = convert_coord_format(pred_bbox, image_size)
                
                # Find closest GT frame
                closest_frame_idx = None
                min_time_diff = float('inf')
                
                for frame_idx, gt_time in gt_frame_times.items():
                    time_diff = abs(gt_time - pred_time)
                    if time_diff < min_time_diff and time_diff < threshold:
                        min_time_diff = time_diff
                        closest_frame_idx = frame_idx
                
                if closest_frame_idx and closest_frame_idx in gt_items:
                    #this part should enforce same object matching 
                    frame_objects = gt_items[closest_frame_idx]
                    matched_gt_bbox = None
                    for gt_obj_name, gt_obj_bboxes in frame_objects.items():
                        if normalize_obj_name(gt_obj_name) == pred_obj_name_norm and gt_obj_bboxes:
                            matched_gt_bbox = gt_obj_bboxes[0]
                            break

                    #if there is no same object GT found at this frame, skip 
                    if matched_gt_bbox is None:
                        continue

                    gt_bbox = convert_coord_format(matched_gt_bbox, image_size)

                    matched_predictions.append({
                        'pred_bbox': pred_bbox,
                        'gt_bbox': gt_bbox,
                        'pred_time': pred_time,
                        'gt_time': gt_frame_times[closest_frame_idx],
                        'frame_idx': closest_frame_idx,
                        'object_name': pred_obj_name,
                        'object_name_norm': pred_obj_name_norm,
                    })
            
            # Need at least 2 matched predictions spanning different frames
            if len(matched_predictions) < 2:
                motion_rewards.append(0.0)
                idx += 1
                continue

            #group by object identity so trajectories are computed on the same object only
            grouped = {}
            for m in matched_predictions:
                grouped.setdefault(m["object_name_norm"], []).append(m)

            per_object_rewards = []

            # Parse predicted motion attributes from v2 tags (or fall back to v1)
            pred_motion_tags = parse_v2_motion_tags(think_content)
            pred_v1_dir_fallback = parse_v1_motion_tags(think_content)

            for obj_name_norm, obj_matches in grouped.items():
                if len(obj_matches) < 2:
                    continue

                unique_frames = set(m['frame_idx'] for m in obj_matches)
                if len(unique_frames) < 2:
                    continue

                obj_matches = sorted(obj_matches, key=lambda x: x['pred_time'])

                first_gt = obj_matches[0]['gt_bbox']
                last_gt = obj_matches[-1]['gt_bbox']
                gt_dt = obj_matches[-1]['gt_time'] - obj_matches[0]['gt_time']

                # --- Compute GT bins from bbox trajectory ---
                # Prefer precomputed gt_motion field (from augment_motion_v2.py)
                # Fall back to computing from bboxes on-the-fly
                gt_motion_field = kwargs.get("gt_motion", [None])
                gt_motion_for_sample = gt_motion_field[idx] if idx < len(gt_motion_field) and gt_motion_field[idx] else None

                # Try to find this object in precomputed gt_motion
                precomputed = None
                if gt_motion_for_sample and isinstance(gt_motion_for_sample, dict):
                    # Try exact name match first, then normalized
                    for gt_obj_key, gt_desc in gt_motion_for_sample.items():
                        if normalize_obj_name(gt_obj_key) == obj_name_norm:
                            precomputed = gt_desc
                            break

                if precomputed:
                    gt_dir = precomputed.get("dir", "STAT")
                    gt_spd = precomputed.get("speed", "stationary")
                    gt_scl = precomputed.get("scale", "stable")
                else:
                    gt_dir = compute_gt_direction_from_bboxes(first_gt, last_gt)
                    gt_spd = compute_gt_speed_bin(first_gt, last_gt, gt_dt, image_size)
                    gt_scl = compute_gt_scale_bin(first_gt, last_gt)

                # --- Get predicted bins ---
                pred_attrs = pred_motion_tags.get(obj_name_norm)

                if pred_attrs:
                    # v2 path: structured discrete bins
                    p_dir = pred_attrs.get("dir", "")
                    p_spd = pred_attrs.get("speed", "")
                    p_scl = pred_attrs.get("scale", "")
                    
                    d_score = direction_score(p_dir, gt_dir)
                    s_score = ordinal_score(p_spd, gt_spd, SPEED_ORDER)
                    sc_score = ordinal_score(p_scl, gt_scl, SCALE_ORDER)
                    
                    object_reward = W_DIR * d_score + W_SPEED * s_score + W_SCALE * sc_score
                else:
                    # v1 fallback: only direction from free-text, no speed/scale
                    p_dir = pred_v1_dir_fallback
                    if p_dir and gt_dir:
                        d_score = direction_score(p_dir, gt_dir)
                        # v1 has no speed/scale info, give half credit for those
                        # to not over-penalise during transition
                        object_reward = W_DIR * d_score + (W_SPEED + W_SCALE) * 0.25
                    else:
                        object_reward = 0.0

                per_object_rewards.append(max(0.0, min(1.0, float(object_reward))))

            if per_object_rewards:
                # we can use mean reward over valid object trajectories
                motion_rewards.append(float(np.mean(per_object_rewards)))
            else:
                motion_rewards.append(0.1 if len(matched_predictions) >= 2 else 0.0)

            # Optional debug logging for GRPO sanity checks
            global _debug_motion_logs
            if DEBUG_MOTION_REWARD and _debug_motion_logs < _DEBUG_MOTION_MAX_LOGS:
                _debug_motion_logs += 1
                # Summarise first object for debug
                first_obj = next(iter(grouped.keys()), "?")
                p_tag = pred_motion_tags.get(first_obj, {})
                print(
                    "[motion_reward] "
                    f"task={task}, "
                    f"claims={len(parsed_claims)}, "
                    f"matched={len(matched_predictions)}, "
                    f"objects={len(per_object_rewards)}, "
                    f"v2_tags={len(pred_motion_tags)}, "
                    f"pred={p_tag}, "
                    f"reward={motion_rewards[-1]:.3f}",
                    flush=True,
                )
        
        except Exception as e:
            motion_rewards.append(0.0)
        
        idx += 1
    
    return motion_rewards