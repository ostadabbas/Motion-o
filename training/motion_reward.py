"""
Motion-aware trajectory reward for video reasoning.

Integrates with Open-o3 Video's modular reward system to add trajectory-level
motion evaluation (direction, speed, smoothness).
"""

import re
import json
import numpy as np
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# No complex motion metrics needed - using simple word matching!


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


def parse_motion_direction(motion_text: str) -> Optional[str]:
    """
    Parse direction word from <motion> text.
    
    Examples:
        "upward motion" -> "upward"
        "leftward motion" -> "leftward"
        "down-left motion" -> "down-left"
        "stationary" -> "stationary"
    
    Returns:
        Direction string or None if not found
    """
    if not motion_text:
        return None
    
    motion_text = motion_text.lower().strip()
    
    # Check for stationary first
    if "stationary" in motion_text:
        return "stationary"
    
    # Check for compound directions
    if "up-left" in motion_text or "up left" in motion_text:
        return "up-left"
    if "up-right" in motion_text or "up right" in motion_text:
        return "up-right"
    if "down-left" in motion_text or "down left" in motion_text:
        return "down-left"
    if "down-right" in motion_text or "down right" in motion_text:
        return "down-right"
    
    # Check for simple directions
    if "upward" in motion_text or "up" in motion_text:
        return "upward"
    if "downward" in motion_text or "down" in motion_text:
        return "downward"
    if "leftward" in motion_text or "left" in motion_text:
        return "leftward"
    if "rightward" in motion_text or "right" in motion_text:
        return "rightward"
    
    return None


def compute_direction_from_bboxes(bbox1, bbox2) -> str:
    """
    Compute direction word from bbox movement.
    
    Args:
        bbox1: [x1, y1, x2, y2] first bbox
        bbox2: [x1, y1, x2, y2] second bbox
    
    Returns:
        Direction string: "upward", "downward", "leftward", "rightward", 
                          "up-left", "up-right", "down-left", "down-right", "stationary"
    """
    # Compute centroids
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    
    dx = cx2 - cx1
    dy = cy2 - cy1
    
    # Threshold for "stationary"
    threshold = 5.0  # pixels
    if abs(dx) < threshold and abs(dy) < threshold:
        return "stationary"
    
    # Determine direction
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    
    # If movement is mostly horizontal
    if abs_dx > abs_dy * 1.5:
        return "leftward" if dx < 0 else "rightward"
    
    # If movement is mostly vertical
    if abs_dy > abs_dx * 1.5:
        return "upward" if dy < 0 else "downward"
    
    # Diagonal movement
    if dx < 0 and dy < 0:
        return "up-left"
    if dx > 0 and dy < 0:
        return "up-right"
    if dx < 0 and dy > 0:
        return "down-left"
    if dx > 0 and dy > 0:
        return "down-right"
    
    return "stationary"


def motion_trajectory_reward(completions, **kwargs):
    """
    Motion-aware trajectory reward function.
    
    Simpler approach inspired by Open-o3-Video's thk_spatial_reward:
    1. Match EACH predicted observation to closest GT frame (within threshold)
    2. If >=2 predictions match to DIFFERENT GT frames, compute trajectory reward
    3. Add trajectory smoothness penalty (self-supervised, no GT needed)
    
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

            # Parse direction from <motion> text once; this code path currently uses
            # the latest motion description in the completion.
            motion_tags = re.findall(r'<motion>([^<]+)</motion>', think_content)
            pred_direction_text = parse_motion_direction(motion_tags[-1]) if motion_tags else None

            for _, obj_matches in grouped.items():
                if len(obj_matches) < 2:
                    continue

                unique_frames = set(m['frame_idx'] for m in obj_matches)
                if len(unique_frames) < 2:
                    continue

                obj_matches = sorted(obj_matches, key=lambda x: x['pred_time'])

                # 1. GT direction for this same-object trajectory
                gt_direction_text = compute_direction_from_bboxes(
                    obj_matches[0]['gt_bbox'], obj_matches[-1]['gt_bbox']
                )

                # 2. Direction agreement
                if pred_direction_text and gt_direction_text:
                    direction_score = 1.0 if pred_direction_text == gt_direction_text else 0.0
                else:
                    direction_score = 0.0

                # 3. Speed agreement
                bbox1 = obj_matches[0]['pred_bbox']
                bbox2 = obj_matches[-1]['pred_bbox']
                pred_cx1 = (bbox1[0] + bbox1[2]) / 2
                pred_cy1 = (bbox1[1] + bbox1[3]) / 2
                pred_cx2 = (bbox2[0] + bbox2[2]) / 2
                pred_cy2 = (bbox2[1] + bbox2[3]) / 2
                pred_displacement = np.sqrt((pred_cx2 - pred_cx1) ** 2 + (pred_cy2 - pred_cy1) ** 2)
                pred_time_diff = obj_matches[-1]['pred_time'] - obj_matches[0]['pred_time']

                gt_bbox1 = obj_matches[0]['gt_bbox']
                gt_bbox2 = obj_matches[-1]['gt_bbox']
                gt_cx1 = (gt_bbox1[0] + gt_bbox1[2]) / 2
                gt_cy1 = (gt_bbox1[1] + gt_bbox1[3]) / 2
                gt_cx2 = (gt_bbox2[0] + gt_bbox2[2]) / 2
                gt_cy2 = (gt_bbox2[1] + gt_bbox2[3]) / 2
                gt_displacement = np.sqrt((gt_cx2 - gt_cx1) ** 2 + (gt_cy2 - gt_cy1) ** 2)
                gt_time_diff = obj_matches[-1]['gt_time'] - obj_matches[0]['gt_time']

                if pred_time_diff > 0 and gt_time_diff > 0:
                    pred_speed = pred_displacement / pred_time_diff
                    gt_speed = gt_displacement / gt_time_diff
                    # Simple speed match: within 50% of GT speed
                    if gt_speed > 0:
                        speed_ratio = min(pred_speed, gt_speed) / max(pred_speed, gt_speed)
                        speed_score = max(0.0, speed_ratio)  # [0, 1]
                    else:
                        speed_score = 1.0 if pred_speed < 5.0 else 0.0
                else:
                    speed_score = 0.0

                object_reward = 0.5 * direction_score + 0.5 * speed_score
                per_object_rewards.append(max(0.0, min(1.0, float(object_reward))))

            if per_object_rewards:
                #we can use mean reward over valid object trajectories? 
                motion_rewards.append(float(np.mean(per_object_rewards)))
            else:
                motion_rewards.append(0.1 if len(matched_predictions) >= 2 else 0.0)
        
        except Exception as e:
            motion_rewards.append(0.0)
        
        idx += 1
    
    return motion_rewards
