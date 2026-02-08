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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.motion_metrics import (
    compute_centroid_trajectory,
    compute_displacement_vectors,
    direction_cosine_similarity,
    speed_fidelity_score,
    trajectory_smoothness_penalty
)


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


def motion_trajectory_reward(completions, **kwargs):
    """
    Motion-aware trajectory reward function.
    
    Evaluates predicted object trajectories against ground truth using:
    1. Direction similarity: Cosine similarity of displacement vectors
    2. Speed fidelity: Velocity magnitude matching
    3. Trajectory smoothness: Acceleration penalty for physically implausible motion
    
    Args:
        completions: List of model completions (list of dicts with 'content')
        **kwargs: Additional arguments including:
            - task: List of task types
            - key_items: GT spatial annotations
            - key_frames: GT temporal keyframes
            - image_size: Image dimensions
            
    Returns:
        List of rewards in [0, 1] for each completion
    """
    motion_rewards = []
    idx = 0
    
    for completion in completions:
        think_match = re.search(r"<think>(.*?)</think>", completion[0]["content"], re.DOTALL)
        
        # Skip if no think section
        if not think_match:
            motion_rewards.append(0.0)
            idx += 1
            continue
        
        task = kwargs.get('task', [''])[0]
        
        # Only compute motion reward for temporal-spatial tasks
        if task not in ["temporal-spatial free-form QA", "General video QA Free-form", "General video QA MCQ"]:
            motion_rewards.append(0.0)
            idx += 1
            continue
        
        think_content = think_match.group(1)
        parsed_claims = parse_temporal_spatial_claims(think_content)
        
        # Need at least 2 temporal points for motion
        if not parsed_claims or len(parsed_claims) < 2:
            motion_rewards.append(0.0)
            idx += 1
            continue
        
        try:
            # Get GT trajectory
            gt_items = kwargs.get("key_items", [None])[idx]
            gt_frames = kwargs.get("key_frames", [None])[idx]
            image_size = kwargs.get("image_size", [(640, 480)])[idx]
            
            if gt_items is None or gt_frames is None:
                motion_rewards.append(0.0)
                idx += 1
                continue
            
            # Default FPS if not provided
            fps = 30.0
            
            # Sort claims by timestamp
            parsed_claims = sorted(parsed_claims, key=lambda x: x['timestamp'])
            
            # Extract predicted bbox trajectory
            pred_bboxes = []
            pred_times = []
            for claim in parsed_claims:
                if claim['bboxes'] and len(claim['bboxes']) > 0:
                    bbox = claim['bboxes'][0]  # Take first bbox
                    # Convert to pixel coordinates if normalized
                    if all(0 <= c <= 1 for c in bbox):
                        bbox = convert_coord_format(bbox, image_size)
                    pred_bboxes.append(bbox)
                    pred_times.append(claim['timestamp'])
            
            if len(pred_bboxes) < 2:
                motion_rewards.append(0.0)
                idx += 1
                continue
            
            # Extract GT bbox trajectory
            gt_bboxes = []
            gt_times = []
            sorted_frames = sorted(gt_frames, key=lambda x: x['time'])
            
            for frame in sorted_frames:
                frame_idx = str(frame["idx"])
                if frame_idx in gt_items and gt_items[frame_idx]:
                    obj_key = list(gt_items[frame_idx].keys())[0]
                    gt_bbox = gt_items[frame_idx][obj_key][0]
                    gt_bbox = convert_coord_format(gt_bbox, image_size)
                    gt_bboxes.append(gt_bbox)
                    gt_times.append(frame['time'])
            
            if len(gt_bboxes) < 2:
                motion_rewards.append(0.0)
                idx += 1
                continue
            
            # Compute motion metrics
            
            # 1. Direction similarity (displacement vectors)
            pred_centroids = [compute_bbox_centroid(bbox) for bbox in pred_bboxes]
            gt_centroids = [compute_bbox_centroid(bbox) for bbox in gt_bboxes]
            
            pred_displacements = [
                compute_displacement_vector(pred_centroids[i], pred_centroids[i+1])
                for i in range(len(pred_centroids) - 1)
            ]
            gt_displacements = [
                compute_displacement_vector(gt_centroids[i], gt_centroids[i+1])
                for i in range(len(gt_centroids) - 1)
            ]
            
            if not pred_displacements or not gt_displacements:
                motion_rewards.append(0.0)
                idx += 1
                continue
            
            # Match predicted and GT displacements (use min length)
            min_len = min(len(pred_displacements), len(gt_displacements))
            direction_score = compute_direction_similarity(
                pred_displacements[:min_len],
                gt_displacements[:min_len]
            )
            
            # 2. Speed fidelity (velocity magnitude)
            pred_speeds = []
            for i in range(len(pred_times) - 1):
                dt = pred_times[i+1] - pred_times[i]
                if dt > 0:
                    dx, dy = pred_displacements[i] if i < len(pred_displacements) else (0, 0)
                    speed = np.sqrt(dx**2 + dy**2) / dt
                    pred_speeds.append(speed)
            
            gt_speeds = []
            for i in range(len(gt_times) - 1):
                dt = gt_times[i+1] - gt_times[i]
                if dt > 0:
                    dx, dy = gt_displacements[i] if i < len(gt_displacements) else (0, 0)
                    speed = np.sqrt(dx**2 + dy**2) / dt
                    gt_speeds.append(speed)
            
            if pred_speeds and gt_speeds:
                avg_pred_speed = np.mean(pred_speeds)
                avg_gt_speed = np.mean(gt_speeds)
                speed_score = compute_speed_fidelity(avg_pred_speed, avg_gt_speed)
            else:
                speed_score = 0.0
            
            # 3. Trajectory smoothness (acceleration penalty)
            if len(pred_speeds) > 1:
                smoothness_score = compute_trajectory_smoothness(pred_speeds)
            else:
                smoothness_score = 0.0
            
            # Combine motion components (weighted average)
            motion_reward = (
                0.4 * direction_score +      # Direction is most important
                0.4 * speed_score +           # Speed matching
                0.2 * smoothness_score        # Smooth trajectories
            )
            
            # Clamp to [0, 1]
            motion_reward = max(0.0, min(1.0, float(motion_reward)))
            motion_rewards.append(motion_reward)
        
        except Exception as e:
            # Debug: print error but continue
            if kwargs.get('debug', False):
                print(f"Error computing motion reward for sample {idx}: {e}")
            motion_rewards.append(0.0)
        
        idx += 1
    
    return motion_rewards
