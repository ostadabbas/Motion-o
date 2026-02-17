#!/usr/bin/env python3
"""
Simplified Motion Chain of Thought (MCoT) Data Augmentation Script.

No external dependencies beyond json - computes motion inline.
"""

import json
import argparse
import re
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


def compute_centroid(bbox: List[float]) -> Tuple[float, float]:
    """Compute centroid of a bounding box."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)


def compute_direction_speed_acceleration(bboxes: List[List[float]], 
                                         timestamps: List[float]) -> Tuple[str, float, float]:
    """
    Compute dominant direction, average speed, and acceleration from bbox trajectory.
    
    Returns:
        (direction_str, avg_speed, acceleration)
    """
    if len(bboxes) < 2:
        return ("stationary", 0.0, 0.0)
    
    # Compute centroids
    centroids = [compute_centroid(bbox) for bbox in bboxes]
    
    # Compute total displacement
    total_dx = 0.0
    total_dy = 0.0
    total_distance = 0.0
    total_time = 0.0
    
    # Also track instantaneous speeds for acceleration
    speeds = []
    
    for i in range(len(centroids) - 1):
        cx1, cy1 = centroids[i]
        cx2, cy2 = centroids[i + 1]
        
        dx = cx2 - cx1
        dy = cy2 - cy1
        distance = math.sqrt(dx**2 + dy**2)
        dt = timestamps[i + 1] - timestamps[i]
        
        total_dx += dx
        total_dy += dy
        total_distance += distance
        total_time += dt
        
        # Instantaneous speed
        if dt > 0:
            speeds.append(distance / dt)
    
    # Check if stationary
    if total_distance < 0.01:
        return ("stationary", 0.0, 0.0)
    
    # Average displacement direction
    avg_dx = total_dx / (len(centroids) - 1)
    avg_dy = total_dy / (len(centroids) - 1)
    
    # Determine direction
    abs_dx = abs(avg_dx)
    abs_dy = abs(avg_dy)
    
    if abs_dx < 0.01 and abs_dy < 0.01:
        direction = "stationary"
    elif abs_dx > 0.02 and abs_dy > 0.02:
        # Diagonal
        h_dir = "left" if avg_dx < 0 else "right"
        v_dir = "up" if avg_dy < 0 else "down"
        direction = f"{v_dir}-{h_dir}"
    elif abs_dx > abs_dy:
        direction = "leftward" if avg_dx < 0 else "rightward"
    else:
        direction = "upward" if avg_dy < 0 else "downward"
    
    # Compute average speed
    avg_speed = total_distance / total_time if total_time > 0 else 0.0
    
    # Compute acceleration (change in speed over time)
    acceleration = 0.0
    if len(speeds) >= 2:
        # Linear regression on speed over time to get acceleration
        # Simple approach: (final_speed - initial_speed) / total_time
        speed_change = speeds[-1] - speeds[0]
        time_span = sum(timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1))
        if time_span > 0:
            acceleration = speed_change / time_span
    
    return (direction, avg_speed, acceleration)


def compute_trajectory_jitter(bboxes: List[List[float]], 
                              timestamps: List[float]) -> float:
    """
    Compute trajectory jitter/noise as standard deviation of direction changes.
    Lower values = smoother trajectory.
    
    Returns:
        jitter_score (0.0 = perfectly smooth, higher = more erratic)
    """
    if len(bboxes) < 3:
        return 0.0
    
    centroids = [compute_centroid(bbox) for bbox in bboxes]
    
    # Compute direction changes (angular deviations)
    direction_changes = []
    for i in range(len(centroids) - 2):
        cx1, cy1 = centroids[i]
        cx2, cy2 = centroids[i + 1]
        cx3, cy3 = centroids[i + 2]
        
        # Vectors
        v1x, v1y = cx2 - cx1, cy2 - cy1
        v2x, v2y = cx3 - cx2, cy3 - cy2
        
        # Magnitude
        mag1 = math.sqrt(v1x**2 + v1y**2)
        mag2 = math.sqrt(v2x**2 + v2y**2)
        
        if mag1 > 0.01 and mag2 > 0.01:
            # Cosine similarity
            cos_sim = (v1x * v2x + v1y * v2y) / (mag1 * mag2)
            cos_sim = max(-1.0, min(1.0, cos_sim))
            angle_change = math.acos(cos_sim)
            direction_changes.append(angle_change)
    
    if not direction_changes:
        return 0.0
    
    # Standard deviation of direction changes
    mean_change = sum(direction_changes) / len(direction_changes)
    variance = sum((x - mean_change) ** 2 for x in direction_changes) / len(direction_changes)
    jitter = math.sqrt(variance)
    
    return jitter


def generate_motion_text(bboxes: List[List[float]], 
                         timestamps: List[float]) -> str:
    """Generate natural language motion description with direction, speed, and acceleration."""
    if not bboxes or len(bboxes) == 0:
        return "no tracking data"
    
    if len(bboxes) == 1:
        return "stationary (single frame)"
    
    # Compute direction, speed, and acceleration
    direction, avg_speed, acceleration = compute_direction_speed_acceleration(bboxes, timestamps)
    
    if direction == "stationary":
        return "stationary (no significant motion)"
    
    # Format motion text with numerical metrics
    speed_str = f"{avg_speed:.3f}"
    accel_str = f"{acceleration:+.3f}"  # + prefix for positive, - for negative
    
    motion_text = f"{direction} motion (speed: {speed_str} units/s, accel: {accel_str} units/s²)"
    
    return motion_text


def group_boxes_by_object(key_items: Dict, key_frames: List[Dict]) -> Dict[str, List[Tuple[List[float], float]]]:
    """Group bounding boxes by object name across timestamps."""
    tracked_objects = defaultdict(list)
    
    # Create mapping from frame idx to timestamp
    idx_to_time = {}
    for frame in key_frames:
        idx_to_time[str(frame['idx'])] = frame['time']
    
    # Group boxes by object
    for frame_idx, objects in key_items.items():
        timestamp = idx_to_time.get(frame_idx)
        if timestamp is None:
            continue
        
        for object_name, bboxes in objects.items():
            if bboxes and len(bboxes) > 0:
                bbox = bboxes[0]
                tracked_objects[object_name].append((bbox, timestamp))
    
    # Sort trajectories by timestamp
    for object_name in tracked_objects:
        tracked_objects[object_name].sort(key=lambda x: x[1])
    
    return dict(tracked_objects)


def find_last_object_mention(reasoning_process: str, object_name: str) -> Optional[int]:
    """Find the position after the last temporal-spatial mention of an object."""
    pattern = rf"<obj>{re.escape(object_name)}</obj>.*?<t>[\d.]+</t>s"
    matches = list(re.finditer(pattern, reasoning_process, re.DOTALL))
    
    if not matches:
        return None
    
    last_match = matches[-1]
    return last_match.end()


def insert_motion_tag(reasoning_process: str, object_name: str, motion_text: str) -> str:
    """Insert <motion> tag after the last mention of an object."""
    pos = find_last_object_mention(reasoning_process, object_name)
    
    if pos is None:
        return reasoning_process
    
    motion_tag = f"<motion>{motion_text}</motion>"
    augmented = reasoning_process[:pos] + motion_tag + reasoning_process[pos:]
    
    return augmented


def augment_sample(sample: Dict) -> Dict:
    """Augment a single sample with motion tags."""
    task = sample.get('task', '')
    if task not in ["temporal-spatial free-form QA", "General video QA Free-form", "General video QA MCQ"]:
        return sample
    
    key_items = sample.get('key_items', {})
    key_frames = sample.get('key_frames', [])
    reasoning_process = sample.get('reasoning_process', '')
    
    if not key_items or not key_frames or not reasoning_process:
        return sample
    
    tracked_objects = group_boxes_by_object(key_items, key_frames)
    
    if not tracked_objects:
        return sample
    
    augmented_reasoning = reasoning_process
    
    for object_name, trajectory_data in tracked_objects.items():
        bboxes = [bbox for bbox, _ in trajectory_data]
        timestamps = [ts for _, ts in trajectory_data]
        
        # CRITICAL FIX: Only add motion tags for multi-frame trajectories
        # Single-frame objects don't provide motion signal for training
        if len(bboxes) >= 2:
            motion_text = generate_motion_text(bboxes, timestamps)
            augmented_reasoning = insert_motion_tag(augmented_reasoning, object_name, motion_text)
    
    augmented_sample = sample.copy()
    augmented_sample['reasoning_process'] = augmented_reasoning
    
    return augmented_sample


def main():
    parser = argparse.ArgumentParser(description="Augment STGR dataset with MCoT tags")
    parser.add_argument('--input', type=str, default='/scratch/bai.xiang/Open-o3-Video/json_data/STGR-SFT.json')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--output-suffix', type=str, default='-motion')
    parser.add_argument('--inspect', type=int, default=0)
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        input_path = Path(args.input)
        output_name = input_path.stem + args.output_suffix + input_path.suffix
        output_path = input_path.parent / output_name
    else:
        output_path = Path(args.output)
    
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    total_samples = len(data)
    samples_to_process = min(args.samples, total_samples) if args.samples else total_samples
    
    print(f"Processing {samples_to_process} samples (out of {total_samples})...")
    
    augmented_data = []
    augmented_count = 0
    
    for i, sample in enumerate(data[:samples_to_process]):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{samples_to_process} samples...")
        
        augmented_sample = augment_sample(sample)
        augmented_data.append(augmented_sample)
        
        if augmented_sample.get('reasoning_process') != sample.get('reasoning_process'):
            augmented_count += 1
    
    print(f"\nAugmentation complete!")
    print(f"  Total samples: {samples_to_process}")
    print(f"  Augmented samples: {augmented_count}")
    print(f"  Unchanged samples: {samples_to_process - augmented_count}")
    
    print(f"\nSaving augmented data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(augmented_data, f, indent=2)
    
    print("Done!")
    
    # Inspection
    if args.inspect > 0:
        print(f"\n{'='*80}")
        print(f"Inspecting first {args.inspect} augmented samples:")
        print('='*80)
        
        for i, sample in enumerate(augmented_data[:args.inspect]):
            print(f"\n--- Sample {i+1} ---")
            print(f"ID: {sample.get('id', 'N/A')}")
            print(f"Task: {sample.get('task', 'N/A')}")
            print(f"Question: {sample.get('question', 'N/A')[:100]}...")
            print(f"\nReasoning Process:")
            reasoning = sample.get('reasoning_process', 'N/A')
            print(reasoning[:800])
            if len(reasoning) > 800:
                print("...")
            print()


if __name__ == '__main__':
    main()
