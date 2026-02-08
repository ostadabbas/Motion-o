#!/usr/bin/env python3
"""
Motion Chain of Thought (MCoT) Data Augmentation Script.

Preprocesses STGR JSON files to add <motion> tags to reasoning_process fields.
Computes motion from existing bounding box trajectories without modifying 
original data files.
"""

import json
import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.motion_text import generate_motion_text


def group_boxes_by_object(key_items: Dict, key_frames: List[Dict]) -> Dict[str, List[Tuple[List[float], float]]]:
    """
    Group bounding boxes by object name across timestamps.
    
    Args:
        key_items: Dict mapping frame_idx to {object_name: [bbox]}
        key_frames: List of dicts with {idx, time, path}
        
    Returns:
        Dict mapping object_name to list of (bbox, timestamp) tuples
    """
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
                # Take first bbox for each object at each timestamp
                bbox = bboxes[0]
                tracked_objects[object_name].append((bbox, timestamp))
    
    # Sort trajectories by timestamp
    for object_name in tracked_objects:
        tracked_objects[object_name].sort(key=lambda x: x[1])
    
    return dict(tracked_objects)


def find_last_object_mention(reasoning_process: str, object_name: str) -> Optional[int]:
    """
    Find the position after the last temporal-spatial mention of an object.
    
    Args:
        reasoning_process: The reasoning text
        object_name: Name of the object to search for
        
    Returns:
        Position index after the last mention, or None if not found
    """
    # Pattern: <obj>object_name</obj>...<t>time</t>s
    pattern = rf"<obj>{re.escape(object_name)}</obj>.*?<t>[\d.]+</t>s"
    
    matches = list(re.finditer(pattern, reasoning_process, re.DOTALL))
    
    if not matches:
        return None
    
    # Return position after the last match
    last_match = matches[-1]
    return last_match.end()


def insert_motion_tag(reasoning_process: str, 
                     object_name: str, 
                     motion_text: str) -> str:
    """
    Insert <motion> tag after the last mention of an object in reasoning process.
    
    Args:
        reasoning_process: Original reasoning text
        object_name: Name of the object
        motion_text: Motion description to insert
        
    Returns:
        Augmented reasoning text with <motion> tag
    """
    pos = find_last_object_mention(reasoning_process, object_name)
    
    if pos is None:
        # Object not found in reasoning, don't add motion tag
        return reasoning_process
    
    # Insert motion tag after the last mention
    motion_tag = f"<motion>{motion_text}</motion>"
    augmented = reasoning_process[:pos] + motion_tag + reasoning_process[pos:]
    
    return augmented


def augment_sample(sample: Dict, fps: float = 30.0) -> Dict:
    """
    Augment a single sample with motion tags.
    
    Args:
        sample: Sample dict from STGR JSON
        fps: Video frames per second
        
    Returns:
        Augmented sample dict
    """
    # Only process video tasks with temporal-spatial grounding
    task = sample.get('task', '')
    if task not in ["temporal-spatial free-form QA", "General video QA Free-form", "General video QA MCQ"]:
        return sample
    
    # Check if we have tracking data
    key_items = sample.get('key_items', {})
    key_frames = sample.get('key_frames', [])
    reasoning_process = sample.get('reasoning_process', '')
    
    if not key_items or not key_frames or not reasoning_process:
        return sample
    
    # Group boxes by object
    tracked_objects = group_boxes_by_object(key_items, key_frames)
    
    if not tracked_objects:
        return sample
    
    # Generate motion text for each tracked object
    augmented_reasoning = reasoning_process
    
    for object_name, trajectory_data in tracked_objects.items():
        # Separate bboxes and timestamps
        bboxes = [bbox for bbox, _ in trajectory_data]
        timestamps = [ts for _, ts in trajectory_data]
        
        # Generate motion text
        motion_text = generate_motion_text(bboxes, timestamps, fps)
        
        # Insert motion tag into reasoning
        augmented_reasoning = insert_motion_tag(
            augmented_reasoning,
            object_name,
            motion_text
        )
    
    # Create augmented sample
    augmented_sample = sample.copy()
    augmented_sample['reasoning_process'] = augmented_reasoning
    
    return augmented_sample


def augment_dataset(input_path: str, 
                   output_path: str, 
                   max_samples: Optional[int] = None,
                   fps: float = 30.0):
    """
    Augment entire STGR dataset with motion tags.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        max_samples: Maximum number of samples to process (None for all)
        fps: Video frames per second
    """
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    total_samples = len(data)
    samples_to_process = min(max_samples, total_samples) if max_samples else total_samples
    
    print(f"Processing {samples_to_process} samples (out of {total_samples})...")
    
    augmented_data = []
    augmented_count = 0
    
    for i, sample in enumerate(data[:samples_to_process]):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{samples_to_process} samples...")
        
        augmented_sample = augment_sample(sample, fps)
        augmented_data.append(augmented_sample)
        
        # Check if sample was actually augmented
        if augmented_sample.get('reasoning_process') != sample.get('reasoning_process'):
            augmented_count += 1
    
    print(f"\nAugmentation complete!")
    print(f"  Total samples: {samples_to_process}")
    print(f"  Augmented samples: {augmented_count}")
    print(f"  Unchanged samples: {samples_to_process - augmented_count}")
    
    # Save augmented data
    print(f"\nSaving augmented data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(augmented_data, f, indent=2)
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Augment STGR dataset with Motion Chain of Thought (MCoT) tags"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/mnt/data/stgr/json_data/STGR-SFT.json',
        help='Input STGR JSON file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (default: auto-generated)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of samples to process (default: all)'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='-motion',
        help='Suffix to add to output filename (default: -motion)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Video frames per second (default: 30.0)'
    )
    parser.add_argument(
        '--inspect',
        type=int,
        default=0,
        help='Print first N augmented samples for inspection'
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        input_path = Path(args.input)
        output_name = input_path.stem + args.output_suffix + input_path.suffix
        output_path = input_path.parent / output_name
    else:
        output_path = Path(args.output)
    
    # Augment dataset
    augment_dataset(
        args.input,
        str(output_path),
        max_samples=args.samples,
        fps=args.fps
    )
    
    # Optionally print samples for inspection
    if args.inspect > 0:
        print(f"\n{'='*80}")
        print(f"Inspecting first {args.inspect} augmented samples:")
        print('='*80)
        
        with open(output_path, 'r') as f:
            augmented_data = json.load(f)
        
        for i, sample in enumerate(augmented_data[:args.inspect]):
            print(f"\n--- Sample {i+1} ---")
            print(f"ID: {sample.get('id', 'N/A')}")
            print(f"Task: {sample.get('task', 'N/A')}")
            print(f"Question: {sample.get('question', 'N/A')[:100]}...")
            print(f"\nReasoning Process:")
            print(sample.get('reasoning_process', 'N/A')[:500])
            if len(sample.get('reasoning_process', '')) > 500:
                print("...")
            print()


if __name__ == '__main__':
    main()
