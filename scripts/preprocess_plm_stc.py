#!/usr/bin/env python3
"""
PLM-STC Dataset Preprocessing for Motion Reasoning.

Converts raw PLM-STC data (masklets, annotations) into training-ready format
with precomputed motion descriptors for GRPO training.

PLM-STC (PerceptionLM) dataset provides:
- 194.2K human-annotated instances
- High-fps segmentation masklets
- Temporal intervals
- Motion-focused captions
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import cv2
from tqdm import tqdm
from datasets import Dataset as HFDataset
import math


def masklets_to_bboxes(masklets: np.ndarray) -> List[List[int]]:
    """
    Convert segmentation masklets to bounding boxes per frame.
    
    Args:
        masklets: Binary mask sequence of shape (num_frames, H, W)
                 or (num_frames, num_objects, H, W)
    
    Returns:
        List of bboxes per frame: [[x1, y1, x2, y2], ...]
        Returns empty list for frames with no mask
    """
    if len(masklets.shape) == 3:
        # Single object: (num_frames, H, W)
        masklets = masklets[:, np.newaxis, :, :]  # Add object dimension
    
    num_frames, num_objects, H, W = masklets.shape
    bboxes_per_frame = []
    
    for frame_idx in range(num_frames):
        frame_bboxes = []
        for obj_idx in range(num_objects):
            mask = masklets[frame_idx, obj_idx]
            
            # Find non-zero pixels
            coords = np.where(mask > 0)
            if len(coords[0]) == 0:
                # No mask for this object in this frame
                continue
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Convert to [x1, y1, x2, y2] format
            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
            frame_bboxes.append(bbox)
        
        bboxes_per_frame.append(frame_bboxes)
    
    return bboxes_per_frame


def compute_centroid_trajectory(bboxes: List[List[int]]) -> List[Tuple[float, float]]:
    """
    Compute centroid (cx, cy) for each bbox in sequence.
    
    Args:
        bboxes: List of [x1, y1, x2, y2] bounding boxes
    
    Returns:
        List of (cx, cy) centroids
    """
    centroids = []
    for bbox in bboxes:
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centroids.append((float(cx), float(cy)))
        else:
            centroids.append((0.0, 0.0))  # Invalid bbox
    return centroids


def compute_displacement_vectors(trajectory: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Compute frame-to-frame displacement vectors from centroid trajectory.
    
    Args:
        trajectory: List of (cx, cy) centroids
    
    Returns:
        List of (dx, dy) displacement vectors between consecutive frames
    """
    if len(trajectory) < 2:
        return []
    
    displacements = []
    for i in range(len(trajectory) - 1):
        cx1, cy1 = trajectory[i]
        cx2, cy2 = trajectory[i + 1]
        dx = cx2 - cx1
        dy = cy2 - cy1
        displacements.append((float(dx), float(dy)))
    
    return displacements


def compute_velocities(displacements: List[Tuple[float, float]], fps: float) -> List[float]:
    """
    Compute velocities (speed in px/sec) from displacement vectors.
    
    Args:
        displacements: List of (dx, dy) vectors
        fps: Frames per second
    
    Returns:
        List of speeds in pixels per second
    """
    velocities = []
    for dx, dy in displacements:
        speed = math.sqrt(dx**2 + dy**2) * fps
        velocities.append(float(speed))
    return velocities


def compute_direction_angles(displacements: List[Tuple[float, float]]) -> List[float]:
    """
    Compute direction angles (in radians) from displacement vectors.
    
    Args:
        displacements: List of (dx, dy) vectors
    
    Returns:
        List of angles in radians (atan2(dy, dx))
    """
    angles = []
    for dx, dy in displacements:
        angle = math.atan2(dy, dx)
        angles.append(float(angle))
    return angles


def compute_motion_descriptors(bboxes: List[List[int]], fps: float) -> Dict:
    """
    Compute ground truth motion features from bbox sequence.
    
    Args:
        bboxes: List of [x1, y1, x2, y2] bounding boxes per frame
        fps: Frames per second
    
    Returns:
        Dictionary with motion descriptors:
        - centroid_trajectory: [(cx, cy), ...]
        - displacement_vectors: [(dx, dy), ...]
        - velocities: [speed_px_per_sec, ...]
        - direction_angles: [theta_radians, ...]
    """
    trajectory = compute_centroid_trajectory(bboxes)
    displacements = compute_displacement_vectors(trajectory)
    velocities = compute_velocities(displacements, fps)
    angles = compute_direction_angles(displacements)
    
    return {
        "centroid_trajectory": trajectory,
        "displacement_vectors": displacements,
        "velocities": velocities,
        "direction_angles": angles
    }


def extract_frames_from_video(video_path: str, 
                              start_time: float, 
                              end_time: float,
                              fps: Optional[float] = None,
                              max_frames: int = 32) -> List[np.ndarray]:
    """
    Extract frames from video within time interval.
    
    Args:
        video_path: Path to video file
        start_time: Start time in seconds
        end_time: End time in seconds
        fps: Target FPS (if None, use video's native FPS)
        max_frames: Maximum number of frames to extract
    
    Returns:
        List of frame arrays (H, W, 3) in RGB
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None:
        fps = video_fps
    
    # Calculate frame indices to extract
    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    
    # Sample uniformly if too many frames
    total_frames = end_frame - start_frame
    if total_frames > max_frames:
        frame_indices = np.linspace(start_frame, end_frame, max_frames, dtype=int)
    else:
        frame_indices = list(range(start_frame, end_frame + 1))
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames


def load_plm_stc_raw(data_dir: str, split: str = "train") -> List[Dict]:
    """
    Load PLM-STC raw data (videos, masklets, annotations).
    
    Expected directory structure:
        data_dir/
            videos/
                {video_id}.mp4
            annotations/
                {split}.json  # Contains: video_id, question, answer, evidence_steps
            masklets/
                {video_id}_{step_idx}.npy  # Masklet array (num_frames, H, W) or (num_frames, num_objects, H, W)
    
    Args:
        data_dir: Root directory of PLM-STC dataset
        split: Dataset split (train/val/test)
    
    Returns:
        List of dataset items with raw data
    """
    data_dir = Path(data_dir)
    annotations_file = data_dir / "annotations" / f"{split}.json"
    
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"Loaded {len(annotations)} annotations from {annotations_file}")
    return annotations


def preprocess_item(item: Dict, data_dir: Path, max_frames: int = 32) -> Optional[Dict]:
    """
    Preprocess a single PLM-STC item.
    
    Args:
        item: Raw annotation item with:
            - video_id: Video identifier
            - question: Motion query
            - answer: Ground truth answer
            - evidence_steps: List of evidence steps, each with:
                - t_s, t_e: Temporal interval
                - masklet_path: Relative path to masklet file
                - caption: Text description
        data_dir: Root directory of PLM-STC dataset
        max_frames: Maximum frames to extract per evidence step
    
    Returns:
        Preprocessed item ready for HuggingFace dataset, or None if error
    """
    try:
        video_id = item["video_id"]
        video_path = data_dir / "videos" / f"{video_id}.mp4"
        
        if not video_path.exists():
            print(f"Warning: Video not found: {video_path}")
            return None
        
        # Extract overall video info
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Process evidence steps
        processed_steps = []
        all_frames = []
        
        for step_idx, step in enumerate(item.get("evidence_steps", [])):
            t_s = step["t_s"]
            t_e = step["t_e"]
            caption = step.get("caption", "")
            
            # Load masklet
            masklet_rel_path = step.get("masklet_path", f"{video_id}_{step_idx}.npy")
            masklet_path = data_dir / "masklets" / masklet_rel_path
            
            if not masklet_path.exists():
                print(f"Warning: Masklet not found: {masklet_path}")
                continue
            
            masklets = np.load(str(masklet_path))
            
            # Convert masklets to bboxes
            bboxes = masklets_to_bboxes(masklets)
            
            # Compute motion descriptors for each object
            # If multiple objects, compute separately then aggregate
            if len(bboxes) == 0:
                continue
            
            # For simplicity, take first object per frame (or can track multiple)
            primary_bboxes = [frame_boxes[0] if len(frame_boxes) > 0 else [0, 0, 0, 0] 
                            for frame_boxes in bboxes]
            
            motion_desc = compute_motion_descriptors(primary_bboxes, fps)
            
            # Extract frames for this step
            step_frames = extract_frames_from_video(
                str(video_path), t_s, t_e, fps=fps, max_frames=max_frames
            )
            
            processed_step = {
                "t_s": float(t_s),
                "t_e": float(t_e),
                "bboxes": bboxes,  # All bboxes per frame
                "motion_desc": motion_desc,
                "caption": caption
            }
            processed_steps.append(processed_step)
            all_frames.extend(step_frames)
        
        if len(processed_steps) == 0:
            return None
        
        # Sample frames uniformly if too many
        if len(all_frames) > max_frames:
            indices = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int)
            all_frames = [all_frames[i] for i in indices]
        
        return {
            "video_id": video_id,
            "video_path": str(video_path),
            "frames": all_frames,
            "question": item["question"],
            "answer": item["answer"],
            "gt_evidence_steps": processed_steps,
            "fps": float(fps)
        }
    
    except Exception as e:
        print(f"Error preprocessing item {item.get('video_id', 'unknown')}: {e}")
        return None


def preprocess_and_save(input_dir: str, 
                       output_dir: str, 
                       split: str = "train",
                       max_frames: int = 32,
                       num_workers: int = 1):
    """
    Main preprocessing pipeline for PLM-STC dataset.
    
    Args:
        input_dir: Input directory with raw PLM-STC data
        output_dir: Output directory for preprocessed HuggingFace dataset
        split: Dataset split (train/val/test)
        max_frames: Maximum frames per video
        num_workers: Number of parallel workers (not implemented yet)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading PLM-STC {split} split from {input_dir}...")
    raw_items = load_plm_stc_raw(str(input_dir), split)
    
    print(f"Preprocessing {len(raw_items)} items...")
    processed_items = []
    
    for item in tqdm(raw_items, desc="Preprocessing"):
        processed = preprocess_item(item, input_dir, max_frames)
        if processed is not None:
            processed_items.append(processed)
    
    print(f"Successfully preprocessed {len(processed_items)} / {len(raw_items)} items")
    
    # Convert to HuggingFace dataset
    print("Creating HuggingFace dataset...")
    dataset = HFDataset.from_list(processed_items)
    
    # Save
    output_path = output_dir / split
    print(f"Saving dataset to {output_path}...")
    dataset.save_to_disk(str(output_path))
    
    print(f"Done! Dataset saved to {output_path}")
    print(f"Dataset info: {len(dataset)} examples")
    print(f"Example keys: {list(dataset[0].keys())}")


def main():
    import sys
    print("STARTING PREPROCESSING SCRIPT", flush=True)
    sys.stdout.flush()
    
    parser = argparse.ArgumentParser(description="Preprocess PLM-STC dataset for motion reasoning")
    parser.add_argument("input_dir", type=str, help="Input directory with raw PLM-STC data")
    parser.add_argument("output_dir", type=str, help="Output directory for preprocessed dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                       help="Dataset split to preprocess")
    parser.add_argument("--max-frames", type=int, default=32,
                       help="Maximum frames per video")
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Number of parallel workers")
    
    print("PARSING ARGS", flush=True)
    sys.stdout.flush()
    args = parser.parse_args()
    
    print(f"CALLING preprocess_and_save with input={args.input_dir}", flush=True)
    sys.stdout.flush()
    
    preprocess_and_save(
        args.input_dir,
        args.output_dir,
        split=args.split,
        max_frames=args.max_frames,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
