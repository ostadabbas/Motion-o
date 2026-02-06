#!/usr/bin/env python3
"""
Visualize the CLEAN dataset frame-by-frame.
Each frame shows ALL GT bboxes that should be visible at that timestamp.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from PIL import Image


def draw_frame_with_bboxes(frame_array, gt_steps, frame_idx, total_frames, fps):
    """
    Draw a single frame with all GT bboxes that match the timestamp.
    
    Args:
        frame_array: numpy array (H, W, 3)
        gt_steps: List of GT steps with {t_s, t_e, bbox, caption}
        frame_idx: Current frame index
        total_frames: Total frames in video
        fps: Frames per second
    
    Returns:
        Annotated frame (H, W, 3)
    """
    # Convert to BGR for OpenCV
    frame = cv2.cvtColor(np.array(frame_array), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    
    # Calculate current timestamp
    duration = total_frames / fps
    current_time = (frame_idx / total_frames) * duration
    
    # Draw all bboxes active at this timestamp
    num_active = 0
    for step in gt_steps:
        t_s = step.get('t_s', 0)
        t_e = step.get('t_e', duration)
        
        # Check if this step is active at current timestamp
        if t_s <= current_time <= t_e:
            bbox = step.get('bbox', None)
            caption = step.get('caption', '')
            
            if bbox and len(bbox) == 4:
                # Convert normalized [0,1] to pixel coordinates
                x1 = int(bbox[0] * w)
                y1 = int(bbox[1] * h)
                x2 = int(bbox[2] * w)
                y2 = int(bbox[3] * h)
                
                # Draw bbox
                color = (0, 255, 0)  # Green
                thickness = 3
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw caption above bbox
                text = f"{caption[:40]}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                text_thickness = 2
                
                # Get text size for background
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
                
                # Draw text background
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, text, (x1, y1 - 5), font, font_scale, (0, 255, 0), text_thickness)
                
                num_active += 1
    
    # Add frame info at top
    info = f"Frame {frame_idx}/{total_frames} | Time: {current_time:.2f}s | Active bboxes: {num_active}"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame


def visualize_sample(sample, output_dir):
    """Visualize all frames for a single sample."""
    video_id = sample['video_id']
    frames = sample['frames']
    gt_steps = sample['gt_evidence_steps']
    
    print(f"\n{'='*80}")
    print(f"VIDEO: {video_id}")
    print(f"{'='*80}")
    print(f"Total frames: {len(frames)}")
    print(f"GT steps: {len(gt_steps)}")
    
    for i, step in enumerate(gt_steps):
        print(f"  Step {i}: [{step['t_s']:.2f}s - {step['t_e']:.2f}s] {step['caption'][:60]}")
        print(f"    Bbox: {step.get('bbox', 'None')}")
    
    # Create output directory for this video
    video_dir = output_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Estimate FPS (assume uniform frame sampling)
    if gt_steps:
        duration = max(step['t_e'] for step in gt_steps)
        fps = len(frames) / duration if duration > 0 else 30
    else:
        fps = 30
    
    print(f"Estimated FPS: {fps:.2f}")
    
    # Draw each frame
    for frame_idx, frame_data in enumerate(frames):
        frame_array = np.array(frame_data, dtype=np.uint8)
        
        # Draw frame with GT bboxes
        annotated = draw_frame_with_bboxes(
            frame_array, gt_steps, frame_idx, len(frames), fps
        )
        
        # Save
        output_path = video_dir / f"frame_{frame_idx:03d}.jpg"
        cv2.imwrite(str(output_path), annotated)
        print(f"  Saved: {output_path}")
    
    print(f"\n✓ Visualization saved to: {video_dir}/")


def main():
    # Load dataset
    dataset_path = Path("/mnt/data/plm_stc/preprocessed_test_clean/train")
    output_dir = Path("/home/bi.ga/Workspace/vlmm-mcot/clean_dataset_visualization")
    
    print("Loading clean dataset...")
    dataset = load_from_disk(str(dataset_path))
    print(f"✓ Loaded {len(dataset)} samples\n")
    
    # Visualize each sample
    for idx in range(len(dataset)):
        sample = dataset[idx]
        visualize_sample(sample, output_dir)
    
    print(f"\n{'='*80}")
    print(f"ALL DONE!")
    print(f"{'='*80}")
    print(f"Visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
