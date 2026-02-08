#!/usr/bin/env python3
"""
Visualize the ACTUAL dataset we have - no assumptions, just show reality.
"""

import sys
import numpy as np
from pathlib import Path
import cv2
from datasets import load_from_disk

print("=" * 80)
print("VISUALIZING ACTUAL DATASET - NO ASSUMPTIONS")
print("=" * 80)
print()

# Load the actual dataset
dataset_path = '/mnt/data/plm_stc/preprocessed_test/train'
print(f"Loading dataset from: {dataset_path}")
dataset = load_from_disk(dataset_path)
print(f"Dataset size: {len(dataset)} samples\n")

# Get first sample
sample = dataset[0]

print("Sample keys:", list(sample.keys()))
print()

# Check frames
frames = sample.get('frames', [])
print(f"Frames: {len(frames)}")

if frames:
    if isinstance(frames[0], list):
        frame_array = np.array(frames[0], dtype=np.uint8)
    else:
        frame_array = frames[0]
    
    H, W = frame_array.shape[:2]
    print(f"First frame shape: {frame_array.shape}")
    print(f"Dimensions: width={W}, height={H}")
    print()

# Check GT evidence
gt_steps = sample.get('gt_evidence_steps', [])
print(f"GT evidence steps: {len(gt_steps)}")
print()

# Examine first 3 GT steps in detail
for i in range(min(3, len(gt_steps))):
    step = gt_steps[i]
    print(f"--- GT Step {i} ---")
    print(f"  Time: {step.get('t_s', 'N/A'):.3f}s - {step.get('t_e', 'N/A'):.3f}s")
    print(f"  Caption: {step.get('caption', 'N/A')}")
    
    # Check bbox format
    if 'bbox' in step:
        print(f"  Format: 'bbox' (single)")
        print(f"  Bbox: {step['bbox']}")
    elif 'bboxes' in step:
        bboxes = step['bboxes']
        print(f"  Format: 'bboxes' (per-frame)")
        print(f"  Number of frames: {len(bboxes)}")
        if bboxes:
            print(f"  First frame bboxes: {bboxes[0]}")
            if len(bboxes) > 1:
                print(f"  Second frame bboxes: {bboxes[1]}")
    print()

# Now visualize first 3 frames with GT bboxes
print("=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)
print()

output_dir = Path("./debug_actual_dataset")
output_dir.mkdir(exist_ok=True)

for frame_idx in range(min(3, len(frames))):
    print(f"Processing frame {frame_idx}...")
    
    # Get frame
    if isinstance(frames[frame_idx], list):
        frame = np.array(frames[frame_idx], dtype=np.uint8)
    else:
        frame = frames[frame_idx]
    
    H, W = frame.shape[:2]
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Draw ALL GT bboxes from ALL steps
    bbox_count = 0
    for step_idx, step in enumerate(gt_steps):
        caption = step.get('caption', '')
        
        # Handle different formats
        if 'bbox' in step and step['bbox']:
            # Single bbox format (normalized or pixels?)
            bbox = step['bbox']
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                
                # Check if normalized (all values < 2.0) or pixels
                if all(c < 2.0 for c in bbox):
                    # Normalized - convert to pixels
                    px1, py1 = int(x1 * W), int(y1 * H)
                    px2, py2 = int(x2 * W), int(y2 * H)
                else:
                    # Already pixels
                    px1, py1, px2, py2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw
                color = (0, 255, 0) if step_idx % 3 == 0 else (255, 0, 0) if step_idx % 3 == 1 else (0, 165, 255)
                cv2.rectangle(frame_bgr, (px1, py1), (px2, py2), color, 3)
                
                # Draw caption
                caption_short = caption[:30] + "..." if len(caption) > 30 else caption
                cv2.putText(frame_bgr, f"S{step_idx}: {caption_short}", (px1, max(py1-10, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                bbox_count += 1
                
                print(f"  Step {step_idx}: bbox={bbox} → pixels=({px1},{py1},{px2},{py2})")
                print(f"    Caption: {caption}")
        
        elif 'bboxes' in step:
            # Per-frame format
            bboxes_per_frame = step['bboxes']
            if frame_idx < len(bboxes_per_frame):
                frame_bboxes = bboxes_per_frame[frame_idx]
                
                for bbox_in_frame in frame_bboxes:
                    if len(bbox_in_frame) == 4:
                        x1, y1, x2, y2 = bbox_in_frame
                        
                        # Check if normalized or pixels
                        if all(c < 2.0 for c in bbox_in_frame):
                            px1, py1 = int(x1 * W), int(y1 * H)
                            px2, py2 = int(x2 * W), int(y2 * H)
                        else:
                            px1, py1, px2, py2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Draw with different colors per step
                        color = (0, 255, 0) if step_idx % 3 == 0 else (255, 0, 0) if step_idx % 3 == 1 else (0, 165, 255)
                        cv2.rectangle(frame_bgr, (px1, py1), (px2, py2), color, 3)
                        
                        # Draw caption
                        caption_short = caption[:30] + "..." if len(caption) > 30 else caption
                        cv2.putText(frame_bgr, f"S{step_idx}: {caption_short}", (px1, max(py1-10, 20)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        bbox_count += 1
                        
                        print(f"  Step {step_idx}, Frame {frame_idx}: bbox={bbox_in_frame} → px=({px1},{py1},{px2},{py2})")
                        print(f"    Caption: {caption}")
    
    # Add info
    cv2.putText(frame_bgr, f"Frame {frame_idx} | {W}x{H} | {bbox_count} bboxes", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Save
    output_file = output_dir / f"actual_frame_{frame_idx}.jpg"
    cv2.imwrite(str(output_file), frame_bgr)
    print(f"  Saved: {output_file}")
    print()

print("=" * 80)
print(f"✓ Visualization complete! Check {output_dir}/")
print("=" * 80)
