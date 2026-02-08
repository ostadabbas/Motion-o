#!/usr/bin/env python3
"""
Visualize dataset WITH CORRECT TIMESTAMP MATCHING.
Only show bboxes from steps that overlap with each frame's timestamp.
"""

import sys
import numpy as np
from pathlib import Path
import cv2
from datasets import load_from_disk

print("=" * 80)
print("VISUALIZING WITH TIMESTAMP MATCHING")
print("=" * 80)
print()

# Load dataset
dataset = load_from_disk('/mnt/data/plm_stc/preprocessed_test/train')
sample = dataset[0]

frames = sample.get('frames', [])
gt_steps = sample.get('gt_evidence_steps', [])
fps = sample.get('fps', 30.0)

print(f"FPS: {fps}")
print(f"Frames: {len(frames)}")
print(f"GT steps: {len(gt_steps)}")
print()

# Calculate frame timestamps
# Assuming frames are sampled uniformly from the video
# The model gets 8 frames, and the video has GT steps spanning 0.0s to ~3.75s
video_duration = max(step.get('t_e', 0) for step in gt_steps)
frame_times = np.linspace(0, video_duration, len(frames))

print("Frame timestamps:")
for i, t in enumerate(frame_times):
    print(f"  Frame {i}: {t:.3f}s")
print()

print("GT Step time intervals:")
for i, step in enumerate(gt_steps):
    t_s = step.get('t_s', 0)
    t_e = step.get('t_e', 0)
    caption = step.get('caption', '')
    print(f"  Step {i}: {t_s:.3f}s - {t_e:.3f}s | {caption}")
print()

# Visualize
output_dir = Path("./debug_timestamp_matched")
output_dir.mkdir(exist_ok=True)

for frame_idx in range(min(len(frames), 8)):
    frame_time = frame_times[frame_idx]
    
    print(f"=" * 80)
    print(f"FRAME {frame_idx} at t={frame_time:.3f}s")
    print(f"=" * 80)
    
    # Get frame
    if isinstance(frames[frame_idx], list):
        frame = np.array(frames[frame_idx], dtype=np.uint8)
    else:
        frame = frames[frame_idx]
    
    H, W = frame.shape[:2]
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Find which steps are active at this frame's timestamp
    bbox_count = 0
    for step_idx, step in enumerate(gt_steps):
        t_s = step.get('t_s', 0)
        t_e = step.get('t_e', 0)
        caption = step.get('caption', '')
        
        # Check if this frame's time overlaps with this step's interval
        if t_s <= frame_time <= t_e:
            print(f"  ✓ Step {step_idx} [{t_s:.3f}s-{t_e:.3f}s]: ACTIVE")
            print(f"    Caption: {caption}")
            
            # This step is active - draw its bbox
            bboxes_per_frame = step.get('bboxes', [])
            
            # Find which frame within this step corresponds to our frame time
            # Calculate relative position within the step
            if len(bboxes_per_frame) > 0:
                step_duration = t_e - t_s
                time_into_step = frame_time - t_s
                relative_pos = time_into_step / step_duration if step_duration > 0 else 0
                
                # Map to frame index within this step's bboxes
                step_frame_idx = int(relative_pos * len(bboxes_per_frame))
                step_frame_idx = min(step_frame_idx, len(bboxes_per_frame) - 1)
                
                print(f"    Using step frame {step_frame_idx}/{len(bboxes_per_frame)}")
                
                if step_frame_idx < len(bboxes_per_frame):
                    frame_bboxes = bboxes_per_frame[step_frame_idx]
                    
                    for bbox in frame_bboxes:
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            px1, py1, px2, py2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Draw with different colors
                            color = (0, 255, 0) if step_idx % 3 == 0 else (255, 0, 0) if step_idx % 3 == 1 else (0, 165, 255)
                            cv2.rectangle(frame_bgr, (px1, py1), (px2, py2), color, 4)
                            
                            # Caption
                            caption_short = caption[:25] + "..." if len(caption) > 25 else caption
                            cv2.putText(frame_bgr, f"S{step_idx}: {caption_short}", (px1, max(py1-10, 20)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                            bbox_count += 1
                            print(f"    Bbox: ({px1},{py1}) to ({px2},{py2})")
        else:
            print(f"  ✗ Step {step_idx} [{t_s:.3f}s-{t_e:.3f}s]: inactive (outside time range)")
    
    # Add frame info
    cv2.putText(frame_bgr, f"Frame {frame_idx} @ t={frame_time:.3f}s | {bbox_count} active", (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Save
    output_file = output_dir / f"frame_{frame_idx}_t{frame_time:.3f}s.jpg"
    cv2.imwrite(str(output_file), frame_bgr)
    print(f"  Saved: {output_file}")
    print()

print("=" * 80)
print(f"✓ Done! Check {output_dir}/")
print("=" * 80)
