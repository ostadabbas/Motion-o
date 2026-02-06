#!/usr/bin/env python3
"""
DEBUG: Visualize ALL masklets for a specific frame to see which one is correct.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_utils


def decode_rle(rle_dict, img_height, img_width):
    """Decode RLE mask."""
    try:
        if not rle_dict or not isinstance(rle_dict, dict):
            return np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Ensure size is correct format
        if 'size' in rle_dict:
            rle_height, rle_width = rle_dict['size']
            if rle_height != img_height or rle_width != img_width:
                print(f"  Warning: RLE size {rle_dict['size']} != expected {(img_height, img_width)}")
        
        # Decode
        mask = mask_utils.decode(rle_dict)
        return mask.astype(np.uint8)
    except Exception as e:
        print(f"  Error decoding RLE: {e}")
        return np.zeros((img_height, img_width), dtype=np.uint8)


def visualize_all_masklets(video_id, frame_idx, output_dir):
    """Visualize all masklets for a specific frame."""
    sav_dir = Path('/mnt/data/plm_stc/raw/sa-v')
    
    # Load SA-V JSON
    json_path = sav_dir / f'{video_id}_manual.json'
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    available_ids = data.get('masklet_id', [])
    img_width = int(data.get('video_width', 1280))
    img_height = int(data.get('video_height', 720))
    masklets = data.get('masklet', [])
    
    print(f"Video: {video_id}")
    print(f"Frame: {frame_idx}")
    print(f"Available masklet IDs: {available_ids}")
    print(f"Dimensions: {img_width} x {img_height}")
    
    # Load video frame
    video_path = sav_dir / f'{video_id}.mp4'
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"ERROR: Could not read frame {frame_idx}")
        return
    
    # Process each object
    frame_objects = masklets[frame_idx] if frame_idx < len(masklets) else []
    
    for obj_idx in range(len(frame_objects)):
        rle_dict = frame_objects[obj_idx]
        mask = decode_rle(rle_dict, img_height, img_width)
        
        # Count non-zero pixels
        num_pixels = np.count_nonzero(mask)
        print(f"\nObject {obj_idx} (masklet_id={available_ids[obj_idx] if obj_idx < len(available_ids) else '?'}):")
        print(f"  Non-zero pixels: {num_pixels:,}")
        
        if num_pixels == 0:
            print(f"  EMPTY - skipping visualization")
            continue
        
        # Compute bbox
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            print(f"  Bbox: [{x1}, {y1}, {x2}, {y2}]")
            print(f"  Bbox size: {x2-x1} x {y2-y1}")
        
        # Overlay mask on frame
        overlay = frame.copy()
        mask_color = np.random.randint(50, 255, 3).tolist()
        overlay[mask > 0] = mask_color
        
        # Blend
        vis = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # Draw bbox
        if len(xs) > 0:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(vis, f"Object {obj_idx} ({num_pixels:,} px)", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save
        output_path = output_dir / f"{video_id}_frame{frame_idx}_obj{obj_idx}.jpg"
        cv2.imwrite(str(output_path), vis)
        print(f"  Saved: {output_path}")


def main():
    output_dir = Path("/home/bi.ga/Workspace/vlmm-mcot/debug_all_masklets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Debug the problematic videos
    problems = [
        ('sav_015834', 10),  # Frame 10 should have "person walking"
        ('sav_003127', 50),  # Frame 50 should have "man walking into mall"
        ('sav_017599', 40),  # Frame 40 should have "boy"
    ]
    
    for video_id, frame_idx in problems:
        print(f"\n{'='*80}")
        visualize_all_masklets(video_id, frame_idx, output_dir)
    
    print(f"\n{'='*80}")
    print(f"DONE! Check: {output_dir}/")


if __name__ == "__main__":
    main()
