#!/usr/bin/env python3
"""
Final comprehensive ball tracking test.

Tests VLM's ability to:
1. Detect and localize the ball in individual frames
2. Track motion across frames
3. Output structured bbox coordinates

This demonstrates the baseline capability before GRPO training.
"""

import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration
)


def extract_frames_from_video(video_path: str, frame_indices: list):
    """Extract specific frames from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    metadata = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, frame_rgb))
    
    cap.release()
    return frames, metadata


def detect_ball_in_frame(frame_rgb, frame_idx, model, processor, device):
    """Run VLM detection on single frame."""
    img = Image.fromarray(frame_rgb)
    
    prompt = """Locate the red ball in this image.

If you see a red ball, provide its bounding box: <bbox>[x1,y1,x2,y2]</bbox>
If no red ball is visible, respond: "No ball visible"

Use normalized coordinates (0 to 1)."""
    
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt}
    ]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
        )
    
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output[:, input_len:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Parse bbox
    bbox = None
    patterns = [
        r'<bbox>\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]</bbox>',
        r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            bbox = [float(match.group(i)) for i in range(1, 5)]
            if bbox[0] < bbox[2] and bbox[1] < bbox[3] and all(0 <= c <= 1.1 for c in bbox):
                break
            else:
                bbox = None
    
    return bbox, response


def visualize_tracking_results(results, video_metadata, output_dir):
    """Create comprehensive visualization of tracking results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_large = font_medium = font_small = ImageFont.load_default()
    
    width = video_metadata['width']
    height = video_metadata['height']
    
    tracking_data = []
    
    print(f"\n{'='*80}")
    print("VISUALIZATION: Creating annotated frames")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results):
        frame_idx = result['frame_idx']
        frame_rgb = result['frame']
        bbox = result['bbox']
        response = result['response']
        
        img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img)
        
        if bbox:
            x1, y1, x2, y2 = bbox
            
            # Convert to pixels
            x1_px = int(x1 * width)
            y1_px = int(y1 * height)
            x2_px = int(x2 * width)
            y2_px = int(y2 * height)
            
            # Calculate center
            center_x = (x1_px + x2_px) / 2
            center_y = (y1_px + y2_px) / 2
            center_x_norm = center_x / width
            center_y_norm = center_y / height
            
            # Draw bounding box
            draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline='red', width=5)
            
            # Draw center point
            radius = 8
            draw.ellipse(
                [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                fill='red',
                outline='yellow',
                width=3
            )
            
            # Add frame number (top-left)
            draw.text((15, 15), f"Frame {frame_idx}", fill='yellow', font=font_large, 
                     stroke_width=2, stroke_fill='black')
            
            # Add bbox coordinates (top-left, below frame number)
            bbox_text = f"Bbox: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]"
            draw.text((15, 55), bbox_text, fill='white', font=font_small,
                     stroke_width=1, stroke_fill='black')
            
            # Add center position (top-left, below bbox)
            center_text = f"Center: ({center_x_norm:.3f}, {center_y_norm:.3f})"
            draw.text((15, 80), center_text, fill='white', font=font_small,
                     stroke_width=1, stroke_fill='black')
            
            # Add position description (bottom-left)
            if center_x_norm < 0.33:
                pos_desc = "LEFT"
                pos_color = 'cyan'
            elif center_x_norm < 0.67:
                pos_desc = "CENTER"
                pos_color = 'green'
            else:
                pos_desc = "RIGHT"
                pos_color = 'magenta'
            
            draw.text((15, height - 45), f"Position: {pos_desc}", 
                     fill=pos_color, font=font_medium,
                     stroke_width=2, stroke_fill='black')
            
            # Store tracking data
            tracking_data.append({
                'frame_idx': frame_idx,
                'bbox_normalized': bbox,
                'bbox_pixels': [x1_px, y1_px, x2_px, y2_px],
                'center_normalized': [center_x_norm, center_y_norm],
                'center_pixels': [int(center_x), int(center_y)],
                'position_label': pos_desc
            })
            
            status = f"✓ Ball detected at x={center_x_norm:.3f}"
        else:
            # No detection
            draw.text((15, 15), f"Frame {frame_idx}", fill='yellow', font=font_large,
                     stroke_width=2, stroke_fill='black')
            draw.text((15, 55), "No ball detected", fill='red', font=font_medium,
                     stroke_width=2, stroke_fill='black')
            
            tracking_data.append({
                'frame_idx': frame_idx,
                'bbox_normalized': None,
                'bbox_pixels': None,
                'center_normalized': None,
                'center_pixels': None,
                'position_label': 'NOT_DETECTED'
            })
            
            status = "✗ No ball detected"
        
        # Save annotated frame
        output_path = output_dir / f"frame_{frame_idx:04d}_tracked.jpg"
        img.save(output_path, quality=95)
        print(f"  Frame {frame_idx:3d}: {status} -> {output_path.name}")
    
    # Save tracking data as JSON
    json_path = output_dir / "tracking_data.json"
    with open(json_path, 'w') as f:
        json.dump({
            'video_metadata': video_metadata,
            'tracking_results': tracking_data
        }, f, indent=2)
    
    print(f"\n✓ Saved tracking data: {json_path}")
    
    return tracking_data


def analyze_motion_trajectory(tracking_data):
    """Analyze the motion trajectory from tracking data."""
    
    print(f"\n{'='*80}")
    print("MOTION ANALYSIS")
    print(f"{'='*80}\n")
    
    detected_frames = [d for d in tracking_data if d['bbox_normalized'] is not None]
    
    if not detected_frames:
        print("✗ No detections found - cannot analyze motion")
        return
    
    print(f"Detected ball in {len(detected_frames)}/{len(tracking_data)} frames\n")
    
    # Print frame-by-frame positions
    print("Frame-by-frame positions:")
    for data in tracking_data:
        frame_idx = data['frame_idx']
        if data['center_normalized']:
            x, y = data['center_normalized']
            pos = data['position_label']
            print(f"  Frame {frame_idx:3d}: x={x:.3f}, y={y:.3f} ({pos})")
        else:
            print(f"  Frame {frame_idx:3d}: NOT DETECTED")
    
    if len(detected_frames) < 2:
        print("\n⚠ Need at least 2 detections to analyze motion")
        return
    
    # Extract x-positions
    x_positions = [d['center_normalized'][0] for d in detected_frames]
    frame_indices = [d['frame_idx'] for d in detected_frames]
    
    # Motion statistics
    first_x = x_positions[0]
    last_x = x_positions[-1]
    total_displacement = last_x - first_x
    
    # Check if motion is monotonic (consistent direction)
    is_monotonic_increasing = all(x_positions[i] <= x_positions[i+1] for i in range(len(x_positions)-1))
    is_monotonic_decreasing = all(x_positions[i] >= x_positions[i+1] for i in range(len(x_positions)-1))
    
    print(f"\nMotion Statistics:")
    print(f"  Start position (frame {frame_indices[0]}): x = {first_x:.3f}")
    print(f"  End position (frame {frame_indices[-1]}):   x = {last_x:.3f}")
    print(f"  Total displacement: Δx = {total_displacement:+.3f}")
    print(f"  Monotonic motion: {is_monotonic_increasing or is_monotonic_decreasing}")
    
    if is_monotonic_increasing:
        print(f"  Direction: LEFT → RIGHT ✓")
    elif is_monotonic_decreasing:
        print(f"  Direction: RIGHT → LEFT")
    else:
        print(f"  Direction: Non-monotonic (inconsistent)")
    
    # Overall assessment
    print(f"\n{'='*80}")
    print("ASSESSMENT")
    print(f"{'='*80}\n")
    
    if len(detected_frames) == len(tracking_data):
        print("✓ DETECTION: Ball detected in all frames")
    else:
        missed = len(tracking_data) - len(detected_frames)
        print(f"~ DETECTION: Detected in {len(detected_frames)}/{len(tracking_data)} frames ({missed} missed)")
    
    if abs(total_displacement) > 0.4 and (is_monotonic_increasing or is_monotonic_decreasing):
        print("✓ TRACKING: Ball motion tracked successfully with consistent direction")
        print("✓ SPATIAL GROUNDING: VLM can output structured bbox coordinates")
    elif abs(total_displacement) > 0.2:
        print("~ TRACKING: Some motion detected but may not be fully accurate")
        print("~ SPATIAL GROUNDING: VLM has partial bbox capability")
    else:
        print("✗ TRACKING: Insufficient motion detected")
        print("✗ SPATIAL GROUNDING: VLM may be guessing bbox coordinates")
    
    print(f"\nKey takeaway for your GRPO training:")
    print(f"  Base VLM has {'GOOD' if len(detected_frames) >= len(tracking_data)*0.8 else 'LIMITED'} bbox generation capability")
    print(f"  Spatial accuracy: {'HIGH' if abs(total_displacement) > 0.4 else 'MEDIUM' if abs(total_displacement) > 0.2 else 'LOW'}")
    print(f"  GRPO training will {'refine and compose' if len(detected_frames) >= len(tracking_data)*0.8 else 'teach'} this into multi-step reasoning chains")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Final ball tracking test with clean visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--frames", type=str, default="0,48,96,144,191",
                       help="Comma-separated frame indices to test")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                       help="HuggingFace model ID")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device (cuda:0, cuda:1, etc.)")
    parser.add_argument("--output-dir", type=str, default="outputs/final_tracking",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    frame_indices = [int(x.strip()) for x in args.frames.split(",")]
    
    print("="*80)
    print("FINAL BALL TRACKING TEST")
    print("="*80)
    print(f"Video: {args.video_path}")
    print(f"Model: {args.model_id}")
    print(f"Frames: {frame_indices}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    # Check video exists
    if not Path(args.video_path).exists():
        print(f"\n✗ ERROR: Video not found: {args.video_path}")
        sys.exit(1)
    
    # Extract frames
    print("\n[1/4] Extracting frames from video...")
    frames, metadata = extract_frames_from_video(args.video_path, frame_indices)
    print(f"  Video: {metadata['width']}x{metadata['height']}, {metadata['total_frames']} total frames")
    print(f"  Extracted: {len(frames)} frames")
    
    # Load model
    print(f"\n[2/4] Loading VLM model...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    
    # Select correct model class
    if "Qwen3" in args.model_id or "qwen3" in args.model_id:
        model_class = Qwen3VLForConditionalGeneration
    elif "Qwen2.5" in args.model_id or "qwen2.5" in args.model_id or "Qwen2_5" in args.model_id:
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        model_class = Qwen2VLForConditionalGeneration
    
    model = model_class.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  ✓ Loaded: {model.__class__.__name__}")
    
    # Run detection on each frame
    print(f"\n[3/4] Running detection on each frame...")
    results = []
    
    for frame_idx, frame_rgb in frames:
        bbox, response = detect_ball_in_frame(frame_rgb, frame_idx, model, processor, args.device)
        results.append({
            'frame_idx': frame_idx,
            'frame': frame_rgb,
            'bbox': bbox,
            'response': response
        })
        
        if bbox:
            center_x = (bbox[0] + bbox[2]) / 2
            print(f"  Frame {frame_idx:3d}: ✓ Detected at x={center_x:.3f}")
        else:
            print(f"  Frame {frame_idx:3d}: ✗ No detection")
    
    # Visualize results
    print(f"\n[4/4] Creating visualizations...")
    tracking_data = visualize_tracking_results(results, metadata, args.output_dir)
    
    # Analyze motion
    analyze_motion_trajectory(tracking_data)
    
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {args.output_dir}/")
    print(f"  - Annotated frames: frame_XXXX_tracked.jpg")
    print(f"  - Tracking data: tracking_data.json")
    print(f"\nYou can now use these results to:")
    print(f"  1. Verify VLM spatial grounding capability")
    print(f"  2. Design GRPO reward functions based on bbox IoU")
    print(f"  3. Create training examples with ground truth spatial annotations")


if __name__ == "__main__":
    main()
