#!/usr/bin/env python3
"""
Diagnostic script to detect VLM hallucination in bbox generation.

Analyzes:
1. Full token output to see how bbox coordinates are generated
2. Attention patterns (if available)
3. Comparison with ground truth (via classical CV)
"""

import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)


def get_ground_truth_ball_position(frame_rgb):
    """
    Get ground truth ball position using classical computer vision.
    
    Uses color thresholding to find the red ball.
    Returns bbox in normalized coordinates or None if not found.
    """
    # Convert to HSV for better color detection
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    
    # Red color range in HSV
    # Red wraps around in HSV, so we need two ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, mask
    
    # Find largest contour (should be the ball)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Filter out very small detections (noise)
    if area < 100:  # Minimum area threshold
        return None, mask
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Convert to normalized coordinates
    height, width = frame_rgb.shape[:2]
    x1_norm = x / width
    y1_norm = y / height
    x2_norm = (x + w) / width
    y2_norm = (y + h) / height
    
    bbox = [x1_norm, y1_norm, x2_norm, y2_norm]
    
    return bbox, mask


def analyze_single_frame_detailed(frame_rgb, frame_idx, model, processor, device):
    """
    Detailed analysis of model output for single frame.
    
    Returns:
        - VLM predicted bbox
        - Ground truth bbox (from CV)
        - Full token output
        - Token IDs
        - Logits (if available)
    """
    img = Image.fromarray(frame_rgb)
    height, width = frame_rgb.shape[:2]
    
    # Get ground truth
    gt_bbox, mask = get_ground_truth_ball_position(frame_rgb)
    
    print(f"\n{'='*80}")
    print(f"FRAME {frame_idx} ANALYSIS")
    print(f"{'='*80}")
    
    if gt_bbox:
        gt_center_x = (gt_bbox[0] + gt_bbox[2]) / 2
        print(f"Ground Truth (CV):  ✓ Ball detected at x={gt_center_x:.3f}")
        print(f"  Bbox: [{gt_bbox[0]:.3f}, {gt_bbox[1]:.3f}, {gt_bbox[2]:.3f}, {gt_bbox[3]:.3f}]")
    else:
        print(f"Ground Truth (CV):  ✗ NO BALL DETECTED")
    
    # Prompt the VLM
    prompt = """Locate the red ball in this image.

If you see a red ball, provide its bounding box: <bbox>[x1,y1,x2,y2]</bbox>
If you DO NOT see a red ball, respond with: "No red ball visible"

Use normalized coordinates (0 to 1)."""
    
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt}
    ]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"\nVLM Inference:")
    print(f"  Input tokens: {inputs['input_ids'].shape[1]}")
    
    # Generate with output_scores to get token probabilities
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=1.0,
            output_scores=True,
            return_dict_in_generate=True,
        )
    
    # Extract generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output.sequences[:, input_len:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"  Output tokens: {len(generated_ids[0])}")
    print(f"\nVLM Response:")
    print(f"  {response}")
    
    # Parse VLM bbox
    vlm_bbox = None
    patterns = [
        r'<bbox>\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]</bbox>',
        r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            vlm_bbox = [float(match.group(i)) for i in range(1, 5)]
            break
    
    # Analyze token-by-token output
    print(f"\nToken Analysis:")
    print(f"  First 30 tokens:")
    tokens = processor.tokenizer.convert_ids_to_tokens(generated_ids[0][:30].tolist())
    for i, (tid, tok) in enumerate(zip(generated_ids[0][:30].tolist(), tokens)):
        print(f"    {i:2d}: {tid:6d} -> {repr(tok)}")
    
    # Compare VLM vs Ground Truth
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")
    
    if gt_bbox and vlm_bbox:
        # Calculate IoU
        x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
        x1_vlm, y1_vlm, x2_vlm, y2_vlm = vlm_bbox
        
        # Intersection
        x1_i = max(x1_gt, x1_vlm)
        y1_i = max(y1_gt, y1_vlm)
        x2_i = min(x2_gt, x2_vlm)
        y2_i = min(y2_gt, y2_vlm)
        
        if x1_i < x2_i and y1_i < y2_i:
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
        else:
            intersection = 0
        
        # Union
        area_gt = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        area_vlm = (x2_vlm - x1_vlm) * (y2_vlm - y1_vlm)
        union = area_gt + area_vlm - intersection
        
        iou = intersection / union if union > 0 else 0
        
        gt_center_x = (x1_gt + x2_gt) / 2
        vlm_center_x = (x1_vlm + x2_vlm) / 2
        center_error = abs(gt_center_x - vlm_center_x)
        
        print(f"Ground Truth:  [{x1_gt:.3f}, {y1_gt:.3f}, {x2_gt:.3f}, {y2_gt:.3f}]  center_x={gt_center_x:.3f}")
        print(f"VLM Predicted: [{x1_vlm:.3f}, {y1_vlm:.3f}, {x2_vlm:.3f}, {y2_vlm:.3f}]  center_x={vlm_center_x:.3f}")
        print(f"\nIoU: {iou:.3f}")
        print(f"Center X Error: {center_error:.3f}")
        
        if iou > 0.5:
            print(f"✓ GOOD: VLM prediction matches ground truth (IoU={iou:.3f})")
        elif iou > 0.2:
            print(f"~ PARTIAL: VLM prediction is close but not accurate (IoU={iou:.3f})")
        else:
            print(f"✗ BAD: VLM prediction is far from ground truth (IoU={iou:.3f})")
    
    elif gt_bbox and not vlm_bbox:
        print(f"Ground Truth:  ✓ Ball exists")
        print(f"VLM Predicted: ✗ No bbox detected")
        print(f"\n✗ FALSE NEGATIVE: VLM failed to detect the ball")
    
    elif not gt_bbox and vlm_bbox:
        vlm_center_x = (vlm_bbox[0] + vlm_bbox[2]) / 2
        print(f"Ground Truth:  ✗ No ball exists")
        print(f"VLM Predicted: ✓ Bbox at x={vlm_center_x:.3f}")
        print(f"\n✗ FALSE POSITIVE: VLM HALLUCINATED a ball that doesn't exist!")
    
    else:
        print(f"Ground Truth:  ✗ No ball")
        print(f"VLM Predicted: ✗ No bbox")
        print(f"\n✓ TRUE NEGATIVE: Both agree no ball exists")
    
    return {
        'frame_idx': frame_idx,
        'ground_truth': gt_bbox,
        'vlm_prediction': vlm_bbox,
        'response': response,
        'tokens': tokens[:30],
        'mask': mask,
    }


def visualize_comparison(frame_rgb, frame_idx, gt_bbox, vlm_bbox, mask, output_dir):
    """Create side-by-side comparison visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    height, width = frame_rgb.shape[:2]
    
    # Create figure with 3 panels: original, GT, VLM
    fig_width = width * 3
    fig_height = height
    figure = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)
    
    # Panel 1: Original frame
    figure[:, :width] = frame_rgb
    
    # Panel 2: Ground truth (with mask overlay)
    gt_panel = frame_rgb.copy()
    # Overlay red mask
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    gt_panel = cv2.addWeighted(gt_panel, 0.7, mask_3ch, 0.3, 0)
    figure[:, width:2*width] = gt_panel
    
    # Panel 3: VLM prediction
    vlm_panel = frame_rgb.copy()
    figure[:, 2*width:] = vlm_panel
    
    # Convert to PIL for drawing
    img = Image.fromarray(figure)
    draw = ImageDraw.Draw(img)
    
    # Draw GT bbox on panel 2
    if gt_bbox:
        x1, y1, x2, y2 = gt_bbox
        x1_px = int(x1 * width) + width
        y1_px = int(y1 * height)
        x2_px = int(x2 * width) + width
        y2_px = int(y2 * height)
        draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline='green', width=4)
        draw.text((x1_px, y1_px - 25), "Ground Truth", fill='green', font=font)
    
    # Draw VLM bbox on panel 3
    if vlm_bbox:
        x1, y1, x2, y2 = vlm_bbox
        x1_px = int(x1 * width) + 2*width
        y1_px = int(y1 * height)
        x2_px = int(x2 * width) + 2*width
        y2_px = int(y2 * height)
        draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline='red', width=4)
        draw.text((x1_px, y1_px - 25), "VLM Prediction", fill='red', font=font)
    
    # Add labels
    draw.text((10, 10), f"Frame {frame_idx}", fill='white', font=font)
    draw.text((width + 10, 10), "Ground Truth (CV)", fill='green', font=font)
    draw.text((2*width + 10, 10), "VLM Prediction", fill='red', font=font)
    
    # Save
    output_path = output_dir / f"comparison_frame_{frame_idx:04d}.jpg"
    img.save(output_path)
    print(f"\n✓ Saved comparison: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose VLM bbox hallucination")
    parser.add_argument("video_path", type=str, help="Path to video")
    parser.add_argument("--frames", type=str, default="0,48,96,144,191",
                       help="Frames to analyze")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    
    frame_indices = [int(x.strip()) for x in args.frames.split(",")]
    
    print("="*80)
    print("BBOX HALLUCINATION DIAGNOSTIC")
    print("="*80)
    print(f"Video: {args.video_path}")
    print(f"Frames: {frame_indices}")
    print(f"Model: {args.model_id}")
    print()
    
    # Load model
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print("✓ Model loaded\n")
    
    # Extract frames
    cap = cv2.VideoCapture(args.video_path)
    
    results = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Analyze
        result = analyze_single_frame_detailed(frame_rgb, frame_idx, model, processor, args.device)
        results.append(result)
        
        # Visualize
        visualize_comparison(
            frame_rgb, frame_idx,
            result['ground_truth'],
            result['vlm_prediction'],
            result['mask'],
            "outputs/hallucination_diagnosis"
        )
    
    cap.release()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    true_positives = sum(1 for r in results if r['ground_truth'] and r['vlm_prediction'])
    false_positives = sum(1 for r in results if not r['ground_truth'] and r['vlm_prediction'])
    false_negatives = sum(1 for r in results if r['ground_truth'] and not r['vlm_prediction'])
    true_negatives = sum(1 for r in results if not r['ground_truth'] and not r['vlm_prediction'])
    
    print(f"\nDetection Statistics:")
    print(f"  True Positives:  {true_positives} (correctly detected ball)")
    print(f"  False Positives: {false_positives} (HALLUCINATED ball)")
    print(f"  False Negatives: {false_negatives} (missed real ball)")
    print(f"  True Negatives:  {true_negatives} (correctly no detection)")
    
    if false_positives > 0:
        print(f"\n⚠ WARNING: VLM hallucinated {false_positives} bboxes!")
        print("  This confirms the model is guessing based on prompt, not visual evidence.")
    
    print(f"\n✓ Analysis complete. Check outputs/hallucination_diagnosis/")


if __name__ == "__main__":
    main()
