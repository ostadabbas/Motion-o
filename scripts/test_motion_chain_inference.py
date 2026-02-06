#!/usr/bin/env python3
"""
Test motion-based chain-of-thought inference.

This script tests if the VLM can produce structured spatio-temporal evidence chains
in the format that GRPO training aims to teach:

Expected format:
  Step 1: [t1-t2] Object <bbox>[x1,y1,x2,y2]</bbox> does action
  Motion: centroid shifts from (x1,y1) to (x2,y2), velocity V px/s, direction D
  Description: Natural language description
  
  Step 2: [t3-t4] Object <bbox>[x3,y3,x4,y4]</bbox> does another action
  Motion: ...
  Description: ...
  
  Answer: Final answer grounded in the evidence chain
"""

import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration
)


def extract_frames_uniformly(video_path: str, num_frames: int):
    """Extract frames uniformly from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    metadata = {
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration
    }
    
    # Sample frames uniformly
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames, frame_indices.tolist(), metadata


def create_motion_reasoning_prompt(question: str, provide_example: bool = True):
    """
    Create a prompt that asks for structured spatio-temporal evidence chains.
    
    This is the format that GRPO training aims to teach.
    """
    
    if provide_example:
        prompt = f"""{question}

Please provide a structured reasoning chain with the following format:

Step 1: [start_time-end_time] Describe what happens with bounding box <bbox>[x1,y1,x2,y2]</bbox>
Motion: Describe motion with coordinates, velocity, direction
Description: Natural language explanation

Step 2: [start_time-end_time] Next event with <bbox>[x1,y1,x2,y2]</bbox>
Motion: Motion descriptors
Description: Explanation

... (continue for each key event)

Answer: Your final answer based on the evidence chain

Example format:
Step 1: [0.0s-2.0s] Ball <bbox>[0.15,0.45,0.25,0.55]</bbox> starts moving
Motion: Centroid at (0.20, 0.50), velocity 0.15 units/s toward right
Description: The red ball begins moving from the left side of the frame

Use normalized coordinates (0-1) for all bounding boxes."""
    
    else:
        prompt = f"""{question}

Provide a step-by-step reasoning chain showing:
- Temporal intervals [start-end]
- Spatial locations with <bbox>[x1,y1,x2,y2]</bbox>
- Motion descriptions (direction, velocity)
- Natural language explanations

End with a final answer based on your evidence."""
    
    return prompt


def run_motion_chain_inference(
    video_path: str,
    question: str,
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    num_frames: int = 8,
    device: str = "cuda:0",
    provide_example: bool = True,
    max_tokens: int = 1024
):
    """
    Run inference to generate motion-based chain of thought.
    """
    
    print("="*80)
    print("MOTION CHAIN-OF-THOUGHT INFERENCE TEST")
    print("="*80)
    print(f"Video: {Path(video_path).name}")
    print(f"Model: {model_id}")
    print(f"Question: {question}")
    print(f"Max output tokens: {max_tokens}")
    print("="*80)
    
    # Extract frames
    print(f"\n[1/4] Extracting {num_frames} frames...")
    frames, frame_indices, metadata = extract_frames_uniformly(video_path, num_frames)
    pil_images = [Image.fromarray(f) for f in frames]
    
    print(f"  Video: {metadata['width']}x{metadata['height']}, {metadata['total_frames']} frames, {metadata['duration']:.2f}s")
    print(f"  Frame indices: {frame_indices}")
    
    # Load model
    print(f"\n[2/4] Loading model...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    if "Qwen3" in model_id or "qwen3" in model_id:
        model_class = Qwen3VLForConditionalGeneration
    elif "Qwen2.5" in model_id or "qwen2.5" in model_id or "Qwen2_5" in model_id:
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        model_class = Qwen2VLForConditionalGeneration
    
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  ✓ Loaded: {model.__class__.__name__}")
    
    # Create prompt
    print(f"\n[3/4] Building motion reasoning prompt...")
    prompt = create_motion_reasoning_prompt(question, provide_example)
    
    print(f"\nPrompt:")
    print("-"*80)
    print(prompt)
    print("-"*80)
    
    # Build messages
    content = []
    for img in pil_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    
    messages = [{"role": "user", "content": content}]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Prepare inputs
    inputs = processor(
        text=[text],
        images=pil_images,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"\n[4/4] Generating motion reasoning chain...")
    print(f"  Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"  Generating up to {max_tokens} tokens...")
    print()
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,  # Some creativity for reasoning
            top_p=0.9,
        )
    
    # Decode
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output[:, input_len:]
    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    num_generated_tokens = len(generated_ids[0])
    
    print("="*80)
    print(f"GENERATED MOTION REASONING CHAIN ({num_generated_tokens} tokens)")
    print("="*80)
    print()
    print(response_text)
    print()
    print("="*80)
    
    return {
        'question': question,
        'response': response_text,
        'metadata': {
            'video_path': video_path,
            'model_id': model_id,
            'num_frames': num_frames,
            'frame_indices': frame_indices,
            'video_metadata': metadata,
            'input_tokens': inputs['input_ids'].shape[1],
            'generated_tokens': num_generated_tokens,
            'max_tokens': max_tokens,
            'provide_example': provide_example
        }
    }


def save_results(result, output_dir):
    """Save inference results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full result as JSON
    json_path = output_dir / "motion_chain_result.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save response as text file
    txt_path = output_dir / "motion_chain_response.txt"
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MOTION CHAIN-OF-THOUGHT INFERENCE RESULT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Video: {result['metadata']['video_path']}\n")
        f.write(f"Model: {result['metadata']['model_id']}\n")
        f.write(f"Frames: {result['metadata']['num_frames']} frames\n")
        f.write(f"Frame indices: {result['metadata']['frame_indices']}\n")
        f.write(f"Video duration: {result['metadata']['video_metadata']['duration']:.2f}s\n")
        f.write(f"Generated tokens: {result['metadata']['generated_tokens']}\n")
        f.write(f"\n")
        f.write(f"Question:\n{result['question']}\n")
        f.write(f"\n")
        f.write("="*80 + "\n")
        f.write("RESPONSE:\n")
        f.write("="*80 + "\n\n")
        f.write(result['response'])
        f.write("\n\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Results saved:")
    print(f"  JSON: {json_path}")
    print(f"  Text: {txt_path}")
    
    return json_path, txt_path


def analyze_response_format(response: str):
    """Analyze if the response follows the expected format."""
    import re
    
    print("\n" + "="*80)
    print("RESPONSE FORMAT ANALYSIS")
    print("="*80)
    
    # Check for temporal intervals
    temporal_pattern = r'\[([0-9.]+)s?[-–]([0-9.]+)s?\]'
    temporal_matches = re.findall(temporal_pattern, response)
    
    # Check for bboxes
    bbox_pattern = r'<bbox>\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]</bbox>'
    bbox_matches = re.findall(bbox_pattern, response)
    
    # Check for step structure
    step_pattern = r'(?:Step|Evidence)\s+(\d+)'
    step_matches = re.findall(step_pattern, response, re.IGNORECASE)
    
    # Check for motion descriptors
    has_motion = bool(re.search(r'(?:motion|velocity|direction|displacement|centroid)', response, re.IGNORECASE))
    
    # Check for answer
    has_answer = bool(re.search(r'(?:answer|conclusion):', response, re.IGNORECASE))
    
    print(f"\nStructure detected:")
    print(f"  Temporal intervals: {len(temporal_matches)} found")
    if temporal_matches:
        for i, (start, end) in enumerate(temporal_matches[:3]):
            print(f"    [{start}-{end}s]")
    
    print(f"  Bounding boxes: {len(bbox_matches)} found")
    if bbox_matches:
        for i, bbox in enumerate(bbox_matches[:3]):
            print(f"    [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
    
    print(f"  Steps/Evidence: {len(step_matches)} found")
    print(f"  Motion descriptors: {'✓' if has_motion else '✗'}")
    print(f"  Final answer: {'✓' if has_answer else '✗'}")
    
    # Overall assessment
    print(f"\n{'='*80}")
    print("ASSESSMENT FOR GRPO TRAINING")
    print(f"{'='*80}")
    
    if len(bbox_matches) >= 2 and len(step_matches) >= 2 and has_motion:
        print("✓ GOOD: Response follows structured reasoning format")
        print("  The model can generate multi-step spatial evidence chains")
        print("  GRPO training will REFINE this capability")
    elif len(bbox_matches) >= 1 or len(step_matches) >= 1:
        print("~ PARTIAL: Response has some structure but incomplete")
        print("  The model has basic capability but inconsistent")
        print("  GRPO training will STRENGTHEN and STANDARDIZE this format")
    else:
        print("✗ LIMITED: Response lacks structured spatial reasoning")
        print("  The model provides unstructured descriptions")
        print("  GRPO training will TEACH this capability from scratch")
    
    print(f"\nKey gaps to address in GRPO training:")
    if len(temporal_matches) < 2:
        print("  - Need more consistent temporal intervals")
    if len(bbox_matches) < 3:
        print("  - Need more bounding boxes throughout reasoning")
    if not has_motion:
        print("  - Need explicit motion descriptors (velocity, direction)")
    if len(step_matches) < 2:
        print("  - Need clearer step-by-step structure")
    
    print(f"\nThis output is your BASELINE - GRPO will improve:")
    print(f"  1. Spatial accuracy (bbox IoU with ground truth)")
    print(f"  2. Temporal consistency (smooth motion trajectories)")
    print(f"  3. Format compliance (always include all required fields)")
    print(f"  4. Motion quantification (explicit velocity/displacement values)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test motion chain-of-thought inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script tests the VLM's ability to produce structured spatio-temporal
evidence chains - the target output format for your GRPO training.

Example questions:
  - "What is the motion trajectory of the red ball?"
  - "Describe how the ball moves across the scene with spatial evidence"
  - "Track the ball's motion and explain its path"
"""
    )
    
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("question", type=str, help="Motion-related question")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                       help="Model ID")
    parser.add_argument("--num-frames", type=int, default=8,
                       help="Number of frames to sample")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="Max tokens to generate")
    parser.add_argument("--no-example", action="store_true",
                       help="Don't provide example format in prompt")
    parser.add_argument("--output-dir", type=str, default="outputs/motion_chain_inference",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Check video exists
    if not Path(args.video_path).exists():
        print(f"ERROR: Video not found: {args.video_path}")
        sys.exit(1)
    
    # Run inference
    result = run_motion_chain_inference(
        args.video_path,
        args.question,
        model_id=args.model_id,
        num_frames=args.num_frames,
        device=args.device,
        provide_example=not args.no_example,
        max_tokens=args.max_tokens
    )
    
    # Analyze response
    analyze_response_format(result['response'])
    
    # Save results
    save_results(result, args.output_dir)
    
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")
    print(f"\nThis is your BASELINE capability before GRPO training.")
    print(f"Use this to:")
    print(f"  1. Compare against GRPO-trained model outputs")
    print(f"  2. Design reward functions that encourage better structure")
    print(f"  3. Identify which aspects need most improvement")


if __name__ == "__main__":
    main()
