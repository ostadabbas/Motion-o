#!/usr/bin/env python3
"""
Test think_bbox → pred_bbox inference with native Qwen bbox format.

Tests three prompt strategies in order of scaffolding:
  1. Per-frame grounding (one frame at a time, like tracking test — sanity check)
  2. Explicit frame-binding (all frames, but prompt binds each frame to a step)
  3. Free-form chain (all frames, structured prompt, model decides steps)

This validates whether the base VLM can produce spatially-varying bboxes
across steps before committing to GRPO training.
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
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


# ============================================================================
# Model-native bbox format detection
# ============================================================================

def detect_bbox_format(model_id: str) -> dict:
    """
    Return bbox format config matched to model's pretrained format.
    
    Qwen2-VL, Qwen2.5-VL, Qwen3-VL all use:
      <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    on a 0-1000 normalized coordinate scale.
    """
    mid = model_id.lower()
    if any(k in mid for k in ["qwen2", "qwen3", "qwen2.5"]):
        return {
            "name": "qwen_native",
            "scale": 1000,
            "open_tag": "<|box_start|>",
            "close_tag": "<|box_end|>",
            "example": "<|box_start|>(150,450),(250,550)<|box_end|>",
            "instruction": (
                "Use bounding boxes in the model's native format: "
                "<|box_start|>(x1,y1),(x2,y2)<|box_end|> "
                "where coordinates are on a 0-1000 normalized scale "
                "(0=top-left, 1000=bottom-right)."
            ),
        }
    # Fallback for non-Qwen models
    return {
        "name": "generic_normalized",
        "scale": 1.0,
        "open_tag": "<bbox>",
        "close_tag": "</bbox>",
        "example": "<bbox>[0.15, 0.45, 0.25, 0.55]</bbox>",
        "instruction": (
            "Use bounding boxes in the format: "
            "<bbox>[x1, y1, x2, y2]</bbox> "
            "with normalized coordinates from 0.0 to 1.0."
        ),
    }


# ============================================================================
# Bbox extraction (handles all Qwen native formats + fallbacks)
# ============================================================================

@dataclass
class ParsedBbox:
    x1: float  # pixel coords
    y1: float
    x2: float
    y2: float
    raw: str  # original text matched
    scale_source: str  # "qwen_1000" | "normalized_01" | "pixels"

    @property
    def cx(self): return (self.x1 + self.x2) / 2
    @property
    def cy(self): return (self.y1 + self.y2) / 2
    @property
    def width(self): return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1
    
    def to_normalized(self, img_w, img_h):
        return [self.x1/img_w, self.y1/img_h, self.x2/img_w, self.y2/img_h]


def extract_all_bboxes(text: str, img_w: int = 1280, img_h: int = 720) -> List[ParsedBbox]:
    """Extract bboxes from text, handling multiple format conventions."""
    bboxes = []

    # 1) Qwen native with tags: <|box_start|>(x1,y1),(x2,y2)<|box_end|>  (0-1000 scale)
    for m in re.finditer(
        r'<\|box_start\|>\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)<\|box_end\|>', text
    ):
        coords = [int(m.group(i)) for i in range(1, 5)]
        bboxes.append(ParsedBbox(
            x1=coords[0] / 1000.0 * img_w,
            y1=coords[1] / 1000.0 * img_h,
            x2=coords[2] / 1000.0 * img_w,
            y2=coords[3] / 1000.0 * img_h,
            raw=m.group(0),
            scale_source="qwen_1000",
        ))

    # 2) Bare Qwen format (COMMON): (x1,y1),(x2,y2) without tags (0-1000 scale)
    # This is what the model actually outputs!
    if not bboxes:  # Only if tagged version wasn't found
        for m in re.finditer(r'\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)', text):
            coords = [int(m.group(i)) for i in range(1, 5)]
            # Validate it's a reasonable bbox
            if coords[0] < coords[2] and coords[1] < coords[3] and all(c <= 1000 for c in coords):
                bboxes.append(ParsedBbox(
                    x1=coords[0] / 1000.0 * img_w,
                    y1=coords[1] / 1000.0 * img_h,
                    x2=coords[2] / 1000.0 * img_w,
                    y2=coords[3] / 1000.0 * img_h,
                    raw=m.group(0),
                    scale_source="qwen_1000",
                ))

    # 3) <bbox>[x1,y1,x2,y2]</bbox>  (0-1 or 0-1000)
    if not bboxes:
        for m in re.finditer(
            r'<bbox>\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]</bbox>', text
        ):
            coords = [float(m.group(i)) for i in range(1, 5)]
            if all(c <= 1.1 for c in coords):
                bboxes.append(ParsedBbox(
                    x1=coords[0]*img_w, y1=coords[1]*img_h,
                    x2=coords[2]*img_w, y2=coords[3]*img_h,
                    raw=m.group(0), scale_source="normalized_01",
                ))
            elif all(c <= 1100 for c in coords):
                bboxes.append(ParsedBbox(
                    x1=coords[0]/1000*img_w, y1=coords[1]/1000*img_h,
                    x2=coords[2]/1000*img_w, y2=coords[3]/1000*img_h,
                    raw=m.group(0), scale_source="qwen_1000",
                ))

    # 4) Plain [x1,y1,x2,y2] as last fallback
    if not bboxes:
        for m in re.finditer(r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]', text):
            coords = [float(m.group(i)) for i in range(1, 5)]
            if coords[0] < coords[2] and coords[1] < coords[3]:
                if all(c <= 1.1 for c in coords):
                    # Normalized 0-1
                    bboxes.append(ParsedBbox(
                        x1=coords[0]*img_w, y1=coords[1]*img_h,
                        x2=coords[2]*img_w, y2=coords[3]*img_h,
                        raw=m.group(0), scale_source="normalized_01",
                    ))
                elif all(c <= 1100 for c in coords):
                    # Qwen 0-1000
                    bboxes.append(ParsedBbox(
                        x1=coords[0]/1000*img_w, y1=coords[1]/1000*img_h,
                        x2=coords[2]/1000*img_w, y2=coords[3]/1000*img_h,
                        raw=m.group(0), scale_source="qwen_1000",
                    ))
                else:
                    # Pixel coordinates
                    bboxes.append(ParsedBbox(
                        x1=coords[0], y1=coords[1],
                        x2=coords[2], y2=coords[3],
                        raw=m.group(0), scale_source="pixels",
                    ))

    return bboxes


# ============================================================================
# Parse think_bbox and pred_bbox sections from a step
# ============================================================================

@dataclass
class ParsedStep:
    step_num: int
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    think_bboxes: List[ParsedBbox] = field(default_factory=list)
    pred_bboxes: List[ParsedBbox] = field(default_factory=list)
    motion_text: str = ""
    description: str = ""
    raw_text: str = ""


def parse_think_pred_chain(text: str, img_w: int = 1280, img_h: int = 720) -> Tuple[List[ParsedStep], str]:
    """
    Parse model output with Think/Predict sections.
    
    Returns (steps, answer).
    """
    steps = []
    answer = ""

    # Split on Step N:
    parts = re.split(r'(?=Step\s+\d+\s*:)', text, flags=re.IGNORECASE)

    for part in parts:
        step_match = re.match(r'Step\s+(\d+)\s*:', part, re.IGNORECASE)
        if not step_match:
            # Check if this contains the answer
            ans_match = re.search(r'Answer\s*:\s*(.+)', part, re.IGNORECASE | re.DOTALL)
            if ans_match:
                answer = ans_match.group(1).strip()
            continue

        step = ParsedStep(step_num=int(step_match.group(1)), raw_text=part.strip())

        # Time interval [start-end]
        t_match = re.search(r'\[(\d+\.?\d*)s?\s*[-–]\s*(\d+\.?\d*)s?\]', part)
        if t_match:
            step.time_start = float(t_match.group(1))
            step.time_end = float(t_match.group(2))

        # Think: line (rough estimate)
        think_match = re.search(r'Think\s*:\s*(.+?)(?:\n|Predict)', part, re.IGNORECASE | re.DOTALL)
        if think_match:
            think_text = think_match.group(1).strip()
            step.think_bboxes = extract_all_bboxes(think_text, img_w, img_h)

        # Predict: line (refined bbox)
        pred_match = re.search(r'Predict\s*:\s*(.+?)(?:\n|Motion|$)', part, re.IGNORECASE | re.DOTALL)
        if pred_match:
            pred_text = pred_match.group(1).strip()
            step.pred_bboxes = extract_all_bboxes(pred_text, img_w, img_h)

        # If no Think/Predict labels, try to extract all bboxes in order
        if not step.think_bboxes and not step.pred_bboxes:
            all_bboxes = extract_all_bboxes(part, img_w, img_h)
            if len(all_bboxes) >= 2:
                # First is think, second is predict
                step.think_bboxes = [all_bboxes[0]]
                step.pred_bboxes = [all_bboxes[1]]
            elif len(all_bboxes) == 1:
                # Only one bbox, treat as predict
                step.pred_bboxes = all_bboxes

        # Motion text
        motion_match = re.search(r'Motion\s*:\s*(.+?)(?:\n|$)', part, re.IGNORECASE)
        if motion_match:
            step.motion_text = motion_match.group(1).strip()

        # Description (first line after Step N:)
        desc_match = re.search(r'Description\s*:\s*(.+?)(?:\n|$)', part, re.IGNORECASE)
        if desc_match:
            step.description = desc_match.group(1).strip()
        else:
            lines = part.split('\n')
            if lines:
                first_line = re.sub(r'^Step\s+\d+\s*:\s*', '', lines[0]).strip()
                first_line = re.sub(r'\[\d+\.?\d*s?[-–]\d+\.?\d*s?\]', '', first_line).strip()
                step.description = first_line[:200]

        # Check for answer in this step
        ans_match = re.search(r'Answer\s*:\s*(.+)', part, re.IGNORECASE | re.DOTALL)
        if ans_match:
            answer = ans_match.group(1).strip()

        steps.append(step)

    return steps, answer


# ============================================================================
# Video frame extraction
# ============================================================================

def extract_frames(video_path: str, num_frames: int = 8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    timestamps = [idx / fps for idx in indices]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    meta = dict(total_frames=total, fps=fps, width=w, height=h,
                duration=duration, frame_indices=indices.tolist(),
                timestamps=timestamps)
    return frames, meta


# ============================================================================
# Three prompt strategies
# ============================================================================

def build_prompt_per_frame(frame_idx: int, timestamp: float, bbox_fmt: dict) -> str:
    """Strategy 1: One frame, one bbox. Sanity check."""
    return (
        f"This is frame {frame_idx} at t={timestamp:.2f}s from a video.\n"
        f"Locate the red ball in this frame.\n\n"
        f"Output format: (x1,y1),(x2,y2) where coordinates are 0-1000 scale.\n"
        f"Example: (250,400),(350,600) means ball at x=250-350, y=400-600.\n\n"
        f"Respond with ONLY the coordinates, nothing else."
    )


def build_prompt_explicit_binding(timestamps: List[float], bbox_fmt: dict,
                                   question: str) -> str:
    """Strategy 2: All frames shown, each explicitly bound to a step."""
    frame_lines = []
    for i, t in enumerate(timestamps):
        frame_lines.append(f"  Frame {i+1} at t={t:.2f}s")

    return f"""{question}

You are shown {len(timestamps)} frames from the video at these times:
{chr(10).join(frame_lines)}

For EACH frame, locate the red ball and provide:

Step N: [time] Description
  Think: (x1,y1),(x2,y2) - rough estimate
  Predict: (x1,y1),(x2,y2) - refined bbox
  Motion: how the ball moved

Coordinates use 0-1000 scale (0=left/top, 1000=right/bottom).

Example format:
Step 1: [0.0s] Ball on left side
  Think: (150,400),(250,600)
  Predict: (160,420),(240,580)
  Motion: Starting position

Step 2: [1.0s] Ball moved right
  Think: (450,390),(550,590)
  Predict: (460,410),(540,570)
  Motion: Moved 300 units right

CRITICAL: Each step must have DIFFERENT coordinates showing actual ball position in that frame.
Do NOT copy coordinates between steps!

After all steps: Answer: your final answer"""


def build_prompt_freeform_chain(bbox_fmt: dict, question: str) -> str:
    """Strategy 3: Free-form chain — the actual GRPO training prompt."""
    return f"""{question}

Break down the motion into key steps. For each step:

Step N: [start-end seconds] What happens
  Think: (x1,y1),(x2,y2) - rough ball location
  Predict: (x1,y1),(x2,y2) - refined bbox
  Motion: direction, speed, displacement

Coordinates: 0-1000 scale (0=left/top, 1000=right/bottom)

Example:
Step 1: [0.0-1.0s] Ball starts on left
  Think: (150,400),(250,600)
  Predict: (160,420),(240,580)
  Motion: Starting position, stationary

Step 2: [1.0-2.0s] Ball moves right
  Think: (500,390),(600,590)
  Predict: (510,410),(590,570)
  Motion: Moved 350 units right, velocity ~350 units/s

Make sure each step has DIFFERENT coordinates tracking the ball's actual position!

End with: Answer: your conclusion"""


# ============================================================================
# Model inference
# ============================================================================

def load_model(model_id: str, device: str, use_4bit: bool = False, multi_gpu: bool = False):
    """
    Load VLM model with optional 4-bit quantization and multi-GPU support.
    
    Args:
        model_id: HuggingFace model ID
        device: Device string (e.g., "cuda:0" or "auto" for multi-GPU)
        use_4bit: Enable 4-bit quantization (recommended for 7B/8B models)
        multi_gpu: Enable multi-GPU model parallelism (auto device mapping)
    """
    from transformers import BitsAndBytesConfig
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Select model class
    mid = model_id.lower()
    if "qwen3" in mid:
        cls = Qwen3VLForConditionalGeneration
    elif "qwen2.5" in mid or "qwen2_5" in mid:
        cls = Qwen2_5_VLForConditionalGeneration
    else:
        cls = Qwen2VLForConditionalGeneration
    
    # Configure quantization
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print(f"  Using 4-bit quantization (NF4)")
    
    # Configure device mapping
    if multi_gpu or device == "auto":
        device_map = "auto"  # Automatically split across all available GPUs
        print(f"  Device mapping: AUTO (will split across all GPUs)")
    else:
        device_map = device
        print(f"  Device mapping: {device}")
    
    model = cls.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    
    # Print device allocation if multi-GPU
    if device_map == "auto" and hasattr(model, 'hf_device_map'):
        print(f"\n  Model device allocation:")
        device_counts = {}
        for name, dev in model.hf_device_map.items():
            device_counts[dev] = device_counts.get(dev, 0) + 1
        for dev, count in sorted(device_counts.items()):
            print(f"    {dev}: {count} layers")
    
    return model, processor


def run_inference(model, processor, images: List[Image.Image],
                  prompt: str, device: str, max_tokens: int = 1024) -> str:
    """Run single inference pass."""
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt")
    
    # Handle device mapping: if device="auto", model is distributed, send to first device
    if device == "auto":
        # Get first device from model's device map if available
        if hasattr(model, 'hf_device_map'):
            first_device = next(iter(set(model.hf_device_map.values())))
        else:
            first_device = "cuda:0"  # Default to first GPU
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=True, temperature=0.7, top_p=0.9)

    gen_ids = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0]


# ============================================================================
# Visualization
# ============================================================================

def draw_bboxes_on_frame(frame_rgb: np.ndarray, think_bboxes: List[ParsedBbox],
                         pred_bboxes: List[ParsedBbox], step_label: str) -> Image.Image:
    """Draw think (magenta dashed-style) and pred (red solid) bboxes."""
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = font_sm = ImageFont.load_default()

    # Think bboxes — magenta, thinner
    for bb in think_bboxes:
        draw.rectangle([bb.x1, bb.y1, bb.x2, bb.y2], outline='magenta', width=2)
        draw.text((bb.x1, bb.y1 - 16), "think", fill='magenta', font=font_sm)

    # Pred bboxes — red, thicker
    for bb in pred_bboxes:
        draw.rectangle([bb.x1, bb.y1, bb.x2, bb.y2], outline='red', width=4)
        draw.text((bb.x1, bb.y1 - 16), "pred", fill='red', font=font_sm)
        # Center dot
        draw.ellipse([bb.cx-5, bb.cy-5, bb.cx+5, bb.cy+5], fill='red', outline='yellow', width=2)

    # Label
    draw.text((10, 10), step_label, fill='yellow', font=font,
              stroke_width=2, stroke_fill='black')

    return img


# ============================================================================
# Analysis
# ============================================================================

def analyze_spatial_variation(steps: List[ParsedStep]) -> dict:
    """Check if bboxes actually change across steps."""
    pred_centers = []
    think_centers = []
    for s in steps:
        if s.pred_bboxes:
            b = s.pred_bboxes[0]
            pred_centers.append((b.cx, b.cy))
        if s.think_bboxes:
            b = s.think_bboxes[0]
            think_centers.append((b.cx, b.cy))

    def variation(centers):
        if len(centers) < 2:
            return {"count": len(centers), "all_identical": True, "max_displacement": 0}
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        displacements = []
        for i in range(len(centers)-1):
            d = np.sqrt((centers[i+1][0]-centers[i][0])**2 +
                        (centers[i+1][1]-centers[i][1])**2)
            displacements.append(d)
        all_same = all(d < 2.0 for d in displacements)
        return {
            "count": len(centers),
            "all_identical": all_same,
            "max_displacement": max(displacements) if displacements else 0,
            "total_displacement": sum(displacements),
            "x_range": max(xs) - min(xs),
            "y_range": max(ys) - min(ys),
            "centers": [(round(c[0],1), round(c[1],1)) for c in centers],
        }

    # Think vs pred refinement
    refinement_improvements = []
    for s in steps:
        if s.think_bboxes and s.pred_bboxes:
            tb, pb = s.think_bboxes[0], s.pred_bboxes[0]
            dist = np.sqrt((tb.cx - pb.cx)**2 + (tb.cy - pb.cy)**2)
            refinement_improvements.append(round(dist, 1))

    return {
        "pred_variation": variation(pred_centers),
        "think_variation": variation(think_centers),
        "think_to_pred_shifts": refinement_improvements,
    }


def print_analysis(steps: List[ParsedStep], answer: str, strategy: str,
                   meta: dict):
    W, H = meta['width'], meta['height']
    print(f"\n{'='*80}")
    print(f"ANALYSIS — Strategy: {strategy}")
    print(f"{'='*80}")
    print(f"Steps parsed: {len(steps)}")
    print(f"Answer: {answer[:200] if answer else 'NONE'}")

    for s in steps:
        t_str = f"[{s.time_start:.1f}s-{s.time_end:.1f}s]" if s.time_start is not None else "[no time]"
        print(f"\n  Step {s.step_num}: {t_str}")
        print(f"    Description: {s.description[:100]}")
        if s.think_bboxes:
            b = s.think_bboxes[0]
            print(f"    Think bbox: ({b.x1:.0f},{b.y1:.0f})-({b.x2:.0f},{b.y2:.0f}) "
                  f"center=({b.cx:.0f},{b.cy:.0f}) [{b.scale_source}]")
        else:
            print(f"    Think bbox: NONE")
        if s.pred_bboxes:
            b = s.pred_bboxes[0]
            print(f"    Pred  bbox: ({b.x1:.0f},{b.y1:.0f})-({b.x2:.0f},{b.y2:.0f}) "
                  f"center=({b.cx:.0f},{b.cy:.0f}) [{b.scale_source}]")
        else:
            print(f"    Pred  bbox: NONE")
        if s.motion_text:
            print(f"    Motion: {s.motion_text[:100]}")

    # Variation analysis
    analysis = analyze_spatial_variation(steps)
    pv = analysis['pred_variation']
    tv = analysis['think_variation']

    print(f"\n{'='*80}")
    print("SPATIAL VARIATION ASSESSMENT")
    print(f"{'='*80}")
    print(f"  Pred bboxes:  {pv['count']} found, all_identical={pv['all_identical']}")
    if pv['count'] >= 2:
        print(f"    x-range: {pv['x_range']:.1f}px, y-range: {pv['y_range']:.1f}px")
        print(f"    max step displacement: {pv['max_displacement']:.1f}px")
        print(f"    centers: {pv['centers']}")
    print(f"  Think bboxes: {tv['count']} found, all_identical={tv['all_identical']}")
    if analysis['think_to_pred_shifts']:
        print(f"  Think→Pred shifts: {analysis['think_to_pred_shifts']} px")

    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT FOR GRPO TRAINING READINESS")
    print(f"{'='*80}")

    has_format = len(steps) >= 2
    has_spatial_variation = pv['count'] >= 2 and not pv['all_identical']
    has_think_pred = tv['count'] >= 1 and pv['count'] >= 1
    has_refinement = len(analysis['think_to_pred_shifts']) >= 1

    checks = [
        ("Format compliance (multi-step)", has_format),
        ("Spatial variation (bboxes differ)", has_spatial_variation),
        ("Think/Pred structure present", has_think_pred),
        ("Refinement behavior (think≠pred)", has_refinement),
    ]
    for label, ok in checks:
        print(f"  {'✓' if ok else '✗'} {label}")

    if all(v for _, v in checks):
        print("\n  → STRONG baseline. GRPO will refine accuracy.")
    elif has_format and has_spatial_variation:
        print("\n  → GOOD baseline. GRPO can work with this.")
    elif has_format:
        print("\n  → WEAK baseline. Model follows format but doesn't ground spatially.")
        print("    Consider: larger model, explicit frame-binding prompt, or light SFT warmup.")
    else:
        print("\n  → INSUFFICIENT. Model can't produce the format at all.")
        print("    Consider: different model or SFT warmup before GRPO.")

    return analysis


# ============================================================================
# Main test runner
# ============================================================================

def run_strategy(strategy_name: str, model, processor, frames_rgb: list,
                 meta: dict, bbox_fmt: dict, question: str, device: str,
                 max_tokens: int, output_dir: Path) -> dict:
    """Run one prompt strategy and return results."""

    print(f"\n{'#'*80}")
    print(f"# STRATEGY: {strategy_name}")
    print(f"{'#'*80}")

    pil_images = [Image.fromarray(f) for f in frames_rgb]
    W, H = meta['width'], meta['height']
    timestamps = meta['timestamps']

    if strategy_name == "per_frame":
        # Strategy 1: one frame at a time
        all_steps = []
        for i, (img, ts) in enumerate(zip(pil_images, timestamps)):
            prompt = build_prompt_per_frame(i, ts, bbox_fmt)
            resp = run_inference(model, processor, [img], prompt, device, max_tokens=256)
            bboxes = extract_all_bboxes(resp, W, H)
            step = ParsedStep(
                step_num=i+1, time_start=ts, time_end=ts,
                pred_bboxes=bboxes, description=resp[:200], raw_text=resp,
            )
            all_steps.append(step)
            if bboxes:
                b = bboxes[0]
                print(f"  Frame {i} (t={ts:.2f}s): bbox center=({b.cx:.0f},{b.cy:.0f}) [{b.scale_source}]")
            else:
                print(f"  Frame {i} (t={ts:.2f}s): NO BBOX detected")
                print(f"    Raw response: {resp[:150]}")
        answer = ""
        full_response = "\n".join(s.raw_text for s in all_steps)

    elif strategy_name == "explicit_binding":
        prompt = build_prompt_explicit_binding(timestamps, bbox_fmt, question)
        print(f"\nPrompt:\n{'-'*40}\n{prompt}\n{'-'*40}")
        full_response = run_inference(model, processor, pil_images, prompt, device, max_tokens)
        all_steps, answer = parse_think_pred_chain(full_response, W, H)

    elif strategy_name == "freeform_chain":
        prompt = build_prompt_freeform_chain(bbox_fmt, question)
        print(f"\nPrompt:\n{'-'*40}\n{prompt}\n{'-'*40}")
        full_response = run_inference(model, processor, pil_images, prompt, device, max_tokens)
        all_steps, answer = parse_think_pred_chain(full_response, W, H)

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Print raw response
    print(f"\nRaw response ({len(full_response)} chars):")
    print(f"{'='*60}")
    print(full_response[:2000])
    if len(full_response) > 2000:
        print(f"... [{len(full_response)-2000} more chars]")
    print(f"{'='*60}")

    # Analyze
    analysis = print_analysis(all_steps, answer, strategy_name, meta)

    # Visualize
    strat_dir = output_dir / strategy_name
    strat_dir.mkdir(parents=True, exist_ok=True)

    for i, step in enumerate(all_steps):
        if i < len(frames_rgb):
            frame = frames_rgb[i]
        else:
            frame = frames_rgb[-1]  # Reuse last frame if more steps than frames

        t_str = f"t={step.time_start:.1f}s" if step.time_start is not None else ""
        label = f"Step {step.step_num} {t_str}"
        vis = draw_bboxes_on_frame(frame, step.think_bboxes, step.pred_bboxes, label)
        vis.save(strat_dir / f"step_{step.step_num:02d}.jpg", quality=95)

    # Save results
    result = {
        "strategy": strategy_name,
        "question": question,
        "response": full_response,
        "answer": answer,
        "steps": [
            {
                "step_num": s.step_num,
                "time_start": s.time_start,
                "time_end": s.time_end,
                "think_bboxes": [
                    {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                     "cx": b.cx, "cy": b.cy, "scale": b.scale_source}
                    for b in s.think_bboxes
                ],
                "pred_bboxes": [
                    {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                     "cx": b.cx, "cy": b.cy, "scale": b.scale_source}
                    for b in s.pred_bboxes
                ],
                "motion_text": s.motion_text,
                "description": s.description[:200],
            }
            for s in all_steps
        ],
        "analysis": analysis,
    }

    with open(strat_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Saved to {strat_dir}/")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test think_bbox inference with native Qwen bbox format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Runs three strategies to test spatial grounding capability:
  1. per_frame       — one frame at a time (sanity check)
  2. explicit_binding — all frames with explicit frame-step mapping
  3. freeform_chain  — all frames with free-form chain prompt (GRPO target)

Example:
  python test_think_bbox_inference.py test_videos/ball.mp4 \\
    "Describe the motion trajectory of the red ball" \\
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
    --strategies per_frame explicit_binding freeform_chain
""",
    )
    parser.add_argument("video_path", type=str)
    parser.add_argument("question", type=str)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device (cuda:0, cuda:1, or 'auto' for multi-GPU)")
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--output-dir", type=str, default="outputs/think_bbox_test")
    parser.add_argument("--use-4bit", action="store_true",
                       help="Use 4-bit quantization (recommended for 7B/8B models)")
    parser.add_argument("--multi-gpu", action="store_true",
                       help="Enable multi-GPU model parallelism (auto device mapping)")
    parser.add_argument(
        "--strategies", nargs="+",
        default=["per_frame", "explicit_binding", "freeform_chain"],
        choices=["per_frame", "explicit_binding", "freeform_chain"],
    )
    args = parser.parse_args()

    if not Path(args.video_path).exists():
        print(f"ERROR: Video not found: {args.video_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect bbox format for this model
    bbox_fmt = detect_bbox_format(args.model_id)
    print(f"Model: {args.model_id}")
    print(f"Bbox format: {bbox_fmt['name']} (scale={bbox_fmt['scale']})")
    print(f"Example: {bbox_fmt['example']}")

    # Extract frames
    print(f"\nExtracting {args.num_frames} frames from {args.video_path}...")
    frames, meta = extract_frames(args.video_path, args.num_frames)
    print(f"Video: {meta['width']}x{meta['height']}, {meta['duration']:.2f}s, "
          f"{meta['total_frames']} total frames")
    print(f"Sampled at: {[f'{t:.2f}s' for t in meta['timestamps']]}")

    # Load model
    print(f"\nLoading model...")
    model, processor = load_model(
        args.model_id, 
        args.device,
        use_4bit=args.use_4bit,
        multi_gpu=args.multi_gpu or args.device == "auto"
    )
    print(f"✓ Loaded {model.__class__.__name__}")

    # Run each strategy
    all_results = {}
    for strategy in args.strategies:
        result = run_strategy(
            strategy, model, processor, frames, meta, bbox_fmt,
            args.question, args.device, args.max_tokens, output_dir,
        )
        all_results[strategy] = result

    # Comparative summary
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*80}")
    for strat, res in all_results.items():
        a = res.get("analysis", {})
        pv = a.get("pred_variation", {})
        n_steps = len(res.get("steps", []))
        n_bboxes = pv.get("count", 0)
        identical = pv.get("all_identical", True)
        x_range = pv.get("x_range", 0)
        print(f"\n  {strat:20s}: {n_steps} steps, {n_bboxes} bboxes, "
              f"all_identical={identical}, x_range={x_range:.0f}px")

    print(f"\n{'='*80}")
    print("WHAT TO DO NEXT")
    print(f"{'='*80}")

    pf = all_results.get("per_frame", {}).get("analysis", {}).get("pred_variation", {})
    ff = all_results.get("freeform_chain", {}).get("analysis", {}).get("pred_variation", {})

    if pf.get("count", 0) >= 2 and not pf.get("all_identical", True):
        print("✓ Per-frame grounding works — model CAN localize objects")
        if ff.get("count", 0) >= 2 and not ff.get("all_identical", True):
            print("✓ Free-form chain also varies — GRPO should work well")
            print("  → Proceed to training. Bboxes vary, GRPO will improve accuracy.")
        else:
            print("✗ Free-form chain degenerates — composition fails")
            print("  → Try explicit_binding strategy as the training prompt")
            print("  → Or try a larger model (7B/8B instead of 2B)")
            print("  → If still failing, consider light SFT warmup (~500 steps)")
    else:
        print("✗ Per-frame grounding fails — model lacks spatial capability")
        print("  → Use a model with stronger bbox pretrained capability")
        print("  → Qwen2.5-VL-7B-Instruct or Qwen3-VL-8B-Instruct recommended")

    # Save comparative results
    with open(output_dir / "comparative_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ All results saved to {output_dir}/")


if __name__ == "__main__":
    main()