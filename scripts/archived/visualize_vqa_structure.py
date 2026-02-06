#!/usr/bin/env python3
"""
Visualize VQA structure: Information, Question, Pause, Answer with video frames.
"""
import json
import sys
from pathlib import Path
from typing import Optional, Tuple
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Import SRT parser
sys.path.insert(0, str(Path(__file__).parent))
try:
    from srt_parser import find_srt_file, extract_transcript_window
except ImportError:
    # Fallback if import fails
    def find_srt_file(video_path: str, srt_base_dir: str = "/mnt/data/dora/srt") -> str:
        return ""
    def extract_transcript_window(srt_path: str, end_time: float, window_before: float = 10.0) -> str:
        return ""

def parse_timestamp(ts_str: str) -> float:
    """Parse timestamp string to seconds."""
    if not ts_str or not ts_str.strip():
        return 0.0
    ts_str = ts_str.replace(',', '.')
    parts = ts_str.split(':')
    if len(parts) != 3:
        return 0.0
    try:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return 0.0

def extract_frame(video_path: str, timestamp: float, width: int = 320, height: int = 240) -> Optional[Image.Image]:
    """Extract a frame from video at given timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return None
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Resize maintaining aspect ratio
    pil_image.thumbnail((width, height), Image.Resampling.LANCZOS)
    
    return pil_image

def get_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """Get Helvetica-like font (DejaVu Sans or similar), fallback to default if not available."""
    import subprocess
    import os
    
    # Try to find DejaVu Sans (Helvetica-like) using fc-list
    try:
        result = subprocess.run(['fc-list', ':family=DejaVu Sans'], 
                               capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            # Try common paths for DejaVu Sans
            dejavu_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            ]
            for font_path in dejavu_paths:
                if os.path.exists(font_path):
                    try:
                        return ImageFont.truetype(font_path, size)
                    except:
                        continue
    except:
        pass
    
    # Try Helvetica on different systems
    helvetica_paths = [
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux (Helvetica-like)
        "arial.ttf",  # Windows
        "/Windows/Fonts/arial.ttf",  # Windows
    ]
    
    for font_path in helvetica_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    
    # Last resort: default font
    return ImageFont.load_default()

def wrap_text(text: str, max_width: int, font: ImageFont.FreeTypeFont) -> list:
    """Wrap text to fit within max_width."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = font.getbbox(test_line)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def create_vqa_visualization(
    video_path: str,
    entry: dict,
    transcript: str = "",
    output_path: Optional[str] = None,
    frame_width: int = 240,
    frame_height: int = 180,
    srt_base_dir: str = "/mnt/work/XXXX-7/dora/dataset/srt"
) -> Image.Image:
    """
    Create horizontal visualization of VQA structure.
    
    Args:
        video_path: Path to video file
        entry: Label entry with timestamps and text
        transcript: Context/transcript text
        output_path: Optional path to save image
        frame_width: Width of extracted frames
        frame_height: Height of extracted frames
    """
    # Parse timestamps
    question_start = parse_timestamp(entry.get("question_start_ts", ""))
    question_end = parse_timestamp(entry.get("question_end_ts_id", ""))
    pause_start = parse_timestamp(entry.get("pause_start", ""))
    pause_end = parse_timestamp(entry.get("pause_end", ""))
    answer_start = parse_timestamp(entry.get("answer_start_ts_id", ""))
    answer_end = parse_timestamp(entry.get("answer_end_ts_id", ""))
    
    question_text = entry.get("question_text", "")
    answer_text = entry.get("answer_text", "")
    
    # Get transcript from SRT file if available
    # Context stops BEFORE question starts - use question_start_ts from processed JSON
    if not transcript:
        question_start = parse_timestamp(entry.get("question_start_ts", ""))
        srt_path = find_srt_file(video_path, srt_base_dir)
        if srt_path and question_start > 0:
            try:
                # Use 20 second window before question_start
                # Context ends BEFORE question starts (pure information only)
                transcript = extract_transcript_window(srt_path, question_start, window_before=20.0)
                if transcript:
                    print(f"  Using transcript window from SRT (stops before question): {srt_path}")
            except Exception as e:
                print(f"  Warning: Could not load SRT transcript: {e}")
    
    # Extract frames at key timestamps
    # Information frame: just before question (or start of video)
    info_timestamp = max(0, question_start - 2.0)  # 2 seconds before question
    info_frame = extract_frame(video_path, info_timestamp, frame_width, frame_height)
    
    # Question frame: middle of question
    question_timestamp = (question_start + question_end) / 2
    question_frame = extract_frame(video_path, question_timestamp, frame_width, frame_height)
    
    # Pause frame: middle of pause
    pause_timestamp = (pause_start + pause_end) / 2
    pause_frame = extract_frame(video_path, pause_timestamp, frame_width, frame_height)
    
    # Answer frame: middle of answer
    answer_timestamp = (answer_start + answer_end) / 2 if answer_start > 0 else pause_end + 1.0
    answer_frame = extract_frame(video_path, answer_timestamp, frame_width, frame_height)
    
    # Create blank canvas - HORIZONTAL layout
    padding = 8  # Reduced padding between sections
    section_width = frame_width + 250  # Frame + text area (reduced)
    section_height = frame_height + 120  # Frame + text space
    
    total_width = section_width * 4 + padding * 5  # 4 sections horizontally
    total_height = section_height + padding * 2
    
    img = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Fonts
    label_font = get_font(16)
    text_font = get_font(13)
    small_font = get_font(11)
    
    x_pos = padding
    
    # Section 1: Information/Context
    section_x = x_pos
    frame_y = padding + 25  # Space for label (reduced)
    
    # Draw label
    draw.text((section_x, padding), "INFORMATION / CONTEXT", fill='black', font=label_font)
    
    # Draw frame
    draw.rectangle([section_x, frame_y, section_x + frame_width, frame_y + frame_height], 
                   outline='black', width=2)
    if info_frame:
        img.paste(info_frame, (section_x, frame_y))
    
    # Draw text below frame
    text_y = frame_y + frame_height + 10
    text_area_width = frame_width - 10
    
    if transcript:
        # Show more text - up to 1500 chars (already truncated in extract_transcript_window)
        # Display more lines to show full context
        transcript_lines = wrap_text(transcript, text_area_width, text_font)
        for line in transcript_lines[:8]:  # Max 8 lines to show more context
            draw.text((section_x + 5, text_y), line, fill='black', font=text_font)
            text_y += 18
    else:
        draw.text((section_x + 5, text_y), "[Context]", fill='gray', font=text_font)
    
    x_pos += section_width + padding
    
    # Section 2: Question
    section_x = x_pos
    frame_y = padding + 25
    
    # Draw label
    draw.text((section_x, padding), "QUESTION", fill='blue', font=label_font)
    
    # Draw frame
    draw.rectangle([section_x, frame_y, section_x + frame_width, frame_y + frame_height], 
                   outline='blue', width=2)
    if question_frame:
        img.paste(question_frame, (section_x, frame_y))
    
    # Draw text below frame (truncate if too long)
    text_y = frame_y + frame_height + 10
    max_chars = 100
    display_question = question_text[:max_chars] + "..." if len(question_text) > max_chars else question_text
    question_lines = wrap_text(display_question, text_area_width, text_font)
    for line in question_lines[:3]:  # Max 3 lines
        draw.text((section_x + 5, text_y), line, fill='black', font=text_font)
        text_y += 18
    
    x_pos += section_width + padding
    
    # Section 3: Pause
    section_x = x_pos
    frame_y = padding + 25
    
    # Draw label
    draw.text((section_x, padding), "PAUSE", fill='orange', font=label_font)
    
    # Draw frame
    draw.rectangle([section_x, frame_y, section_x + frame_width, frame_y + frame_height], 
                   outline='orange', width=2)
    if pause_frame:
        img.paste(pause_frame, (section_x, frame_y))
    
    # Draw text below frame
    text_y = frame_y + frame_height + 10
    pause_duration = pause_end - pause_start
    draw.text((section_x + 5, text_y), f"Waiting...", fill='gray', font=text_font)
    text_y += 18
    draw.text((section_x + 5, text_y), f"({pause_duration:.1f}s)", fill='gray', font=small_font)
    
    x_pos += section_width + padding
    
    # Section 4: Answer
    section_x = x_pos
    frame_y = padding + 25
    
    # Draw label
    draw.text((section_x, padding), "ANSWER", fill='green', font=label_font)
    
    # Draw frame
    draw.rectangle([section_x, frame_y, section_x + frame_width, frame_y + frame_height], 
                   outline='green', width=2)
    if answer_frame:
        img.paste(answer_frame, (section_x, frame_y))
    
    # Draw text below frame (truncate if too long)
    text_y = frame_y + frame_height + 10
    if answer_text:
        max_chars = 100
        display_answer = answer_text[:max_chars] + "..." if len(answer_text) > max_chars else answer_text
        answer_lines = wrap_text(display_answer, text_area_width, text_font)
        for line in answer_lines[:3]:  # Max 3 lines
            draw.text((section_x + 5, text_y), line, fill='black', font=text_font)
            text_y += 18
    else:
        draw.text((section_x + 5, text_y), "[No answer]", fill='gray', font=text_font)
    
    # Save if output path provided
    if output_path:
        img.save(output_path)
        print(f"✓ Saved visualization to: {output_path}")
    
    return img

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize VQA structure with frames and text")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("labels_path", type=str, help="Path to labels JSON file")
    parser.add_argument("--entry-index", type=int, default=0, 
                       help="Index of entry to visualize (default: 0)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output image path (default: vqa_visualization.png)")
    parser.add_argument("--transcript", type=str, default="",
                       help="Context/transcript text (optional)")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    labels_path = Path(args.labels_path)
    
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    if not labels_path.exists():
        print(f"❌ Labels file not found: {labels_path}")
        sys.exit(1)
    
    # Load labels
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    if not isinstance(labels, list):
        print(f"❌ Labels file is not a JSON array")
        sys.exit(1)
    
    if args.entry_index >= len(labels):
        print(f"❌ Entry index {args.entry_index} out of range (total: {len(labels)})")
        sys.exit(1)
    
    entry = labels[args.entry_index]
    
    # Use transcript from entry if available, otherwise use provided transcript
    # SRT will be loaded automatically in create_vqa_visualization if not provided
    transcript = entry.get("transcript", "") or args.transcript
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"vqa_visualization_entry_{args.entry_index}.png"
    
    print(f"Creating visualization for entry {args.entry_index}...")
    print(f"Question: {entry.get('question_text', '')[:60]}...")
    
    # Create visualization
    img = create_vqa_visualization(
        str(video_path),
        entry,
        transcript=transcript,
        output_path=output_path
    )
    
    print(f"✓ Visualization created: {output_path}")
    print(f"  Size: {img.size[0]}x{img.size[1]} pixels")

if __name__ == "__main__":
    main()

