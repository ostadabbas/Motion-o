#!/usr/bin/env python3
"""
SRT subtitle file parser.
Extracts transcript text with timestamps from SRT files.
"""
import re
import sys
from typing import List, Dict, Tuple
from pathlib import Path
import os

# Import shared text cleaning utility
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.text_cleaning import clean_transcript_text


def parse_srt_timestamp(timestamp_str: str) -> float:
    """
    Parse SRT timestamp format (HH:MM:SS,mmm) to seconds.
    
    Args:
        timestamp_str: SRT timestamp like "00:01:26,586"
        
    Returns:
        Seconds as float
    """
    # Handle both comma and dot as decimal separator
    timestamp_str = timestamp_str.replace(',', '.')
    parts = timestamp_str.split(':')
    if len(parts) != 3:
        return 0.0
    try:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return 0.0


def parse_srt_file(srt_path: str) -> List[Dict[str, any]]:
    """
    Parse SRT subtitle file and extract segments with timestamps.
    
    Args:
        srt_path: Path to SRT file
        
    Returns:
        List of dicts with 'text', 'start', 'end' fields
    """
    srt_path = Path(srt_path)
    if not srt_path.exists():
        return []
    
    with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # SRT format:
    # 1
    # 00:00:00,000 --> 00:00:05,000
    # Subtitle text here
    # (blank line)
    
    segments = []
    
    # Split by double newlines (subtitle blocks)
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # First line is sequence number (skip)
        # Second line is timestamp
        timestamp_line = lines[1].strip()
        
        # Parse timestamp: "00:00:00,000 --> 00:00:05,000"
        match = re.match(r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})', timestamp_line)
        if not match:
            continue
        
        start_time = parse_srt_timestamp(match.group(1))
        end_time = parse_srt_timestamp(match.group(2))
        
        # Remaining lines are subtitle text
        text_lines = [line.strip() for line in lines[2:] if line.strip()]
        text = ' '.join(text_lines)
        
        # Clean text to remove singing symbols and non-textual characters
        text = clean_transcript_text(text)
        
        if text:
            segments.append({
                'text': text,
                'start': start_time,
                'end': end_time
            })
    
    return segments


def extract_transcript_up_to_time(srt_path: str, end_time: float) -> str:
    """
    Extract transcript text from SRT file up to a given timestamp.
    
    Args:
        srt_path: Path to SRT file
        end_time: End timestamp in seconds
        
    Returns:
        Combined transcript text up to end_time
    """
    segments = parse_srt_file(srt_path)
    
    # Collect all segments that end before or at end_time
    transcript_parts = []
    for seg in segments:
        if seg['end'] <= end_time:
            transcript_parts.append(seg['text'])
    
    return ' '.join(transcript_parts)


def extract_transcript_window(srt_path: str, question_start: float, window_before: float = 20.0) -> str:
    """
    Extract transcript text from SRT file using a time window before the question.
    Context stops BEFORE question starts - pure information only.
    
    Args:
        srt_path: Path to SRT file
        question_start: Question start timestamp (context stops BEFORE this)
        window_before: Seconds before question_start to include (default 20.0)
        
    Returns:
        Combined transcript text from the time window (stops before question)
    """
    segments = parse_srt_file(srt_path)
    
    # Window: (question_start - window_before) to question_start
    # Context stops BEFORE question starts
    start_time = max(0.0, question_start - window_before)
    end_time = question_start  # Stop before question
    
    # Collect segments that end BEFORE question_start
    # Segment must end strictly before question starts, and start within our window
    transcript_parts = []
    for seg in segments:
        # Segment must end strictly before question starts
        # And start within our window
        if seg['end'] < end_time and seg['start'] >= start_time:
            seg_text = seg['text'].strip()
            if seg_text:
                transcript_parts.append(seg_text)
    
    transcript = ' '.join(transcript_parts)
    
    # Check if there's content before our window (indicates truncation from beginning)
    # If start_time > 0, there's content before the window, so add "..."
    has_content_before = start_time > 0.0
    
    # Truncate from beginning if too long (keep recent context, like dataset generator)
    max_length = 1500  # Max length for visualization
    if len(transcript) > max_length:
        truncated = transcript[-max_length:]
        if max_length > 10:
            truncated = "..." + truncated[3:]
        return truncated
    
    # Add "..." prefix if there's content before the window (matches training protocol)
    # This indicates we're showing a window, not the full transcript from start
    if has_content_before and transcript:
        return "..." + transcript
    
    return transcript


def find_srt_file(video_path: str, srt_base_dir: str = "/mnt/work/XXXX-7/dora/dataset/srt") -> str:
    """
    Find corresponding SRT file for a video.
    
    Args:
        video_path: Path to video file
        srt_base_dir: Base directory for SRT files
        
    Returns:
        Path to SRT file if found, empty string otherwise
    """
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem.lower()
    
    # First, check if SRT file exists in same directory as video
    srt_in_same_dir = video_path_obj.parent / f"{video_path_obj.stem}.srt"
    if srt_in_same_dir.exists():
        return str(srt_in_same_dir)
    
    # Try to match video name with SRT files
    # Video: "Dora.the.explorer.s01e01.avi"
    # SRT might be: "Dora.the.Explorer.S01E01.WEBRip.Amazon.srt" or similar
    
    srt_base = Path(srt_base_dir)
    if not srt_base.exists():
        return ""
    
    # Extract season/episode info
    season_match = re.search(r's(\d+)e(\d+)', video_name, re.IGNORECASE)
    if not season_match:
        season = video_path.split("/")[-2]
        ep = video_path.split("/")[-1]
        lbl_pth = srt_base_dir
        if season == "S1" or season == "S2":
            if len(ep.split(".")) < 4:
                ep_idx = "s02e"+ep.split(".")[0].split('-')[-1]
            else:
                ep_idx = ep.split(".")[3]
            srt_name = "Dora.the.Explorer." + ep_idx.upper() + ".WEBRip.Amazon.srt"
            return os.path.join(lbl_pth, season, srt_name)
        elif season == "S3" or season == "S4" or season == "S5":
            assert os.path.exists(os.path.join(lbl_pth, season, ep.replace(".avi", ".srt")))
            return os.path.join(lbl_pth, season, ep.replace(".avi", ".srt"))
        else:
            print("garama!", season)
    else:
        print("\ndid it happen\n")
    
    season = season_match.group(1)
    episode = season_match.group(2)
    
    # Look in season directory - try both S1 and S01 formats
    # First try without zero-padding (S1, S2, etc.)
    season_dir = srt_base / f"S{int(season)}"
    if not season_dir.exists():
        # Try with zero-padded season (S01, S02, etc.)
        season_dir = srt_base / f"S{int(season):02d}"
        if not season_dir.exists():
            print("Season dir not exist")
            return ""
    
    # Try to find matching SRT file
    # Match patterns like: *S01E01*, *s01e01*, *01-01*, etc.
    patterns = [
        f"*S{season}E{episode}*",
        f"*s{season}e{episode}*",
        f"*S{int(season):02d}E{int(episode):02d}*",
        f"*s{int(season):02d}e{int(episode):02d}*",
        f"*{season}-{episode}*",
        f"*{int(season):02d}-{int(episode):02d}*",
    ]
    
    for pattern in patterns:
        matches = list(season_dir.glob(f"{pattern}.srt"))
        if matches:
            return str(matches[0])
    
    # Also try matching by season/episode in filename
    season_episode_pattern = f"S{season}E{episode}"
    for srt_file in season_dir.glob("*.srt"):
        srt_name = srt_file.stem
        # Check if season/episode pattern appears in SRT filename (case insensitive)
        if season_episode_pattern.lower() in srt_name.lower():
            return str(srt_file)
    print("SRT not found")
    return ""

