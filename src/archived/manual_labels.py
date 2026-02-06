"""
Utilities for loading manually labeled Q&A segments.
"""
import json
from typing import List, Optional
from pathlib import Path
from .qa_segmenter import QASegment
from .transcript_utils import extract_transcript_with_timestamps


def load_manual_labels(labels_path: str, video_path: str = None, extract_transcript_if_missing: bool = True) -> List[QASegment]:
    """
    Load manually labeled Q&A segments from JSON file.
    
    Supports two formats:
    1. Old format (single video):
    {
        "video_path": "path/to/video.mp4",
           "segments": [...]
       }
    
    2. New unified format (multiple videos):
       {
           "videos": [
               {"video_path": "path/to/video1.mp4", "segments": [...]},
               {"video_path": "path/to/video2.mp4", "segments": [...]}
           ]
       }
    
    If video_path is provided, loads segments for that specific video.
    Otherwise, loads all segments from all videos.
    
    Args:
        labels_path: Path to labels JSON file
        video_path: Optional video path (filters to that video in unified format)
        extract_transcript_if_missing: If True, extract transcript using Whisper if missing
        
    Returns:
        List of QASegment objects
    """
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle old format (backward compatibility)
    if "video_path" in data and "segments" in data:
        videos_data = [{"video_path": data["video_path"], "segments": data["segments"]}]
    # Handle new unified format
    elif "videos" in data:
        videos_data = data["videos"]
    else:
        raise ValueError(f"Invalid labels file format: {labels_path}")
    
    # Filter by video_path if provided
    if video_path:
        video_name = Path(video_path).name
        videos_data = [
            v for v in videos_data 
            if Path(v.get("video_path", "")).name == video_name
        ]
        if not videos_data:
            raise ValueError(f"Video '{video_path}' not found in labels file")
    
    all_segments = []
    
    # Process each video
    for video_entry in videos_data:
        video_entry_path = video_entry.get("video_path", "")
        final_video_path = video_path if video_path else video_entry_path
        segments_data = video_entry.get("segments", [])
        
        if not segments_data:
            continue
        
        # Extract transcript with timestamps if needed
        transcript_timestamps = None
        if extract_transcript_if_missing and final_video_path and Path(final_video_path).exists():
            # Check if any segment is missing transcript
            needs_transcript = any(not seg_data.get("transcript", "").strip() 
                                  for seg_data in segments_data)
            if needs_transcript:
                print(f"Extracting transcript from {Path(final_video_path).name}...")
                try:
                    transcript_timestamps = extract_transcript_with_timestamps(
                        final_video_path,
                        model_name="base",
                        language=None
                    )
                    print(f"✓ Extracted transcript with {len(transcript_timestamps)} segments")
                except Exception as e:
                    print(f"Warning: Could not extract transcript: {e}")
                    transcript_timestamps = None
        
        # Process segments for this video
        for seg_data in segments_data:
            transcript = seg_data.get("transcript", "").strip()
            
            # If transcript is missing and we have timestamps, build it from context
            if not transcript and transcript_timestamps:
                question_end = float(seg_data.get("question_start", 0.0))
                pause_start = float(seg_data.get("pause_start", 0.0))
                
                # Collect all transcript segments up to the question
                context_segments = []
                for ts_seg in transcript_timestamps:
                    seg_start = ts_seg.get('start', 0.0)
                    seg_end = ts_seg.get('end', 0.0)
                    if seg_end <= pause_start:  # Up to pause (includes question)
                        context_segments.append(ts_seg.get('text', '').strip())
                
                # Add question text
                question_text = seg_data.get("question", "").strip()
                if question_text:
                    context_segments.append(question_text)
                
                transcript = ' '.join(context_segments)
                print(f"✓ Built transcript for segment (length: {len(transcript)})")
            
            segment = QASegment(
                question_start=float(seg_data.get("question_start", 0.0)),
                pause_start=float(seg_data.get("pause_start", 0.0)),
                pause_end=float(seg_data.get("pause_end", 0.0)),
                answer_end=float(seg_data.get("answer_end", 0.0)),
                question=seg_data.get("question", "").strip(),
                answer=seg_data.get("answer", "").strip(),
                transcript=transcript,
                confidence=1.0  # Manual labels are always high confidence
            )
            all_segments.append(segment)
    
    return all_segments

