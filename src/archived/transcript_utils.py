"""
Transcript extraction utilities for Dora videos.
Supports both manual transcript loading and automatic speech-to-text extraction.
"""
import os
import json
from typing import Optional, Dict, List, Tuple
from pathlib import Path


def load_manual_transcript(transcript_path: str) -> str:
    """
    Load transcript from a text file or JSON file.
    
    Args:
        transcript_path: Path to transcript file (.txt or .json)
        
    Returns:
        Transcript text as string
    """
    transcript_path = Path(transcript_path)
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    
    if transcript_path.suffix == '.json':
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Support different JSON formats
            if isinstance(data, dict):
                return data.get('transcript', data.get('text', ''))
            elif isinstance(data, list):
                # Assume list of segments with 'text' field
                return ' '.join([seg.get('text', '') for seg in data if isinstance(seg, dict)])
            else:
                return str(data)
    else:
        # Assume plain text
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return f.read()


def extract_transcript_from_audio(video_path: str, 
                                   model_name: str = "base",
                                   language: Optional[str] = None) -> str:
    """
    Extract transcript from video audio using Whisper.
    
    Args:
        video_path: Path to video file
        model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
        language: Language code (e.g., "en"). If None, auto-detect.
        
    Returns:
        Transcript text as string
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "Whisper not installed. Install with: pip install openai-whisper"
        )
    
    model = whisper.load_model(model_name)
    result = model.transcribe(video_path, language=language)
    return result["text"]


def extract_transcript_with_timestamps(video_path: str,
                                       model_name: str = "base",
                                       language: Optional[str] = None) -> List[Dict[str, any]]:
    """
    Extract transcript with timestamps from video audio using Whisper.
    
    Args:
        video_path: Path to video file
        model_name: Whisper model size
        language: Language code (e.g., "en"). If None, auto-detect.
        
    Returns:
        List of segments with 'text', 'start', and 'end' fields
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "Whisper not installed. Install with: pip install openai-whisper"
        )
    
    model = whisper.load_model(model_name)
    result = model.transcribe(video_path, language=language, word_timestamps=False)
    
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"]
        })
    
    return segments


def normalize_transcript(text: str) -> str:
    """
    Normalize transcript text by cleaning whitespace and common issues.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Normalized transcript text
    """
    import re
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Fix common punctuation issues
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    return text


def get_transcript(video_path: str,
                   transcript_path: Optional[str] = None,
                   use_whisper: bool = False,
                   whisper_model: str = "base",
                   language: Optional[str] = None) -> str:
    """
    Get transcript for a video using manual or automatic extraction.
    
    Args:
        video_path: Path to video file
        transcript_path: Optional path to manual transcript file
        use_whisper: If True, use Whisper for automatic extraction
        whisper_model: Whisper model size (if use_whisper=True)
        language: Language code for Whisper (if use_whisper=True)
        
    Returns:
        Normalized transcript text
    """
    if transcript_path and os.path.exists(transcript_path):
        transcript = load_manual_transcript(transcript_path)
    elif use_whisper:
        transcript = extract_transcript_from_audio(video_path, whisper_model, language)
    else:
        raise ValueError(
            "Either transcript_path must be provided or use_whisper must be True"
        )
    
    return normalize_transcript(transcript)

