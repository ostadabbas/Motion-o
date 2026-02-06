"""
Q&A segmentation utilities for Dora videos.
Detects question → pause → answer structure using audio analysis and pattern matching.
"""
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None


@dataclass
class QASegment:
    """Represents a Q&A segment with timestamps.
    
    Structure: [Information/Context] → Question → [Pause] → Answer
    The transcript includes all context before the question.
    """
    question_start: float
    pause_start: float
    pause_end: float
    answer_end: float
    question: str
    answer: str
    transcript: str  # Full transcript including context, question, and answer
    confidence: float = 1.0


# Common question patterns in Dora videos
QUESTION_PATTERNS = [
    r"how many\s+",
    r"what (is|are|do|does|can|will)\s+",
    r"where (is|are|do|does|can|will)\s+",
    r"when (is|are|do|does|can|will)\s+",
    r"who (is|are|do|does|can|will)\s+",
    r"why (is|are|do|does|can|will)\s+",
    r"can you\s+",
    r"do you\s+",
    r"which (one|ones|is|are)\s+",
    r"how (do|does|can|will)\s+",
]


def detect_question_in_text(text: str) -> Optional[Tuple[str, float]]:
    """
    Detect if text contains a question and extract it.
    
    Args:
        text: Text to analyze
        
    Returns:
        Tuple of (question_text, confidence) or None if no question found
    """
    text_lower = text.lower().strip()
    
    # Check for question mark
    if '?' in text:
        # Extract sentence ending with question mark
        sentences = re.split(r'[.!?]', text)
        for sent in sentences:
            if '?' in sent:
                sent = sent.strip()
                if len(sent) > 5:  # Minimum question length
                    return (sent, 0.9)
    
    # Check for question patterns
    for pattern in QUESTION_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            # Extract sentence containing the pattern
            start = match.start()
            # Find sentence boundaries
            text_before = text[:start]
            sentence_start = max(0, text_before.rfind('.') + 1, text_before.rfind('!') + 1)
            sentence_end = text.find('.', start)
            if sentence_end == -1:
                sentence_end = text.find('!', start)
            if sentence_end == -1:
                sentence_end = len(text)
            
            question = text[sentence_start:sentence_end].strip()
            if len(question) > 5:
                return (question, 0.7)
    
    return None


def detect_pauses_in_audio(audio_path: str,
                           silence_threshold: float = -40.0,
                           min_silence_duration: float = 1.0,
                           hop_length: int = 512) -> List[Tuple[float, float]]:
    """
    Detect silence/pause periods in audio file.
    
    Args:
        audio_path: Path to audio file
        silence_threshold: dB threshold for silence (negative value)
        min_silence_duration: Minimum duration in seconds to consider as pause
        hop_length: Hop length for audio analysis
        
    Returns:
        List of (start_time, end_time) tuples for detected pauses
    """
    if librosa is None or sf is None:
        raise ImportError("librosa and soundfile required for audio analysis")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Compute RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    # Convert to dB
    rms_db = librosa.power_to_db(rms**2, ref=np.max)
    
    # Find silence regions
    silence_mask = rms_db < silence_threshold
    
    # Convert frame indices to time
    times = librosa.frames_to_time(np.arange(len(silence_mask)), sr=sr, hop_length=hop_length)
    
    # Find continuous silence regions
    pauses = []
    in_silence = False
    silence_start = 0.0
    
    for i, is_silent in enumerate(silence_mask):
        if is_silent and not in_silence:
            silence_start = times[i]
            in_silence = True
        elif not is_silent and in_silence:
            silence_end = times[i]
            duration = silence_end - silence_start
            if duration >= min_silence_duration:
                pauses.append((silence_start, silence_end))
            in_silence = False
    
    # Handle case where audio ends in silence
    if in_silence:
        silence_end = times[-1]
        duration = silence_end - silence_start
        if duration >= min_silence_duration:
            pauses.append((silence_start, silence_end))
    
    return pauses


def extract_audio_from_video(video_path: str, output_audio_path: Optional[str] = None) -> str:
    """
    Extract audio track from video file.
    
    Args:
        video_path: Path to video file
        output_audio_path: Optional path to save audio. If None, uses temp file.
        
    Returns:
        Path to extracted audio file
    """
    import tempfile
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip
    
    if output_audio_path is None:
        output_audio_path = tempfile.mktemp(suffix='.wav')
    
    clip = VideoFileClip(video_path)
    # Try with verbose parameter (older moviepy), fallback without it (newer versions)
    try:
        clip.audio.write_audiofile(output_audio_path, verbose=False, logger=None)
    except TypeError:
        # Newer moviepy versions don't support verbose parameter
        clip.audio.write_audiofile(output_audio_path)
    clip.close()
    
    return output_audio_path


def segment_qa_from_transcript(transcript: str,
                                timestamps: Optional[List[Dict[str, Any]]] = None,
                                pauses: Optional[List[Tuple[float, float]]] = None,
                                question_window_before: float = 10.0,
                                answer_window_after: float = 10.0) -> List[QASegment]:
    """
    Generic Q&A segmentation using pauses as anchors.
    Structure: [Information] → Question → [Pause] → Answer
    
    This approach is generic because it:
    1. Uses pauses as primary signal (natural structure in educational videos)
    2. Finds question temporally closest to pause (within window before)
    3. Finds answer temporally closest after pause (within window after)
    4. Doesn't rely on specific question patterns
    
    Args:
        transcript: Full transcript text
        timestamps: Optional list of transcript segments with timestamps
        pauses: Optional list of (start, end) pause timestamps
        question_window_before: Seconds before pause to search for question
        answer_window_after: Seconds after pause to search for answer
        
    Returns:
        List of QASegment objects
    """
    segments = []
    
    # If we have pauses, use them as anchors (most generic approach)
    if pauses and timestamps:
        for pause_start, pause_end in pauses:
            # Find question: look for segments before the pause
            question_seg = None
            question_start = None
            question_end = None
            
            # Search backwards from pause for question
            for seg in reversed(timestamps):
                seg_start = seg.get('start', 0.0)
                seg_end = seg.get('end', 0.0)
                seg_text = seg.get('text', '').strip()
                
                # Check if segment is within question window before pause
                if seg_end <= pause_start and seg_start >= (pause_start - question_window_before):
                    # Check if it looks like a question (has question mark or question patterns)
                    if '?' in seg_text or detect_question_in_text(seg_text):
                        question_seg = seg
                        question_start = seg_start
                        question_end = seg_end
                        break
            
            # If no question found with pattern, take the last segment before pause
            if question_seg is None:
                for seg in reversed(timestamps):
                    seg_start = seg.get('start', 0.0)
                    seg_end = seg.get('end', 0.0)
                    if seg_end <= pause_start and seg_start >= (pause_start - question_window_before):
                        question_seg = seg
                        question_start = seg_start
                        question_end = seg_end
                        break
            
            # Find answer: look for segments after the pause
            answer_seg = None
            answer_start = None
            answer_end = None
            
            # Search forwards from pause for answer
            for seg in timestamps:
                seg_start = seg.get('start', 0.0)
                seg_end = seg.get('end', 0.0)
                seg_text = seg.get('text', '').strip()
                
                # Check if segment is within answer window after pause
                if seg_start >= pause_end and seg_end <= (pause_end + answer_window_after):
                    # Take first substantial segment after pause as answer
                    if len(seg_text) > 3:  # Minimum length
                        answer_seg = seg
                        answer_start = seg_start
                        answer_end = seg_end
                        break
            
            # Build context transcript (all segments from start to question, NOT including answer)
            context_segments = []
            for seg in timestamps:
                seg_start = seg.get('start', 0.0)
                seg_end = seg.get('end', 0.0)
                if seg_end <= question_end if question_end else pause_start:
                    context_segments.append(seg.get('text', '').strip())
            
            question_text = question_seg.get('text', '').strip() if question_seg else ""
            answer_text = answer_seg.get('text', '').strip() if answer_seg else ""
            # Transcript should include context + question, but NOT the answer
            # (Answer is stored separately and only used as training target)
            full_transcript = ' '.join(context_segments + [question_text])
            
            if question_text:  # Only create segment if we found a question
                segments.append(QASegment(
                    question_start=question_start if question_start else pause_start - 2.0,
                    pause_start=pause_start,
                    pause_end=pause_end,
                    answer_end=answer_end if answer_end else pause_end + 5.0,
                    question=question_text,
                    answer=answer_text,
                    transcript=full_transcript,
                    confidence=0.8 if answer_text else 0.5
                ))
    
    # Fallback: if we have timestamps but no pauses, use question detection
    elif timestamps:
        for i, seg in enumerate(timestamps):
            text = seg.get('text', '')
            start = seg.get('start', 0.0)
            end = seg.get('end', 0.0)
            
            # Check if this segment contains a question
            question_info = detect_question_in_text(text)
            if question_info:
                question_text, confidence = question_info
                
                # Find corresponding pause and answer
                pause_start = end
                pause_end = pause_start + 2.0  # Default pause duration
                answer_end = pause_end + 5.0   # Default answer duration
                
                # Extract answer from next segment(s)
                answer_text = ""
                if i + 1 < len(timestamps):
                    answer_seg = timestamps[i + 1]
                    answer_text = answer_seg.get('text', '').strip()
                    answer_end = answer_seg.get('end', pause_end + 5.0)
                
                segments.append(QASegment(
                    question_start=start,
                    pause_start=pause_start,
                    pause_end=pause_end,
                    answer_end=answer_end,
                    question=question_text,
                    answer=answer_text,
                    transcript=text,
                    confidence=confidence
                ))
    
    # Fallback: simple text-based segmentation (no timestamps)
    else:
        sentences = re.split(r'[.!?]', transcript)
        current_question = None
        question_start = 0.0
        
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
            
            question_info = detect_question_in_text(sent)
            if question_info:
                if current_question:
                    segments.append(QASegment(
                        question_start=question_start,
                        pause_start=question_start + len(current_question) * 0.1,
                        pause_end=question_start + len(current_question) * 0.1 + 2.0,
                        answer_end=question_start + len(current_question) * 0.1 + 4.0,
                        question=current_question,
                        answer="",
                        transcript=current_question,
                        confidence=0.5
                    ))
                
                current_question, confidence = question_info
                question_start = i * 2.0
            elif current_question:
                segments.append(QASegment(
                    question_start=question_start,
                    pause_start=question_start + len(current_question) * 0.1,
                    pause_end=question_start + len(current_question) * 0.1 + 2.0,
                    answer_end=question_start + len(current_question) * 0.1 + 4.0,
                    question=current_question,
                    answer=sent,
                    # Transcript should NOT include answer (only context + question)
                    transcript=current_question,
                    confidence=0.6
                ))
                current_question = None
    
    return segments


def load_manual_segments(segments_path: str) -> List[QASegment]:
    """
    Load manually specified Q&A segments from JSON/YAML file.
    
    Args:
        segments_path: Path to segments file
        
    Returns:
        List of QASegment objects
    """
    import json
    from pathlib import Path
    
    path = Path(segments_path)
    if path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files. Install with: pip install PyYAML")
    
    segments = []
    for item in data:
        segments.append(QASegment(
            question_start=item.get('question_start', 0.0),
            pause_start=item.get('pause_start', 0.0),
            pause_end=item.get('pause_end', 0.0),
            answer_end=item.get('answer_end', 0.0),
            question=item.get('question', ''),
            answer=item.get('answer', ''),
            transcript=item.get('transcript', ''),
            confidence=item.get('confidence', 1.0)
        ))
    
    return segments

