#!/usr/bin/env python3
"""
Text cleaning utilities for removing non-textual characters from transcripts and answers.
"""
import re


def clean_text(text: str) -> str:
    """
    Clean text by removing non-textual characters (singing symbols, etc.).
    
    This function is used for both transcripts/context and answers to ensure
    consistent, clean text throughout the dataset.
    
    Removes:
    - Singing symbols (♪, ♫, ♩, ♬, etc.)
    - Other musical symbols
    - Non-printable characters
    - Excessive whitespace
    - Leading/trailing punctuation that's not meaningful
    
    Args:
        text: Raw text (transcript or answer)
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove singing/musical symbols
    # Common singing symbols: ♪ ♫ ♩ ♬ ♭ ♯ ♮
    musical_symbols = ['♪', '♫', '♩', '♬', '♭', '♯', '♮']
    for symbol in musical_symbols:
        text = text.replace(symbol, '')
    
    # Remove other non-printable characters except common punctuation and letters
    # Keep: letters, numbers, spaces, common punctuation (. , ! ? : ; - ' " ( ) [ ] …)
    # Remove: other unicode symbols, control characters, etc.
    text = re.sub(r'[^\w\s.,!?:;\-\'\"()\[\]…]', '', text)
    
    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = text.strip()
    
    # Remove leading/trailing punctuation that's likely not meaningful
    # But keep if it's meaningful (like "Yes!" or "No.")
    # Only remove if it's just punctuation at the start/end
    text = text.strip('.,;:')
    
    return text


def clean_answer_text(text: str) -> str:
    """
    Clean answer text (alias for clean_text for backward compatibility).
    
    Args:
        text: Raw answer text
        
    Returns:
        Cleaned answer text
    """
    return clean_text(text)


def clean_transcript_text(text: str) -> str:
    """
    Clean transcript/context text (alias for clean_text).
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned transcript text
    """
    return clean_text(text)

