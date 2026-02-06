#!/usr/bin/env python3
"""
Show examples of cleaned answers to verify text cleaning is working.
"""
import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python show_cleaned_answers.py <filtered_labels_path>")
        sys.exit(1)
    
    labels_path = Path(sys.argv[1])
    if not labels_path.exists():
        print(f"❌ File not found: {labels_path}")
        sys.exit(1)
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    print(f"Total entries: {len(labels)}\n")
    print("=" * 80)
    print("SAMPLE CLEANED ANSWERS")
    print("=" * 80)
    
    # Show first 10 answers
    for idx, entry in enumerate(labels[:10], 1):
        question = entry.get("question_text", "")[:60]
        answer = entry.get("answer_text", "")
        print(f"\n{idx}. Question: {question}...")
        print(f"   Answer: {answer}")
    
    # Check for any remaining non-textual characters
    musical_symbols = ['♪', '♫', '♩', '♬', '♭', '♯', '♮']
    entries_with_symbols = []
    for entry in labels:
        answer = entry.get("answer_text", "")
        if any(symbol in answer for symbol in musical_symbols):
            entries_with_symbols.append(entry)
    
    if entries_with_symbols:
        print(f"\n⚠️  WARNING: {len(entries_with_symbols)} entries still contain musical symbols")
        print("Sample entries with symbols:")
        for entry in entries_with_symbols[:3]:
            print(f"  - {entry.get('answer_text', '')}")
    else:
        print(f"\n✓ All answers cleaned - no musical symbols found")

if __name__ == "__main__":
    main()

