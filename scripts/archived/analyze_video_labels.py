#!/usr/bin/env python3
"""
Simple script to analyze one video and its labels.
Understanding the data structure before modifying the pipeline.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

def parse_timestamp(ts_str: str) -> float:
    """Parse timestamp string like '00:01:26,586' or '00:01:26.586' to seconds."""
    # Handle both comma and dot as decimal separator
    ts_str = ts_str.replace(',', '.')
    parts = ts_str.split(':')
    if len(parts) != 3:
        return 0.0
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def analyze_labels_file(labels_path: str) -> Dict[str, Any]:
    """Analyze a labels JSON file and return statistics."""
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    if not isinstance(labels, list):
        return {"error": "Labels file is not a JSON array"}
    
    total = len(labels)
    with_answers = sum(1 for item in labels if item.get("has_answer", False))
    without_answers = total - with_answers
    
    # Analyze timestamp formats
    timestamp_fields = [
        "question_start_ts", "question_end_ts_id", 
        "answer_start_ts_id", "answer_end_ts_id",
        "pause_start", "pause_end"
    ]
    
    # Check for missing fields
    missing_fields = {}
    for field in timestamp_fields:
        missing = sum(1 for item in labels if field not in item or not item[field])
        missing_fields[field] = missing
    
    # Sample entries
    sample_with_answer = next((item for item in labels if item.get("has_answer", False)), None)
    sample_without_answer = next((item for item in labels if not item.get("has_answer", False)), None)
    
    # Calculate pause durations
    pause_durations = []
    for item in labels:
        pause_start_str = item.get("pause_start", "")
        pause_end_str = item.get("pause_end", "")
        if pause_start_str and pause_end_str:
            try:
                pause_start = parse_timestamp(pause_start_str)
                pause_end = parse_timestamp(pause_end_str)
                if pause_end > pause_start:
                    pause_durations.append(pause_end - pause_start)
            except:
                pass
    
    return {
        "total_entries": total,
        "with_answers": with_answers,
        "without_answers": without_answers,
        "answer_rate": with_answers / total if total > 0 else 0,
        "missing_fields": missing_fields,
        "pause_durations": {
            "min": min(pause_durations) if pause_durations else None,
            "max": max(pause_durations) if pause_durations else None,
            "avg": sum(pause_durations) / len(pause_durations) if pause_durations else None,
        },
        "sample_with_answer": sample_with_answer,
        "sample_without_answer": sample_without_answer,
    }

def print_analysis(analysis: Dict[str, Any], labels_path: str):
    """Print analysis results."""
    print("=" * 80)
    print(f"ANALYSIS: {Path(labels_path).name}")
    print("=" * 80)
    
    if "error" in analysis:
        print(f"ERROR: {analysis['error']}")
        return
    
    print(f"\n📊 STATISTICS:")
    print(f"  Total entries: {analysis['total_entries']}")
    print(f"  With answers: {analysis['with_answers']} ({analysis['answer_rate']*100:.1f}%)")
    print(f"  Without answers: {analysis['without_answers']} ({100-analysis['answer_rate']*100:.1f}%)")
    
    print(f"\n⏱️  PAUSE DURATIONS:")
    pd = analysis['pause_durations']
    if pd['min'] is not None:
        print(f"  Min: {pd['min']:.2f}s")
        print(f"  Max: {pd['max']:.2f}s")
        print(f"  Avg: {pd['avg']:.2f}s")
    else:
        print("  No valid pause durations found")
    
    print(f"\n🔍 MISSING FIELDS:")
    for field, count in analysis['missing_fields'].items():
        if count > 0:
            print(f"  {field}: {count} missing")
    
    print(f"\n✅ SAMPLE ENTRY WITH ANSWER:")
    if analysis['sample_with_answer']:
        item = analysis['sample_with_answer']
        print(f"  Question: {item.get('question_text', 'N/A')[:80]}...")
        print(f"  Answer: {item.get('answer_text', 'N/A')[:80]}...")
        print(f"  Question TS: {item.get('question_start_ts', 'N/A')} → {item.get('question_end_ts_id', 'N/A')}")
        print(f"  Pause: {item.get('pause_start', 'N/A')} → {item.get('pause_end', 'N/A')}")
        print(f"  Answer TS: {item.get('answer_start_ts_id', 'N/A')} → {item.get('answer_end_ts_id', 'N/A')}")
    
    print(f"\n❌ SAMPLE ENTRY WITHOUT ANSWER:")
    if analysis['sample_without_answer']:
        item = analysis['sample_without_answer']
        print(f"  Question: {item.get('question_text', 'N/A')[:80]}...")
        print(f"  Answer: '{item.get('answer_text', '')}'")
        print(f"  Question TS: {item.get('question_start_ts', 'N/A')} → {item.get('question_end_ts_id', 'N/A')}")
        print(f"  Pause: {item.get('pause_start', 'N/A')} → {item.get('pause_end', 'N/A')}")
        print(f"  Answer TS: {item.get('answer_start_ts_id', 'N/A')} → {item.get('answer_end_ts_id', 'N/A')}")
        # Check if answer timestamp is at end of video (likely placeholder)
        answer_start = item.get('answer_start_ts_id', '')
        if '23:55' in answer_start or '23:57' in answer_start:
            print(f"  ⚠️  Answer timestamp appears to be placeholder (near end of video)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_video_labels.py <labels_json_path>")
        print("\nExample:")
        print("  python analyze_video_labels.py /mnt/work/XXXX-7/dora/dataset/processed/S1/Dora.the.Explorer.S01E01.WEBRip.Amazon_formatted.json")
        sys.exit(1)
    
    labels_path = sys.argv[1]
    
    if not Path(labels_path).exists():
        print(f"ERROR: Labels file not found: {labels_path}")
        sys.exit(1)
    
    print(f"Analyzing: {labels_path}\n")
    
    analysis = analyze_labels_file(labels_path)
    print_analysis(analysis, labels_path)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

