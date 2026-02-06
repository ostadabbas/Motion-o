#!/usr/bin/env python3
"""
Validate and filter label files.
Removes entries without answers and invalid timestamps.
"""
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Import shared text cleaning utility
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.text_cleaning import clean_text

def parse_timestamp(ts_str: str) -> float:
    """Parse timestamp string like '00:01:26,586' or '00:01:26.586' to seconds."""
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

def is_placeholder_timestamp(ts_str: str) -> bool:
    """Check if timestamp is a placeholder (near end of video)."""
    if not ts_str:
        return True
    # Check for common placeholder patterns (23:55, 23:57, etc.)
    return '23:55' in ts_str or '23:57' in ts_str or '23:59' in ts_str

def clean_answer_text(text: str) -> str:
    """
    Clean answer text by removing non-textual characters.
    
    Uses shared clean_text utility for consistency.
    
    Args:
        text: Raw answer text
        
    Returns:
        Cleaned answer text
    """
    return clean_text(text)

def validate_entry(entry: Dict[str, Any], index: int) -> Tuple[bool, List[str]]:
    """
    Validate a single entry.
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required fields
    required_fields = ["question_text", "question_start_ts", "pause_start", "pause_end"]
    for field in required_fields:
        if field not in entry or not entry[field]:
            issues.append(f"Missing required field: {field}")
            return False, issues
    
    # Check question text
    question_text = entry.get("question_text", "").strip()
    if not question_text:
        issues.append("Empty question_text")
        return False, issues
    
    # Parse timestamps
    question_start = parse_timestamp(entry.get("question_start_ts", ""))
    question_end = parse_timestamp(entry.get("question_end_ts_id", ""))
    pause_start = parse_timestamp(entry.get("pause_start", ""))
    pause_end = parse_timestamp(entry.get("pause_end", ""))
    answer_start = parse_timestamp(entry.get("answer_start_ts_id", ""))
    answer_end = parse_timestamp(entry.get("answer_end_ts_id", ""))
    
    # Check if timestamps are valid
    if question_start == 0.0 and question_end == 0.0:
        issues.append("Invalid question timestamps")
        return False, issues
    
    # Check timestamp order
    if question_end < question_start:
        issues.append(f"Question end ({question_end:.2f}s) < start ({question_start:.2f}s)")
    
    if pause_end < pause_start:
        issues.append(f"Pause end ({pause_end:.2f}s) < start ({pause_start:.2f}s)")
    
    if answer_start > 0 and answer_end > 0 and answer_end < answer_start:
        issues.append(f"Answer end ({answer_end:.2f}s) < start ({answer_start:.2f}s)")
    
    # Check for placeholder timestamps in answer
    has_answer = entry.get("has_answer", False)
    answer_text = entry.get("answer_text", "").strip()
    
    if has_answer:
        if not answer_text:
            issues.append("has_answer=True but answer_text is empty")
        if is_placeholder_timestamp(entry.get("answer_start_ts_id", "")):
            issues.append("has_answer=True but answer timestamp is placeholder")
    
    # Check if entry has valid answer
    has_valid_answer = (
        has_answer and 
        answer_text and 
        not is_placeholder_timestamp(entry.get("answer_start_ts_id", "")) and
        answer_start > 0 and
        answer_end > answer_start
    )
    
    # Check reasonable pause duration (not too long, likely placeholder)
    pause_duration = pause_end - pause_start
    if pause_duration > 600:  # More than 10 minutes is suspicious
        issues.append(f"Pause duration too long ({pause_duration:.1f}s), likely placeholder")
    
    # If there are critical issues, mark as invalid
    critical_issues = [
        "Pause end" in issue and "< start" in issue for issue in issues
    ]
    if any(critical_issues):
        return False, issues
    
    return True, issues

def filter_labels(labels: List[Dict[str, Any]], 
                  filter_no_answer: bool = True,
                  filter_invalid_timestamps: bool = True,
                  filter_placeholders: bool = True) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Filter labels based on validation criteria.
    
    Args:
        labels: List of label entries
        filter_no_answer: Filter out entries without valid answers
        filter_invalid_timestamps: Filter out entries with invalid timestamp order
        filter_placeholders: Filter out entries with placeholder timestamps
    
    Returns:
        (filtered_labels, statistics)
    """
    stats = {
        "total": len(labels),
        "valid_with_answer": 0,
        "valid_without_answer": 0,
        "invalid": 0,
        "filtered_no_answer": 0,
        "filtered_invalid_timestamps": 0,
        "filtered_placeholders": 0,
        "issues_by_type": defaultdict(int),
    }
    
    filtered = []
    
    for idx, entry in enumerate(labels):
        is_valid, issues = validate_entry(entry, idx)
        
        # Count issues by type
        for issue in issues:
            # Extract issue type
            if "Missing" in issue:
                stats["issues_by_type"]["missing_fields"] += 1
            elif "Pause end" in issue and "< start" in issue:
                stats["issues_by_type"]["invalid_pause_timestamps"] += 1
            elif "placeholder" in issue.lower():
                stats["issues_by_type"]["placeholder_timestamps"] += 1
            elif "Empty" in issue:
                stats["issues_by_type"]["empty_fields"] += 1
            else:
                stats["issues_by_type"]["other"] += 1
        
        if not is_valid:
            stats["invalid"] += 1
            continue
        
        # Check if has valid answer
        has_answer = entry.get("has_answer", False)
        answer_text = entry.get("answer_text", "").strip()
        answer_start_ts = entry.get("answer_start_ts_id", "")
        
        has_valid_answer = (
            has_answer and 
            answer_text and 
            not is_placeholder_timestamp(answer_start_ts) and
            parse_timestamp(answer_start_ts) > 0
        )
        
        # Apply filters
        should_filter = False
        filter_reason = None
        
        if filter_no_answer and not has_valid_answer:
            should_filter = True
            filter_reason = "no_answer"
            stats["filtered_no_answer"] += 1
        
        if filter_invalid_timestamps and any("timestamp" in issue.lower() for issue in issues):
            should_filter = True
            filter_reason = "invalid_timestamps"
            stats["filtered_invalid_timestamps"] += 1
        
        if filter_placeholders and is_placeholder_timestamp(answer_start_ts):
            should_filter = True
            filter_reason = "placeholder"
            stats["filtered_placeholders"] += 1
        
        if should_filter:
            continue
        
        # Entry passed all filters
        if has_valid_answer:
            stats["valid_with_answer"] += 1
        else:
            stats["valid_without_answer"] += 1
        
        # Clean answer text before adding to filtered list
        entry_copy = entry.copy()
        if "answer_text" in entry_copy:
            original_answer = entry_copy["answer_text"]
            cleaned_answer = clean_answer_text(original_answer)
            entry_copy["answer_text"] = cleaned_answer
            
            # If cleaning removed everything, skip this entry
            if has_valid_answer and not cleaned_answer.strip():
                stats["filtered_no_answer"] += 1
                continue
        
        filtered.append(entry_copy)
    
    return filtered, stats

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate and filter label files")
    parser.add_argument("labels_path", type=str, help="Path to labels JSON file")
    parser.add_argument("--output", "-o", type=str, default=None, 
                       help="Output path for filtered labels (default: add '_filtered' suffix)")
    parser.add_argument("--keep-no-answer", action="store_true",
                       help="Keep entries without answers (default: filter them out)")
    parser.add_argument("--keep-invalid-timestamps", action="store_true",
                       help="Keep entries with invalid timestamps (default: filter them out)")
    parser.add_argument("--keep-placeholders", action="store_true",
                       help="Keep entries with placeholder timestamps (default: filter them out)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only show statistics, don't save filtered file")
    
    args = parser.parse_args()
    
    labels_path = Path(args.labels_path)
    if not labels_path.exists():
        print(f"❌ Labels file not found: {labels_path}")
        sys.exit(1)
    
    # Load labels
    print(f"Loading labels from: {labels_path}")
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    if not isinstance(labels, list):
        print(f"❌ Labels file is not a JSON array")
        sys.exit(1)
    
    print(f"Loaded {len(labels)} entries\n")
    
    # Filter labels
    filtered, stats = filter_labels(
        labels,
        filter_no_answer=not args.keep_no_answer,
        filter_invalid_timestamps=not args.keep_invalid_timestamps,
        filter_placeholders=not args.keep_placeholders
    )
    
    # Print statistics
    print("=" * 80)
    print("VALIDATION & FILTERING RESULTS")
    print("=" * 80)
    print(f"Total entries: {stats['total']}")
    print(f"\nValid entries:")
    print(f"  With answer: {stats['valid_with_answer']}")
    print(f"  Without answer: {stats['valid_without_answer']}")
    print(f"  Invalid: {stats['invalid']}")
    print(f"\nFiltered out:")
    print(f"  No answer: {stats['filtered_no_answer']}")
    print(f"  Invalid timestamps: {stats['filtered_invalid_timestamps']}")
    print(f"  Placeholders: {stats['filtered_placeholders']}")
    print(f"\nRemaining after filtering: {len(filtered)} ({len(filtered)/stats['total']*100:.1f}%)")
    
    if stats['issues_by_type']:
        print(f"\nIssues found:")
        for issue_type, count in sorted(stats['issues_by_type'].items()):
            print(f"  {issue_type}: {count}")
    
    print("=" * 80)
    
    # Save filtered labels if not dry-run
    if not args.dry_run:
        if args.output:
            output_path = Path(args.output)
        else:
            # Add _filtered suffix
            output_path = labels_path.parent / f"{labels_path.stem}_filtered{labels_path.suffix}"
        
        print(f"\nSaving filtered labels to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(filtered)} entries")
    else:
        print("\n[DRY RUN] No file saved")

if __name__ == "__main__":
    main()

