#!/usr/bin/env python3
"""
Batch validate and filter all label files in a directory.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Import validation function from validate_and_filter_labels
sys.path.insert(0, str(Path(__file__).parent))
from validate_and_filter_labels import filter_labels, validate_entry

def process_directory(labels_dir: str, 
                     output_dir: str = None,
                     filter_no_answer: bool = True,
                     filter_invalid_timestamps: bool = True,
                     filter_placeholders: bool = True,
                     dry_run: bool = False) -> Dict[str, Any]:
    """
    Process all label files in a directory.
    
    Returns:
        Summary statistics
    """
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        print(f"❌ Directory not found: {labels_dir}")
        return {}
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = labels_dir
    
    # Find all JSON files
    label_files = list(labels_dir.glob("*.json"))
    
    if not label_files:
        print(f"❌ No JSON files found in {labels_dir}")
        return {}
    
    print(f"Found {len(label_files)} label files\n")
    
    # Process each file
    total_stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "total_entries": 0,
        "total_valid_with_answer": 0,
        "total_valid_without_answer": 0,
        "total_invalid": 0,
        "total_filtered": 0,
    }
    
    file_results = []
    
    for label_file in sorted(label_files):
        print(f"Processing: {label_file.name}")
        
        try:
            # Load labels
            with open(label_file, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            
            if not isinstance(labels, list):
                print(f"  ⚠️  Skipping: Not a JSON array")
                total_stats["files_skipped"] += 1
                continue
            
            # Filter labels
            filtered, stats = filter_labels(
                labels,
                filter_no_answer=filter_no_answer,
                filter_invalid_timestamps=filter_invalid_timestamps,
                filter_placeholders=filter_placeholders
            )
            
            # Update totals
            total_stats["files_processed"] += 1
            total_stats["total_entries"] += stats["total"]
            total_stats["total_valid_with_answer"] += stats["valid_with_answer"]
            total_stats["total_valid_without_answer"] += stats["valid_without_answer"]
            total_stats["total_invalid"] += stats["invalid"]
            total_stats["total_filtered"] += (
                stats["filtered_no_answer"] + 
                stats["filtered_invalid_timestamps"] + 
                stats["filtered_placeholders"]
            )
            
            file_result = {
                "file": label_file.name,
                "total": stats["total"],
                "valid_with_answer": stats["valid_with_answer"],
                "valid_without_answer": stats["valid_without_answer"],
                "invalid": stats["invalid"],
                "filtered": stats["total"] - len(filtered),
                "remaining": len(filtered),
            }
            file_results.append(file_result)
            
            print(f"  ✓ {len(filtered)}/{stats['total']} entries remaining ({len(filtered)/stats['total']*100:.1f}%)")
            
            # Save filtered file if not dry-run
            if not dry_run:
                output_path = output_dir / f"{label_file.stem}_filtered{label_file.suffix}"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(filtered, f, indent=2, ensure_ascii=False)
                print(f"  ✓ Saved to: {output_path.name}")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            total_stats["files_skipped"] += 1
            continue
        
        print()
    
    # Print summary
    print("=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Files skipped: {total_stats['files_skipped']}")
    print(f"\nTotal entries: {total_stats['total_entries']}")
    print(f"  Valid with answer: {total_stats['total_valid_with_answer']}")
    print(f"  Valid without answer: {total_stats['total_valid_without_answer']}")
    print(f"  Invalid: {total_stats['total_invalid']}")
    print(f"  Filtered out: {total_stats['total_filtered']}")
    print(f"\nRemaining: {total_stats['total_valid_with_answer'] + total_stats['total_valid_without_answer']} "
          f"({(total_stats['total_valid_with_answer'] + total_stats['total_valid_without_answer'])/total_stats['total_entries']*100:.1f}%)")
    
    # Show per-file breakdown
    if file_results:
        print(f"\nPer-file breakdown:")
        print(f"{'File':<50} {'Total':<8} {'Valid':<8} {'Filtered':<10} {'Remaining':<10}")
        print("-" * 80)
        for result in file_results[:10]:  # Show first 10
            print(f"{result['file']:<50} {result['total']:<8} {result['valid_with_answer']:<8} "
                  f"{result['filtered']:<10} {result['remaining']:<10}")
        if len(file_results) > 10:
            print(f"... and {len(file_results) - 10} more files")
    
    print("=" * 80)
    
    return total_stats

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch validate and filter label files")
    parser.add_argument("labels_dir", type=str, help="Directory containing label JSON files")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                       help="Output directory for filtered labels (default: same as input)")
    parser.add_argument("--keep-no-answer", action="store_true",
                       help="Keep entries without answers")
    parser.add_argument("--keep-invalid-timestamps", action="store_true",
                       help="Keep entries with invalid timestamps")
    parser.add_argument("--keep-placeholders", action="store_true",
                       help="Keep entries with placeholder timestamps")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only show statistics, don't save filtered files")
    
    args = parser.parse_args()
    
    process_directory(
        args.labels_dir,
        output_dir=args.output_dir,
        filter_no_answer=not args.keep_no_answer,
        filter_invalid_timestamps=not args.keep_invalid_timestamps,
        filter_placeholders=not args.keep_placeholders,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()

