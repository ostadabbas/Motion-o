#!/usr/bin/env python3
"""
Validate and filter all label files across all seasons.
Saves filtered files to local output directory.
"""
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
from batch_validate_labels import process_directory

def main():
    base_labels_dir = Path("/projects/XXXX-1/dora/qap/")
    output_base = Path("/projects/XXXX-1/dora/filtered_labels")
    
    seasons = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    
    print("=" * 80)
    print("VALIDATING ALL SEASONS")
    print("=" * 80)
    print(f"Input: {base_labels_dir}")
    print(f"Output: {output_base}")
    print()
    
    all_stats = {}
    
    for season in seasons:
        season_dir = base_labels_dir / season
        if not season_dir.exists():
            print(f"⚠️  Season {season} not found, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"PROCESSING {season}")
        print(f"{'='*80}")
        
        output_dir = output_base / season
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = process_directory(
            str(season_dir),
            output_dir=str(output_dir),
            filter_no_answer=True,
            filter_invalid_timestamps=True,
            filter_placeholders=True,
            dry_run=False
        )
        
        all_stats[season] = stats
    
    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY - ALL SEASONS")
    print("=" * 80)
    
    total_entries = sum(s.get("total_entries", 0) for s in all_stats.values())
    total_valid = sum(s.get("total_valid_with_answer", 0) for s in all_stats.values())
    total_filtered = sum(s.get("total_filtered", 0) for s in all_stats.values())
    
    print(f"Total entries across all seasons: {total_entries}")
    print(f"Valid entries with answers: {total_valid} ({total_valid/total_entries*100:.1f}%)")
    print(f"Filtered out: {total_filtered} ({total_filtered/total_entries*100:.1f}%)")
    
    print(f"\nPer-season breakdown:")
    print(f"{'Season':<10} {'Total':<10} {'Valid':<10} {'Filtered':<10} {'% Valid':<10}")
    print("-" * 60)
    for season in sorted(all_stats.keys()):
        stats = all_stats[season]
        total = stats.get("total_entries", 0)
        valid = stats.get("total_valid_with_answer", 0)
        filtered = stats.get("total_filtered", 0)
        pct = (valid / total * 100) if total > 0 else 0
        print(f"{season:<10} {total:<10} {valid:<10} {filtered:<10} {pct:<10.1f}%")
    
    print("=" * 80)
    print(f"\n✓ Filtered labels saved to: {output_base}")

if __name__ == "__main__":
    main()

