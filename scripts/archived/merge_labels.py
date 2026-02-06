#!/usr/bin/env python3
"""
Utility to merge multiple label files or handle them separately.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_labels_file(labels_path: str) -> Dict[str, Any]:
    """Load a labels JSON file."""
    with open(labels_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_labels_file(data: Dict[str, Any], output_path: str):
    """Save labels to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved to {output_path}")


def merge_labels_files(label_files: List[str], output_path: str, video_path: str = None):
    """
    Merge multiple label files into one.
    
    Args:
        label_files: List of paths to label files
        output_path: Output path for merged labels
        video_path: Optional video path to use (overrides ones in files)
    """
    all_segments = []
    seen_videos = set()
    
    print(f"Merging {len(label_files)} label files...")
    
    for labels_file in label_files:
        print(f"  Loading {labels_file}...")
        data = load_labels_file(labels_file)
        
        file_video_path = data.get("video_path", "")
        if video_path:
            final_video_path = video_path
        elif file_video_path:
            final_video_path = file_video_path
        else:
            final_video_path = ""
        
        if final_video_path:
            seen_videos.add(final_video_path)
        
        segments = data.get("segments", [])
        print(f"    Found {len(segments)} segments")
        all_segments.extend(segments)
    
    # Use provided video_path or first one found
    if video_path:
        merged_video_path = video_path
    elif seen_videos:
        merged_video_path = list(seen_videos)[0]
        if len(seen_videos) > 1:
            print(f"⚠ Warning: Multiple videos found. Using: {merged_video_path}")
            print(f"  Other videos: {list(seen_videos)[1:]}")
    else:
        merged_video_path = ""
    
    merged_data = {
        "video_path": merged_video_path,
        "segments": all_segments
    }
    
    print(f"\n✓ Merged {len(all_segments)} total segments")
    save_labels_file(merged_data, output_path)
    
    return merged_data


def list_labels_files(labels_dir: str = "labels"):
    """List all label files in directory."""
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}")
        return []
    
    label_files = list(labels_dir.glob("*.json"))
    return sorted(label_files)


def show_labels_summary(label_files: List[str]):
    """Show summary of all label files."""
    print("\n" + "="*60)
    print("Label Files Summary")
    print("="*60)
    
    for i, label_file in enumerate(label_files, 1):
        try:
            data = load_labels_file(str(label_file))
            video_path = data.get("video_path", "N/A")
            segments = data.get("segments", [])
            print(f"\n{i}. {label_file.name}")
            print(f"   Video: {Path(video_path).name if video_path != 'N/A' else 'N/A'}")
            print(f"   Segments: {len(segments)}")
            if segments:
                print(f"   Questions:")
                for seg in segments[:3]:  # Show first 3
                    q = seg.get("question", "")[:50]
                    print(f"     - {q}")
                if len(segments) > 3:
                    print(f"     ... and {len(segments) - 3} more")
        except Exception as e:
            print(f"\n{i}. {label_file.name} - ERROR: {e}")
    
    print("\n" + "="*60)


def extract_separate_datasets(label_files: List[str], video_path: str, base_output_dir: str):
    """
    Extract datasets from each label file separately.
    
    Args:
        label_files: List of label file paths
        video_path: Path to video file
        base_output_dir: Base directory for outputs
    """
    from src.dataset_extractor import DatasetExtractor
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "dataset_config.yaml"
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    extractor = DatasetExtractor(
        use_whisper=config.get("use_whisper", False),
        whisper_model=config.get("whisper_model", "base"),
        language=config.get("language", None),
        silence_threshold=config.get("silence_threshold", -40.0),
        min_silence_duration=config.get("min_silence_duration", 1.0),
        question_window_before=config.get("question_window_before", 10.0),
        answer_window_after=config.get("answer_window_after", 10.0),
        pause_fps=config.get("pause_fps", 0.25),
        pause_max_frames=config.get("pause_max_frames", 4),
        use_full_video=config.get("use_full_video", True),
        full_video_fps=config.get("full_video_fps", 0.2),
        full_video_max_frames=config.get("full_video_max_frames", 40),
        grid_cols=config.get("grid_cols", 2),
        grid_rows=config.get("grid_rows", 2),
        max_collages=config.get("max_collages", 3),
        max_size=config.get("max_size", 512),
    )
    
    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExtracting datasets from {len(label_files)} label files...")
    
    for label_file in label_files:
        label_path = Path(label_file)
        label_name = label_path.stem
        output_path = base_output_dir / f"dataset_{label_name}"
        
        # Get video path from label file if not provided
        label_data = load_labels_file(str(label_path))
        label_video_path = label_data.get("video_path", "")
        
        # Use video_path from label file, or fall back to provided video_path
        if label_video_path and Path(label_video_path).exists():
            actual_video_path = label_video_path
        elif label_video_path:
            # Try to find video in dora_videos directory
            video_name = Path(label_video_path).name
            possible_path = Path(video_path).parent / video_name if video_path else Path("dora_videos") / video_name
            if possible_path.exists():
                actual_video_path = str(possible_path)
            else:
                actual_video_path = video_path
        else:
            actual_video_path = video_path
        
        print(f"\n  Processing {label_path.name}...")
        print(f"    Video: {Path(actual_video_path).name if actual_video_path else 'N/A'}")
        print(f"    Output: {output_path}")
        
        if not actual_video_path or not Path(actual_video_path).exists():
            print(f"    ✗ Error: Video file not found: {actual_video_path}")
            continue
        
        try:
            dataset = extractor.extract_and_save(
                video_path=actual_video_path,
                output_path=str(output_path),
                manual_labels_path=str(label_path),
                create_splits=False,
                format="arrow"
            )
            print(f"    ✓ Created dataset with {len(dataset)} examples")
        except Exception as e:
            print(f"    ✗ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge or handle multiple label files"
    )
    parser.add_argument(
        "action",
        choices=["merge", "list", "extract-separate"],
        help="Action to perform: merge labels, list files, or extract separate datasets"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Label files to process (for merge/extract-separate)"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="labels",
        help="Directory containing label files (default: labels)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path (for merge) or base directory (for extract-separate)"
    )
    parser.add_argument(
        "--video-path",
        type=str,
        help="Video file path (for extract-separate or to override in merge)"
    )
    
    args = parser.parse_args()
    
    if args.action == "list":
        label_files = list_labels_files(args.labels_dir)
        if not label_files:
            print("No label files found.")
            return
        show_labels_summary(label_files)
    
    elif args.action == "merge":
        if not args.labels:
            # List available files and let user choose
            label_files = list_labels_files(args.labels_dir)
            if not label_files:
                print("No label files found.")
                return
            show_labels_summary(label_files)
            print("\nUsage: python scripts/merge_labels.py merge --labels <file1> <file2> ... --output <output.json>")
            return
        
        if not args.output:
            args.output = args.labels_dir + "/merged_labels.json"
            print(f"No output specified, using: {args.output}")
        
        merge_labels_files(args.labels, args.output, args.video_path)
    
    elif args.action == "extract-separate":
        if not args.labels:
            label_files = list_labels_files(args.labels_dir)
            if not label_files:
                print("No label files found.")
                return
            show_labels_summary(label_files)
            print("\nUsage: python scripts/merge_labels.py extract-separate --labels <file1> <file2> ... --video-path <video.mp4> --output <output_dir>")
            return
        
        if not args.video_path:
            print("Error: --video-path required for extract-separate")
            return
        
        if not args.output:
            args.output = "./outputs/datasets"
            print(f"No output directory specified, using: {args.output}")
        
        extract_separate_datasets(args.labels, args.video_path, args.output)


if __name__ == "__main__":
    main()

