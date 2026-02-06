#!/usr/bin/env python3
"""
CLI tool to extract Q&A dataset from Dora videos.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_extractor import DatasetExtractor
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Q&A dataset from Dora videos"
    )
    parser.add_argument(
        "video_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to video file (optional if using unified labels with multiple videos)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for dataset"
    )
    parser.add_argument(
        "--transcript",
        "-t",
        type=str,
        default=None,
        help="Path to manual transcript file"
    )
    parser.add_argument(
        "--manual-segments",
        type=str,
        default=None,
        help="Path to manually specified segments (JSON/YAML old format)"
    )
    parser.add_argument(
        "--manual-labels",
        type=str,
        default=None,
        help="Path to manual labels JSON file with timestamps (recommended)"
    )
    parser.add_argument(
        "--use-whisper",
        action="store_true",
        help="Use Whisper for automatic transcript extraction"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code for Whisper (e.g., 'en')"
    )
    parser.add_argument(
        "--no-splits",
        action="store_true",
        help="Don't create train/val/test splits"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="arrow",
        choices=["arrow", "json", "parquet"],
        help="Dataset format"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Create extractor with config
    extractor = DatasetExtractor(
        use_whisper=args.use_whisper or config.get("use_whisper", False),
        whisper_model=args.whisper_model or config.get("whisper_model", "base"),
        language=args.language or config.get("language", None),
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
    
    # Extract and save dataset
    if args.manual_labels:
        print(f"Using manual labels from: {args.manual_labels}")
        if args.video_path:
            print(f"Extracting from video: {args.video_path}")
        else:
            print("Extracting from all videos in unified labels file...")
    elif args.manual_segments:
        print(f"Using manual segments from: {args.manual_segments}")
        if args.video_path:
            print(f"Extracting from video: {args.video_path}")
    else:
        if not args.video_path:
            print("Error: video_path required for automatic detection")
            return
        print(f"Extracting dataset from {args.video_path}...")
        print("Using automatic Q&A detection")
    
    dataset = extractor.extract_and_save(
        video_path=args.video_path,
        output_path=args.output,
        transcript_path=args.transcript,
        manual_segments_path=args.manual_segments,
        manual_labels_path=args.manual_labels,
        create_splits=not args.no_splits,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        format=args.format
    )
    
    # Print summary
    if hasattr(dataset, "keys"):  # DatasetDict
        print("\nDataset created with splits:")
        for split_name, split_dataset in dataset.items():
            print(f"  {split_name}: {len(split_dataset)} examples")
    else:
        print(f"\nDataset created: {len(dataset)} examples")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

