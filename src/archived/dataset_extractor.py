"""
Main dataset extraction pipeline for Dora videos.
Orchestrates transcript extraction, Q&A segmentation, and dataset building.
"""
import os
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path

from .transcript_utils import (
    get_transcript,
    extract_transcript_with_timestamps,
    normalize_transcript
)
from .qa_segmenter import (
    QASegment,
    segment_qa_from_transcript,
    detect_pauses_in_audio,
    extract_audio_from_video,
    load_manual_segments
)
from .manual_labels import load_manual_labels
from .dataset_builder import (
    build_dataset_from_segments,
    create_train_val_test_split,
    save_dataset
)
from datasets import Dataset, DatasetDict


class DatasetExtractor:
    """
    Main class for extracting Q&A datasets from Dora videos.
    """
    
    def __init__(self,
                 use_whisper: bool = False,
                 whisper_model: str = "base",
                 language: Optional[str] = None,
                 silence_threshold: float = -40.0,
                 min_silence_duration: float = 1.0,
                 question_window_before: float = 10.0,
                 answer_window_after: float = 10.0,
                 pause_fps: float = 0.25,
                 pause_max_frames: int = 4,
                 use_full_video: bool = True,
                 full_video_fps: float = 0.2,
                 full_video_max_frames: int = 40,
                 grid_cols: int = 2,
                 grid_rows: int = 2,
                 max_collages: int = 3,
                 max_size: int = 512):
        """
        Initialize dataset extractor.
        
        Args:
            use_whisper: Whether to use Whisper for automatic transcript extraction
            whisper_model: Whisper model size
            language: Language code for Whisper
            silence_threshold: dB threshold for pause detection
            min_silence_duration: Minimum duration for pause detection
            question_window_before: Seconds before pause to search for question
            answer_window_after: Seconds after pause to search for answer
            pause_fps: FPS for sampling frames during pause
            pause_max_frames: Maximum frames during pause
            use_full_video: Whether to include full video context
            full_video_fps: FPS for full video context
            full_video_max_frames: Maximum frames for full video context
            grid_cols: Grid columns for collages
            grid_rows: Grid rows for collages
            max_collages: Maximum number of collages
            max_size: Maximum size for frame resizing
        """
        self.use_whisper = use_whisper
        self.whisper_model = whisper_model
        self.language = language
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.question_window_before = question_window_before
        self.answer_window_after = answer_window_after
        self.pause_fps = pause_fps
        self.pause_max_frames = pause_max_frames
        self.use_full_video = use_full_video
        self.full_video_fps = full_video_fps
        self.full_video_max_frames = full_video_max_frames
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.max_collages = max_collages
        self.max_size = max_size
    
    def extract_from_video(self,
                          video_path: str,
                          transcript_path: Optional[str] = None,
                          manual_segments_path: Optional[str] = None,
                          manual_labels_path: Optional[str] = None) -> List[QASegment]:
        """
        Extract Q&A segments from a video.
        
        Args:
            video_path: Path to video file
            transcript_path: Optional path to manual transcript
            manual_segments_path: Optional path to manually specified segments (YAML/JSON old format)
            manual_labels_path: Optional path to manual labels JSON (new format with timestamps)
            
        Returns:
            List of QASegment objects
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Priority 1: Manual labels (new format with timestamps) - most accurate
        if manual_labels_path and os.path.exists(manual_labels_path):
            print(f"Loading manual labels from: {manual_labels_path}")
            return load_manual_labels(manual_labels_path, video_path=str(video_path))
        
        # Priority 2: Manual segments (old format)
        if manual_segments_path and os.path.exists(manual_segments_path):
            return load_manual_segments(manual_segments_path)
        
        # Get transcript
        if self.use_whisper:
            # Get transcript with timestamps
            timestamps = extract_transcript_with_timestamps(
                str(video_path),
                model_name=self.whisper_model,
                language=self.language
            )
            transcript = ' '.join([seg['text'] for seg in timestamps])
        else:
            transcript = get_transcript(
                str(video_path),
                transcript_path=transcript_path,
                use_whisper=False
            )
            timestamps = None
        
        # Detect pauses if we have audio analysis capability
        pauses = None
        if self.use_whisper or transcript_path:
            try:
                # Extract audio temporarily
                audio_path = extract_audio_from_video(str(video_path))
                pauses = detect_pauses_in_audio(
                    audio_path,
                    silence_threshold=self.silence_threshold,
                    min_silence_duration=self.min_silence_duration
                )
                # Clean up temp audio file
                if os.path.exists(audio_path) and audio_path.startswith(tempfile.gettempdir()):
                    try:
                        os.remove(audio_path)
                    except:
                        pass
            except Exception as e:
                print(f"Warning: Could not detect pauses from audio: {e}")
                pauses = None
        
        # Segment Q&A pairs using generic pause-based approach
        segments = segment_qa_from_transcript(
            transcript,
            timestamps=timestamps,
            pauses=pauses,
            question_window_before=self.question_window_before,
            answer_window_after=self.answer_window_after
        )
        
        return segments
    
    def build_dataset(self,
                     video_path: str,
                     segments: Optional[List[QASegment]] = None,
                     transcript_path: Optional[str] = None,
                     manual_segments_path: Optional[str] = None,
                     manual_labels_path: Optional[str] = None,
                     store_images: bool = True) -> Dataset:
        """
        Build HuggingFace dataset from video.
        
        Args:
            video_path: Path to video file
            segments: Optional pre-extracted segments
            transcript_path: Optional path to manual transcript
            manual_segments_path: Optional path to manual segments
            store_images: Whether to store images in dataset
            
        Returns:
            HuggingFace Dataset
        """
        if segments is None:
            segments = self.extract_from_video(
                video_path,
                transcript_path=transcript_path,
                manual_segments_path=manual_segments_path,
                manual_labels_path=manual_labels_path
            )
        
        if not segments:
            raise ValueError("No Q&A segments found in video")
        
        dataset = build_dataset_from_segments(
            video_path,
            segments,
            pause_fps=self.pause_fps,
            pause_max_frames=self.pause_max_frames,
            use_full_video=self.use_full_video,
            full_video_fps=self.full_video_fps,
            full_video_max_frames=self.full_video_max_frames,
            grid_cols=self.grid_cols,
            grid_rows=self.grid_rows,
            max_collages=self.max_collages,
            max_size=self.max_size,
            store_images=store_images
        )
        
        return dataset
    
    def extract_and_save(self,
                        video_path: Optional[str] = None,
                        output_path: str = "./outputs/dataset",
                        transcript_path: Optional[str] = None,
                        manual_segments_path: Optional[str] = None,
                        manual_labels_path: Optional[str] = None,
                        create_splits: bool = True,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15,
                        seed: int = 42,
                        format: str = "arrow") -> Dataset:
        """
        Extract dataset and save to disk.
        
        Supports unified labels format with multiple videos.
        If manual_labels_path contains multiple videos, extracts from all of them.
        
        Args:
            video_path: Optional path to video file (if None and unified labels, extracts from all videos)
            output_path: Path to save dataset
            transcript_path: Optional path to manual transcript
            manual_segments_path: Optional path to manual segments
            manual_labels_path: Optional path to manual labels (supports unified format)
            create_splits: Whether to create train/val/test splits
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for splits
            format: Dataset format ("arrow", "json", "parquet")
            
        Returns:
            Dataset or DatasetDict
        """
        # Check if unified labels format (multiple videos)
        if manual_labels_path and Path(manual_labels_path).exists():
            import json
            with open(manual_labels_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
            
            # Check if unified format
            if "videos" in labels_data:
                videos_list = labels_data["videos"]
                
                # Filter by video_path if provided
                if video_path:
                    video_name = Path(video_path).name
                    videos_list = [
                        v for v in videos_list 
                        if Path(v.get("video_path", "")).name == video_name
                    ]
                
                if len(videos_list) > 1:
                    # Extract from multiple videos and combine
                    from datasets import concatenate_datasets
                    all_datasets = []
                    
                    print(f"Extracting from {len(videos_list)} videos in unified labels file...")
                    for video_entry in videos_list:
                        entry_video_path = video_entry.get("video_path", "")
                        if not entry_video_path or not Path(entry_video_path).exists():
                            # Try to find video in dora_videos directory
                            video_name = Path(entry_video_path).name if entry_video_path else ""
                            possible_path = Path("dora_videos") / video_name
                            if possible_path.exists():
                                entry_video_path = str(possible_path)
                            else:
                                print(f"⚠ Skipping {video_name}: video file not found")
                                continue
                        
                        print(f"  Processing {Path(entry_video_path).name}...")
                        try:
                            video_dataset = self.build_dataset(
                                entry_video_path,
                                transcript_path=transcript_path,
                                manual_segments_path=manual_segments_path,
                                manual_labels_path=manual_labels_path
                            )
                            if video_dataset:
                                all_datasets.append(video_dataset)
                                print(f"    ✓ Extracted {len(video_dataset)} examples")
                        except Exception as e:
                            print(f"    ✗ Error: {e}")
                            continue
                    
                    if not all_datasets:
                        raise ValueError("No datasets extracted from any video")
                    
                    # Combine all datasets
                    if len(all_datasets) == 1:
                        dataset = all_datasets[0]
                    else:
                        dataset = concatenate_datasets(all_datasets)
                    print(f"\n✓ Combined dataset: {len(dataset)} total examples")
                else:
                    # Single video (or filtered to one)
                    if videos_list:
                        entry_video_path = videos_list[0].get("video_path", "")
                        if not entry_video_path or not Path(entry_video_path).exists():
                            # Try to find in dora_videos
                            video_name = Path(entry_video_path).name if entry_video_path else ""
                            possible_path = Path("dora_videos") / video_name
                            if possible_path.exists():
                                entry_video_path = str(possible_path)
                            else:
                                entry_video_path = video_path  # Fallback to provided
                        else:
                            entry_video_path = entry_video_path
                    else:
                        entry_video_path = video_path
                    
                    dataset = self.build_dataset(
                        entry_video_path,
                        transcript_path=transcript_path,
                        manual_segments_path=manual_segments_path,
                        manual_labels_path=manual_labels_path
                    )
            else:
                # Old format (single video)
                if not video_path:
                    video_path = labels_data.get("video_path", "")
                dataset = self.build_dataset(
                    video_path,
                    transcript_path=transcript_path,
                    manual_segments_path=manual_segments_path,
                    manual_labels_path=manual_labels_path
                )
        else:
            # No manual labels, use video_path
            if not video_path:
                raise ValueError("video_path required when not using manual labels")
            dataset = self.build_dataset(
                video_path,
                transcript_path=transcript_path,
                manual_segments_path=manual_segments_path,
                manual_labels_path=manual_labels_path
            )
        
        if create_splits and len(dataset) > 1:
            dataset = create_train_val_test_split(
                dataset,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed
            )
        
        save_dataset(dataset, output_path, format=format)
        return dataset

