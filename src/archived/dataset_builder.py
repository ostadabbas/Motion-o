"""
Dataset builder for creating HuggingFace datasets from Q&A segments.
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import numpy as np

from datasets import Dataset, DatasetDict
from .qa_segmenter import QASegment
from .video_utils import extract_frames_between, sample_full_video_as_collages


def extract_frames_for_segment(video_path: str,
                                segment: QASegment,
                                pause_fps: float = 0.25,
                                pause_max_frames: int = 4,
                                use_full_video: bool = True,
                                full_video_fps: float = 0.2,
                                full_video_max_frames: int = 40,
                                grid_cols: int = 2,
                                grid_rows: int = 2,
                                max_collages: int = 3,
                                max_size: int = 512) -> List[Tuple[Image.Image, float]]:
    """
    Extract frames for a Q&A segment.
    
    Args:
        video_path: Path to video file
        segment: QASegment object
        pause_fps: FPS for sampling frames during pause
        pause_max_frames: Maximum frames to extract during pause
        use_full_video: Whether to include full video context before pause
        full_video_fps: FPS for sampling full video context
        full_video_max_frames: Maximum frames for full video context
        grid_cols: Grid columns for collages
        grid_rows: Grid rows for collages
        max_collages: Maximum number of collages
        max_size: Maximum size for frame resizing
        
    Returns:
        List of (PIL.Image, timestamp) tuples
    """
    frames = []
    
    # Optionally include full-video context before the pause
    if use_full_video and segment.pause_start > 0:
        collages = sample_full_video_as_collages(
            video_path,
            pause_start=segment.pause_start,
            fps=full_video_fps,
            max_frames=full_video_max_frames,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            max_collages=max_collages,
            tile_size=max_size,
        )
        frames.extend(collages)
    
    # Extract frames from pause window
    pause_frames = extract_frames_between(
        video_path,
        segment.pause_start,
        segment.pause_end,
        fps=pause_fps,
        max_frames=pause_max_frames,
        max_size=max_size
    )
    frames.extend(pause_frames)
    
    return frames


def frames_to_list(frames: List[Tuple[Image.Image, float]]) -> List[Dict[str, Any]]:
    """
    Convert frames to serializable format for dataset.
    
    Args:
        frames: List of (PIL.Image, timestamp) tuples
        
    Returns:
        List of dictionaries with frame data
    """
    frame_list = []
    for img, timestamp in frames:
        # Convert PIL Image to numpy array for storage
        img_array = np.array(img)
        frame_list.append({
            "image": img_array,
            "timestamp": float(timestamp)
        })
    return frame_list


def build_dataset_from_segments(video_path: str,
                                 segments: List[QASegment],
                                 pause_fps: float = 0.25,
                                 pause_max_frames: int = 4,
                                 use_full_video: bool = True,
                                 full_video_fps: float = 0.2,
                                 full_video_max_frames: int = 40,
                                 grid_cols: int = 2,
                                 grid_rows: int = 2,
                                 max_collages: int = 3,
                                 max_size: int = 512,
                                 store_images: bool = True) -> Dataset:
    """
    Build HuggingFace Dataset from Q&A segments.
    
    Args:
        video_path: Path to video file
        segments: List of QASegment objects
        pause_fps: FPS for sampling frames during pause
        pause_max_frames: Maximum frames to extract during pause
        use_full_video: Whether to include full video context
        full_video_fps: FPS for sampling full video context
        full_video_max_frames: Maximum frames for full video context
        grid_cols: Grid columns for collages
        grid_rows: Grid rows for collages
        max_collages: Maximum number of collages
        max_size: Maximum size for frame resizing
        store_images: Whether to store images in dataset (if False, only store paths)
        
    Returns:
        HuggingFace Dataset object
    """
    data = []
    
    for i, segment in enumerate(segments):
        # Extract frames
        frames = extract_frames_for_segment(
            video_path,
            segment,
            pause_fps=pause_fps,
            pause_max_frames=pause_max_frames,
            use_full_video=use_full_video,
            full_video_fps=full_video_fps,
            full_video_max_frames=full_video_max_frames,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            max_collages=max_collages,
            max_size=max_size
        )
        
        if not frames:
            continue
        
        # Prepare data entry
        entry = {
            "video_path": video_path,
            "segment_id": i,
            "question_start": segment.question_start,
            "pause_start": segment.pause_start,
            "pause_end": segment.pause_end,
            "answer_end": segment.answer_end,
            "question": segment.question,
            "answer": segment.answer,
            "transcript": segment.transcript,
            "confidence": segment.confidence,
        }
        
        if store_images:
            # Store frames as list of image arrays
            entry["frames"] = frames_to_list(frames)
            entry["num_frames"] = len(frames)
        else:
            # Store frame paths (would need to save frames separately)
            entry["num_frames"] = len(frames)
            entry["frame_timestamps"] = [f[1] for f in frames]
        
        data.append(entry)
    
    # Create dataset
    dataset = Dataset.from_list(data)
    return dataset


def create_train_val_test_split(dataset: Dataset,
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15,
                                 seed: int = 42) -> DatasetDict:
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset: HuggingFace Dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set (must sum to 1.0)
        seed: Random seed for shuffling
        
    Returns:
        DatasetDict with 'train', 'validation', 'test' splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })


def save_dataset(dataset: Dataset,
                 output_path: str,
                 format: str = "arrow"):
    """
    Save dataset to disk.
    
    Args:
        dataset: HuggingFace Dataset or DatasetDict
        output_path: Path to save dataset
        format: Format to save ("arrow", "json", "parquet")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(dataset, DatasetDict):
        dataset.save_to_disk(str(output_path))
    else:
        if format == "arrow":
            dataset.save_to_disk(str(output_path))
        elif format == "json":
            dataset.to_json(str(output_path))
        elif format == "parquet":
            dataset.to_parquet(str(output_path))
        else:
            raise ValueError(f"Unsupported format: {format}")


def load_dataset(dataset_path: str) -> Dataset:
    """
    Load dataset from disk.
    
    Args:
        dataset_path: Path to saved dataset
        
    Returns:
        HuggingFace Dataset or DatasetDict
    """
    from datasets import load_from_disk
    
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        return load_from_disk(str(dataset_path))
    elif dataset_path.suffix == '.json':
        from datasets import load_dataset
        return load_dataset('json', data_files=str(dataset_path))['train']
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")

