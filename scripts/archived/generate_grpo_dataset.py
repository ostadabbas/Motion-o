#!/usr/bin/env python3
"""
Generate GRPO training dataset from filtered labels.

This script:
1. Reads filtered labels from outputs/filtered_labels/S{1-8}/
2. Extracts transcript from SRT files (20s window before question, stops before question)
3. Extracts frames from videos at key timestamps using existing video_utils
4. Creates HuggingFace Dataset in format expected by GRPO training

Usage:
    python scripts/generate_grpo_dataset.py \
        --labels-dir outputs/filtered_labels \
        --videos-dir /mnt/data/dora/mp4 \
        --srt-dir /mnt/work/XXXX-7/dora/dataset/srt \
        --output-dir outputs/dataset \
        --seasons 1 2 3 4 5 6 7 8
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset
from tqdm import tqdm
import os 
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import random

# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing utilities from src/
from src.video_utils import extract_frames_between, sample_full_video_as_collages
from src.dataset_builder import frames_to_list
from src.text_cleaning import clean_text, clean_answer_text
from scripts.srt_parser import find_srt_file, extract_transcript_window
from scripts.visualize_vqa_structure import parse_timestamp
import numpy as np
from PIL import Image
import re


def extract_frames_for_entry(
    video_path: str,
    entry: Dict[str, Any],
    pause_fps: float = 0.25,
    pause_max_frames: int = 4,
    max_size: int = 448,
    use_full_video: bool = False,
    full_video_fps: float = 0.2,
    full_video_max_frames: int = 40,
    grid_cols: int = 2,
    grid_rows: int = 2,
    max_collages: int = 3,
    clip: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Extract frames from video for a Q&A entry using existing video_utils.
    
    Extracts frames during pause window (main context for GRPO training).
    Optionally includes frames from video start to pause (full video context).
    
    Args:
        video_path: Path to video file
        entry: Label entry with timestamps
        pause_fps: FPS for sampling frames during pause
        pause_max_frames: Maximum frames to extract during pause
        max_size: Maximum size for frame resizing (maintains aspect ratio)
        use_full_video: Whether to include frames from video start to pause
        full_video_fps: FPS for sampling full video context
        full_video_max_frames: Maximum frames for full video context
        grid_cols: Grid columns for collages
        grid_rows: Grid rows for collages
        max_collages: Maximum number of collages
    
    Returns:
        List of frame dictionaries with "image" and "timestamp" keys
    """
    # Parse timestamps
    pause_start = parse_timestamp(entry.get("pause_start", ""))
    pause_end = parse_timestamp(entry.get("pause_end", ""))
    if pause_start == pause_end:
        pause_start -= 1
        pause_end += 1   
    
    # Extract frames - optionally include full video context before pause
    frames = []
    
    # Optionally include full-video context before the pause (like dataset_builder.py)
    if use_full_video and pause_start > 0:
        collages = sample_full_video_as_collages(
            video_path,
            pause_start=pause_start,
            fps=full_video_fps,
            max_frames=full_video_max_frames,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            max_collages=max_collages,
            tile_size=max_size,
            clip=clip
        )
        frames.extend(collages)
    
    # Extract frames during pause (main context window)
    if pause_start > 0 and pause_end > pause_start:
        # Use existing extract_frames_between utility with cached clip
        pause_frames = extract_frames_between(
            video_path,
            pause_start,
            pause_end,
            fps=pause_fps,
            max_frames=pause_max_frames,
            max_size=max_size,
            clip=clip
        )
        frames.extend(pause_frames)
    
    # Convert to serializable format using existing utility
    frame_list = frames_to_list(frames)
    
    # Pre-resize images to exactly 448x448 (square) for efficiency during training
    # This avoids resizing every batch
    preprocessed_frames = []
    for frame_dict in frame_list:
        img_array = frame_dict.get("image")
        if img_array is not None:
            if isinstance(img_array, np.ndarray):
                img = Image.fromarray(img_array)
            else:
                img = Image.fromarray(np.array(img_array, dtype=np.uint8))
            
            # Resize to exactly 448x448 (square, center crop or pad)
            img = img.resize((448, 448), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.uint8)
            
            preprocessed_frames.append({
                "image": img_array.tolist(),  # Convert to list for JSON serialization
                "timestamp": frame_dict.get("timestamp", 0.0)
            })
    
    return preprocessed_frames

def remove_questions_from_transcript(transcript: str, all_questions: List[str]) -> str:
    """
    Remove ALL questions from transcript to prevent answer leakage.
    This is done once during data generation instead of every batch.
    
    Args:
        transcript: Full transcript text
        all_questions: List of all questions in the dataset
    
    Returns:
        Transcript with all questions removed
    """
    if not transcript or not all_questions:
        return transcript
    
    # Find the earliest position of any question in the sequence
    earliest_question_pos = len(transcript)
    
    for q in all_questions:
        if not q or not q.strip():
            continue
        
        # Try multiple patterns for robust matching
        patterns_to_try = [
            re.escape(q),
            re.escape(q.strip().lower()),
            re.escape(re.sub(r'[^\w\s]', '', q.strip().lower())),
        ]
        
        for pattern_str in patterns_to_try:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(transcript):
                pos = match.start()
                # Check if it's a good match (at sentence boundary)
                if pos == 0 or transcript[max(0, pos-2):pos].strip() in ('', '.', '!', '?', '\n'):
                    if pos < earliest_question_pos:
                        earliest_question_pos = pos
                    break  # Found a match for this question, move to next
    
    # Truncate transcript to end BEFORE the first question
    if earliest_question_pos < len(transcript):
        transcript = transcript[:earliest_question_pos].rstrip()
        # Clean up: remove trailing sentence fragments
        transcript = re.sub(r'[.!?]\s*$', '', transcript).strip()
    
    return transcript


def pre_truncate_transcript(transcript: str, max_chars: int = 2000) -> str:
    """
    Pre-truncate transcript based on character length (safe buffer for tokenization).
    This avoids tokenization during every batch.
    
    Args:
        transcript: Transcript text (already with questions removed)
        max_chars: Maximum characters (safe buffer, actual tokens will be less)
    
    Returns:
        Truncated transcript with "..." prefix if truncated
    """
    if len(transcript) <= max_chars:
        return transcript
    
    # Take the last N characters (removes from beginning, keeps end)
    truncated = transcript[-max_chars:]
    
    # Add ellipsis at the beginning to indicate truncation
    if max_chars > 10:
        truncated = "..." + truncated[3:]
    
    return truncated


def process_single_label_file(
    label_file_path: str,
    video_path: str,
    srt_dir: Path,
    corr_dict: Dict[str, str],
    videos_dir: Path,
    season: int,
    window_before: float,
    max_frames: int,
    use_full_video: float,
    full_video_fps: float,
    full_video_max_frames: int,
    max_collages: int
) -> tuple:
    """
    Worker function to process a single label file.
    Returns (file_data_list, skipped_no_answer, skipped_no_video, processed_count, error_msg)
    """
    try:
        # Re-import needed modules (for multiprocessing)
        import json
        import os
        import sys
        from pathlib import Path
        
        # Add project root to path for imports
        script_dir = Path(__file__).parent.parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        from scripts.srt_parser import find_srt_file, extract_transcript_window
        from scripts.visualize_vqa_structure import parse_timestamp
        from src.text_cleaning import clean_text, clean_answer_text
        
        # Import dependencies for extract_frames_for_entry to avoid circular imports
        from src.video_utils import extract_frames_between, sample_full_video_as_collages
        from src.dataset_builder import frames_to_list
        from typing import Optional, Any, List, Dict
        import numpy as np
        from PIL import Image
        
        # Define extract_frames_for_entry locally to avoid circular import in multiprocessing
        def extract_frames_for_entry(
            video_path: str,
            entry: Dict[str, Any],
            pause_fps: float = 0.25,
            pause_max_frames: int = 4,
            max_size: int = 448,
            use_full_video: bool = False,
            full_video_fps: float = 0.2,
            full_video_max_frames: int = 40,
            grid_cols: int = 2,
            grid_rows: int = 2,
            max_collages: int = 3,
            clip: Optional[Any] = None
        ) -> List[Dict[str, Any]]:
            """Local version to avoid circular imports in multiprocessing"""
            pause_start = parse_timestamp(entry.get("pause_start", ""))
            pause_end = parse_timestamp(entry.get("pause_end", ""))
            if pause_start == pause_end:
                pause_start -= 1
                pause_end += 1           
            frames = []
            
            if use_full_video and pause_start > 0:
                collages = sample_full_video_as_collages(
                    video_path,
                    pause_start=pause_start,
                    fps=full_video_fps,
                    max_frames=full_video_max_frames,
                    grid_cols=grid_cols,
                    grid_rows=grid_rows,
                    max_collages=max_collages,
                    tile_size=max_size,
                    clip=clip
                )
                frames.extend(collages)
            
            if pause_start > 0 and pause_end > pause_start:
                pause_frames = extract_frames_between(
                    video_path,
                    pause_start,
                    pause_end,
                    fps=pause_fps,
                    max_frames=pause_max_frames,
                    max_size=max_size,
                    clip=clip
                )
                frames.extend(pause_frames)
            
            frame_list = frames_to_list(frames)
            
            preprocessed_frames = []
            for frame_dict in frame_list:
                img_array = frame_dict.get("image")
                if img_array is not None:
                    if isinstance(img_array, np.ndarray):
                        img = Image.fromarray(img_array)
                    else:
                        img = Image.fromarray(np.array(img_array, dtype=np.uint8))
                    
                    img = img.resize((448, 448), Image.Resampling.LANCZOS)
                    img_array = np.array(img, dtype=np.uint8)
                    
                    preprocessed_frames.append({
                        "image": img_array.tolist(),
                        "timestamp": frame_dict.get("timestamp", 0.0)
                    })
            
            return preprocessed_frames
        
        file_data = []
        skipped_no_answer = 0
        processed_count = 0
        
        # Load labels
        with open(label_file_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        
        # Check if video exists
        if not os.path.exists(video_path):
            return ([], 0, len(entries), 0, f"Video not found: {video_path}")
        
        # Find SRT file
        srt_path = find_srt_file(str(video_path), str(srt_dir))
        
        # Process each entry with timing and video clip caching
        import time as time_module
        
        # Cache video clip to avoid reopening for each entry
        try:
            try:
                from moviepy import VideoFileClip
            except ImportError:
                from moviepy.editor import VideoFileClip
            
            video_clip = VideoFileClip(video_path)
        except Exception as e:
            return ([], 0, len(entries), 0, f"Failed to open video {video_path}: {str(e)}")
        
        t_total_start = time_module.time()
        t_transcript_total = 0
        t_frames_total = 0
        
        try:
            for entry_idx, entry in enumerate(entries):
                if not entry.get("answer_text", "").strip():
                    skipped_no_answer += 1
                    continue
                
                t_entry_start = time_module.time()
                
                # Extract transcript
                t_transcript_start = time_module.time()
                transcript = ""
                question_start = parse_timestamp(entry.get("question_start_ts", ""))
                
                if srt_path and question_start > 0:
                    try:
                        transcript = extract_transcript_window(srt_path, question_start, window_before=window_before)
                    except Exception as e:
                        print("transcript error", e)
                else:
                    print("Garama!", srt_path, question_start)
                if not transcript:
                    transcript = entry.get("transcript", "")
                
                transcript = clean_text(transcript)
                # print("transcript len, ", transcript)
                answer = clean_answer_text(entry.get("answer_text", ""))
                t_transcript_total += time_module.time() - t_transcript_start
                # Extract frames - use cached video clip
                t_frames_start = time_module.time()
                frames = extract_frames_for_entry(
                    str(video_path),
                    entry,
                    pause_fps=0.25,
                    pause_max_frames=max_frames,
                    max_size=448,
                    use_full_video=True if random.random() < use_full_video else False,
                    full_video_fps=full_video_fps,
                    full_video_max_frames=full_video_max_frames,
                    grid_cols=2,
                    grid_rows=2,
                    max_collages=max_collages,
                    clip=video_clip  # Pass cached clip
                )
                if len(frames) == 0:
                    print(entry, "no frame associated")
                    continue
                t_frames_total += time_module.time() - t_frames_start
                
                dataset_entry = {
                    "transcript": transcript,
                    "question": entry.get("question_text", ""),
                    "answer": answer,
                    "frames": frames,
                    "video_path": str(video_path),
                    "season": season,
                    "episode": entry.get("episode", ""),
                    "entry_id": entry.get("id", ""),
                }
                
                file_data.append(dataset_entry)
                processed_count += 1
                
                # Log timing for first few entries
                if entry_idx < 3:
                    t_entry = time_module.time() - t_entry_start
                    t_transcript_elapsed = time_module.time() - t_transcript_start
                    t_frames_elapsed = time_module.time() - t_frames_start
                    print(f"      [TIMING] Entry {entry_idx}: total={t_entry*1000:.1f}ms, transcript={t_transcript_elapsed*1000:.1f}ms, frames={t_frames_elapsed*1000:.1f}ms")
        finally:
            # Close video clip
            try:
                video_clip.close()
            except:
                pass
        
        t_total = time_module.time() - t_total_start
        if processed_count > 0:
            avg_time = t_total / processed_count
            print(f"      [TIMING] File {Path(label_file_path).name}: {processed_count} entries, total={t_total:.2f}s, avg={avg_time*1000:.1f}ms/entry, transcript={t_transcript_total:.2f}s, frames={t_frames_total:.2f}s")
        
        return (file_data, skipped_no_answer, 0, processed_count, None)
    except Exception as e:
        import traceback
        return ([], 0, 0, 0, f"Error processing {label_file_path}: {str(e)}\n{traceback.format_exc()}")


def get_season_ep_idx(lbl_pth, mp4_pth):
    the_dict = {}
    for season in os.listdir(lbl_pth):
        if season == "S1" or season == "S2":
            for ep in os.listdir(os.path.join(lbl_pth, season)):
                # Dora.the.explorer.s01e01.avi
                ep_idx = ep.split(".")[3].lower()
                mp4_name = "Dora.the.explorer." + ep_idx + ".avi"
                the_dict[ep] = os.path.join(season, mp4_name)
                if ep_idx == "s02e25":
                    the_dict[ep] = os.path.join(season, "英文版第2季-25.Whose Birthday Is It.avi")
                elif ep_idx == "s02e26":
                    the_dict[ep] = os.path.join(season, "英文版第2季-26.Quack! Quack!.avi")
        elif season == "S3" or season == "S4" or season == "S5":
            # "英文版第4季-01.Dora Starcatching Adventure_formatted_filtered.json"
            for ep in os.listdir(os.path.join(lbl_pth, season)):
                the_dict[ep] = os.path.join(season, ep.split("_")[0]+".avi")
        else:
            for ep in os.listdir(os.path.join(lbl_pth, season)):
                the_dict[ep] = os.path.join(season, ep.split("_")[0]+".mp4")
    return the_dict


def process_season(
    season: int,
    labels_dir: Path,
    videos_dir: Path,
    srt_dir: Path,
    window_before: float = 20.0,
    max_frames: int = 4,
    use_full_video: float = 0.3,
    full_video_fps: float = 0.2,
    full_video_max_frames: int = 40,
    max_collages: int = 3,
    num_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    Process all episodes in a season.
    
    Returns:
        List of dataset entries
    """
    if os.path.exists(f"./temp/{season}.pickle"):
        with open(f"./temp/{season}.pickle", 'rb') as f:
            # Deserialize the data from the file
            season_data = pickle.load(f)
            print(f"Season {season} data loaded.")
            return season_data
        # return []

    season_data = []
    corr_dict = get_season_ep_idx(labels_dir, videos_dir)

    
    # Find all label files for this season
    season_labels_dir = labels_dir / f"S{season}"
    if not season_labels_dir.exists():
        print(f"  ⚠️  Season {season} labels directory not found: {season_labels_dir}")
        return season_data
    
    # Try filtered files first, then formatted files, then regular JSON files
    label_files = sorted(season_labels_dir.glob("*_filtered.json"))
    if not label_files:
        label_files = sorted(season_labels_dir.glob("*_formatted.json"))
    if not label_files:
        label_files = sorted(season_labels_dir.glob("*.json"))
    print(f"  Found {len(label_files)} label files")
    
    skipped_no_answer_total = 0
    skipped_no_video_total = 0
    processed_count_total = 0
    
    # Prepare tasks for parallel processing
    tasks = []
    for label_file in label_files:
        video_path = os.path.join(videos_dir, corr_dict[label_file.stem+".json"])
        tasks.append((
            str(label_file),
            video_path,
            srt_dir,
            corr_dict,
            videos_dir,
            season,
            window_before,
            max_frames,
            use_full_video,
            full_video_fps,
            full_video_max_frames,
            max_collages
        ))
    
    # Process files in parallel or sequentially
    if num_workers > 1 and len(tasks) > 1:
        print(f"  Processing {len(tasks)} files with {num_workers} workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_label_file, *task): task[0] for task in tasks}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                label_file_path = futures[future]
                try:
                    # Add timeout to prevent hanging
                    file_data, skipped_no_answer, skipped_no_video, processed_count, error_msg = future.result(timeout=600)  # 10 min timeout per file
                    
                    if error_msg:
                        print(f"    ⚠️  {error_msg}")
                        skipped_no_video_total += skipped_no_video
                        continue
                    
                    season_data.extend(file_data)
                    skipped_no_answer_total += skipped_no_answer
                    skipped_no_video_total += skipped_no_video
                    processed_count_total += processed_count
                    
                    # Show progress
                    print(f"    [{completed}/{len(tasks)}] {Path(label_file_path).name}: {processed_count} entries processed")
                    
                    if skipped_no_answer > 0:
                        print(f"      [DEBUG] Skipped no_answer={skipped_no_answer}")
                except Exception as e:
                    print(f"    ⚠️  Error processing {Path(label_file_path).name}: {e}")
                    import traceback
                    print(traceback.format_exc())
    else:
        # Sequential processing (fallback or num_workers=1)
        for label_file in tqdm(label_files, desc=f"  Processing S{season}"):
            # Load labels
            print(label_file)
            with open(label_file, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            
            # Find corresponding video file
            video_path = os.path.join(videos_dir, corr_dict[label_file.stem+".json"])
            if not os.path.exists(video_path):
                print(f"    ⚠️  Video not found for {video_path}, skipping")
                skipped_no_video_total += len(entries)
                continue
            
            # Find SRT file
            srt_path = find_srt_file(str(video_path), str(srt_dir))
            
            # Process each entry
            skipped_no_answer = 0
            processed_count = 0
            for entry in entries:
                # Skip if no answer
                if not entry.get("answer_text", "").strip():
                    skipped_no_answer += 1
                    continue
                
                # Extract transcript from SRT (20s window before question, stops before question)
                transcript = ""
                question_start = parse_timestamp(entry.get("question_start_ts", ""))
                
                if srt_path and question_start > 0:
                    try:
                        transcript = extract_transcript_window(srt_path, question_start, window_before=window_before)
                    except Exception as e:
                        print(f"    ⚠️  Error extracting transcript: {e}")
                
                # Fallback to entry transcript if SRT extraction failed
                if not transcript:
                    transcript = entry.get("transcript", "")
                
                # Clean transcript to remove singing symbols and non-textual characters
                transcript = clean_text(transcript)
                
                # Clean answer text (already done in validation, but ensure consistency)
                answer = clean_answer_text(entry.get("answer_text", ""))
                
                # Extract frames using existing video_utils
                frames = extract_frames_for_entry(
                    str(video_path), 
                    entry, 
                    pause_fps=0.25,
                    pause_max_frames=max_frames,
                    max_size=448,
                    use_full_video=use_full_video,
                    full_video_fps=full_video_fps,
                    full_video_max_frames=full_video_max_frames,
                    grid_cols=2,
                    grid_rows=2,
                    max_collages=max_collages
                )
                
                # Create dataset entry
                dataset_entry = {
                    "transcript": transcript,
                    "question": entry.get("question_text", ""),
                    "answer": answer,
                    "frames": frames,
                    "video_path": str(video_path),
                    "season": season,
                    "episode": entry.get("episode", ""),
                    "entry_id": entry.get("id", ""),
                }
                if len(frames) == 0:
                    print("an issue occured. Garama!")
                
                season_data.append(dataset_entry)
                processed_count += 1
            
            skipped_no_answer_total += skipped_no_answer
            processed_count_total += processed_count
            
            if skipped_no_answer > 0:
                print(f"    [DEBUG] File {label_file.name}: Processed {processed_count}, skipped no_answer={skipped_no_answer}")
    
    if skipped_no_answer_total > 0 or skipped_no_video_total > 0:
        print(f"  [DEBUG Season {season}] Total: processed={processed_count_total}, skipped_no_answer={skipped_no_answer_total}, skipped_no_video={skipped_no_video_total}")
    
    os.makedirs("./temp", exist_ok=True)
    with open(f'./temp/{season}.pickle', 'wb') as f:
        pickle.dump(season_data, f)
    return season_data

def gen(labels_dir, videos_dir, srt_dir, args):
    # First pass: collect all data
    print("\n[Preprocessing] Collecting all data...")
    for season in args.seasons:
        t0 = time.time()
        print(f"\nProcessing Season {season}...")
        season_start = time.time()
        season_data = process_season(
            season,
            labels_dir,
            videos_dir,
            srt_dir,
            window_before=args.window_before,
            max_frames=args.max_frames,
            use_full_video=getattr(args, 'use_full_video', 0.3),
            full_video_fps=getattr(args, 'full_video_fps', 0.2),
            full_video_max_frames=getattr(args, 'full_video_max_frames', 40),
            max_collages=getattr(args, 'max_collages', 3),
            num_workers=getattr(args, 'num_workers', 4)
        )
        # all_data.extend(season_data)
        season_time = time.time() - season_start
        print(f"  ✓ Season {season} loaded: {len(season_data)} entries ({season_time:.2f}s)")
    
        collection_time = time.time() - t0
        print(f"[TIMING] Data collection: {collection_time:.2f}s for {len(season_data)} items")
        
        # Collect all unique questions for preprocessing
        print("\n[Preprocessing] Collecting all questions...")
        t1 = time.time()
        all_questions = []
        for item in season_data:
            question = item.get("question", "").strip()
            if question and question not in all_questions:
                all_questions.append(question)
        question_collection_time = time.time() - t1
        print(f"  ✓ Found {len(all_questions)} unique questions ({question_collection_time:.2f}s)")
        
        # Second pass: preprocess each item
        print("\n[Preprocessing] Removing questions from transcripts and truncating...")
        t2 = time.time()
        # Truncate to ~512 tokens: conservative estimate is ~4 chars per token, so 512 tokens ≈ 2000 chars
        # But we want to be more precise - use 1500 chars to ensure we stay under 512 tokens
        max_transcript_chars = 1500  # Conservative: 512 tokens * ~3 chars/token = ~1500 chars (safe buffer)
        
        for i, item in enumerate(season_data):
            transcript = item.get("transcript", "")
            # print("1:",transcript)
            question = item.get("question", "")
            
            # Remove all questions from transcript
            # transcript = remove_questions_from_transcript(transcript, all_questions)
            # print("2:",transcript)
            # Pre-truncate transcript (character-based, safe buffer)
            transcript = pre_truncate_transcript(transcript, max_chars=max_transcript_chars)
            # print("3:",transcript)
            # Update item with preprocessed transcript
            item["transcript"] = transcript
            
            # Add flag to indicate preprocessing is done
            item["_preprocessed"] = True
            
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t2
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  ✓ Preprocessed {i + 1}/{len(season_data)} items ({rate:.1f} items/s)")
        
        preprocessing_time = time.time() - t2
        print(f"  ✓ Preprocessing complete: {len(season_data)} items ({preprocessing_time:.2f}s)")
        print(f"[TIMING] Total preprocessing: {preprocessing_time:.2f}s")
        
        # Yield preprocessed items
        for item in season_data:
            if "frames" in item and len(item["transcript"]) > 0:
                yield item
            else:
                print("trans:", item["transcript"])
                print("frames" in item)
                print("Garama! frames not in data field or transcript len 0")



def main():
    parser = argparse.ArgumentParser(description="Generate GRPO training dataset from filtered labels")
    parser.add_argument("--labels-dir", type=str, default="outputs/filtered_labels",
                       help="Directory containing filtered label files (S1/, S2/, etc.)")
    parser.add_argument("--videos-dir", type=str, default="/mnt/data/dora/mp4",
                       help="Directory containing video files (S1/, S2/, etc.)")
    parser.add_argument("--srt-dir", type=str, default="/mnt/work/XXXX-7/dora/dataset/srt",
                       help="Directory containing SRT subtitle files")
    parser.add_argument("--output-dir", type=str, default="outputs/dataset",
                       help="Output directory for HuggingFace Dataset")
    parser.add_argument("--seasons", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8],
                       help="Seasons to process")
    parser.add_argument("--window-before", type=float, default=60.0,
                       help="Seconds before question to include in transcript window")
    parser.add_argument("--max-frames", type=int, default=4,
                       help="Maximum number of frames to extract per entry")
    parser.add_argument("--use-full-video", type=float, default=0.3,
                       help="the percentage of randomly Include frames from video start to pause (full video context)")
    parser.add_argument("--full-video-fps", type=float, default=0.2,
                       help="FPS for sampling full video context (default: 0.2)")
    parser.add_argument("--full-video-max-frames", type=int, default=40,
                       help="Maximum frames for full video context (default: 40)")
    parser.add_argument("--max-collages", type=int, default=3,
                       help="Maximum number of collages for full video context (default: 3)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of parallel workers for processing videos (default: 4, set to 1 to disable parallelization)")
    
    args = parser.parse_args()
    
    labels_dir = Path(args.labels_dir)
    videos_dir = Path(args.videos_dir)
    srt_dir = Path(args.srt_dir)
    output_dir = Path(args.output_dir)
    
    # Validate directories
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return 1
    
    if not videos_dir.exists():
        print(f"❌ Videos directory not found: {videos_dir}")
        return 1
    
    if not srt_dir.exists():
        print(f"❌ SRT directory not found: {srt_dir}")
        return 1
    
    # Process all seasons
    print("=" * 80)
    print("Generating GRPO Training Dataset")
    print("=" * 80)
    print(f"Labels directory: {labels_dir}")
    print(f"Videos directory: {videos_dir}")
    print(f"SRT directory: {srt_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Window before question: {args.window_before}s")
    print(f"Max frames per entry: {args.max_frames}")
    print(f"Seasons to process: {args.seasons}")
    print("=" * 80)
    

    
    # Create HuggingFace Dataset
    print("\nCreating HuggingFace Dataset...")
    dataset = Dataset.from_generator(gen, 
    gen_kwargs={"labels_dir":labels_dir, "videos_dir":videos_dir, "srt_dir":srt_dir, "args":args}
    )
    
    # Save dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(str(output_dir))
    
    # Print summary
    print("\n" + "=" * 80)
    print("✅ Dataset Generation Complete")
    print("=" * 80)
    print(f"Total entries: {len(dataset)}")
    print(f"Output directory: {output_dir}")
    print("\nDataset fields:")
    if len(dataset) > 0:
        sample = dataset[0]
        for key in sample.keys():
            if key == "frames":
                print(f"  - {key}: List of {len(sample[key])} frame dicts")
            else:
                print(f"  - {key}: {type(sample[key]).__name__}")
    
    print("\nTo use this dataset for training:")
    print(f"  python scripts/train_grpo_vl.py {output_dir} --use-frames")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

# python scripts/generate_grpo_dataset.py     --labels-dir /projects/XXXX-1/dora/filtered_labels     --videos-dir /projects/XXXX-1/dora/mp4/mp4     --srt-dir /projects/XXXX-1/dora/srt/srt     --output-dir /projects/XXXX-1/dora/grpo_dataset_test --seasons 1 2 3 4 5 6 7 8

