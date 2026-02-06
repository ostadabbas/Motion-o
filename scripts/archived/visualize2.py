#!/usr/bin/env python3
"""
Extract representative frames for each spatial reasoning category.
Saves frames that can be embedded in the visualization chart.
"""
import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.visualize_vqa_structure import parse_timestamp, extract_frame
import os

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip


from evaluate_grpo_vl_simple import load_model_and_processor, generate_answer
from src.grpo_dataset import DoraGRPODataset
from src.grpo_reward import extract_final_answer, tokenize_answer
from src.ppo_trainer_simple import string_f1
from tqdm import tqdm
from datasets import load_from_disk
from collections import defaultdict
import json
from torch.utils.data import DataLoader
import random

from multiprocessing import Pool

def extract(idx_item):
    idx, item = idx_item
    return item["question"], idx

def build_dataset_mcq_dict(dora_dataset):
    cache_path = "./temp/dataset_dict.json"

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    # Parallel iteration over dataset
    with Pool(processes=8) as p:
        results = list(
            tqdm(
                p.imap(extract, enumerate(dora_dataset)),
                total=len(dora_dataset)
            )
        )

    the_dict = dict(results)

    with open(cache_path, "w") as f:
        json.dump(the_dict, f)

    return the_dict

def build_mcq_index(mcq_pth):
    with open(mcq_pth, "r") as f:
        mcq_lib = json.load(f)
    the_dict = {}
    for item in mcq_lib:
        the_dict[item["question"]] = item
    return the_dict

def load_model(model_base, ckpt_pth=None):
    if ckpt_pth is not None:
        model, processor = load_model_and_processor(
                model_base,
                checkpoint_path=ckpt_pth,
                use_lora=True,
            )
    else:
        model, processor = load_model_and_processor(
                model_base,
                use_lora=True,
            )
    return model, processor


def answer_question(model, processor, item):
    max_new_tokens = 256
    messages = item["prompt"]
    generated_text = generate_answer(model, processor, messages, max_new_tokens)
    pred_answer = extract_final_answer(generated_text)
    return pred_answer


def find_video_path(episode: str, season: int, videos_dir: Path, corr_dict: Dict[str, str]) -> Optional[str]:
    """Find video file path for an episode."""
    # Try to get from correlation dict first
    if episode in corr_dict:
        video_path = videos_dir / corr_dict[episode]
        if video_path.exists():
            return str(video_path)
    
    # Try to find by episode pattern
    season_dir = videos_dir / f"S{season}"
    if not season_dir.exists():
        season_dir = videos_dir / f"S{season:02d}"
    
    if not season_dir.exists():
        return None
    
    # Look for video files matching episode
    episode_lower = episode.lower()
    for video_file in season_dir.glob("*"):
        if episode_lower in video_file.name.lower():
            return str(video_file)
    
    return None


def extract_frame_from_video(video_path: str, timestamp: float, output_path: Path, size: tuple = (448, 448)) -> bool:
    """Extract a single frame from video at timestamp and save it."""
    try:
        clip = VideoFileClip(video_path)
        frame = clip.get_frame(timestamp)
        clip.close()
        
        # Convert to PIL Image
        img = Image.fromarray(frame.astype(np.uint8))
        
        # Resize
        img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Save
        img.save(output_path)
        return True
    except Exception as e:
        print(f"  ⚠️  Error extracting frame from {video_path} at {timestamp}s: {e}")
        return False


def load_original_labels(episode: str, season: int, labels_dir: Path) -> Optional[Dict]:
    """Load original label file to get timestamps."""
    season_dir = labels_dir / f"S{season}"
    if not season_dir.exists():
        season_dir = labels_dir / f"S{season:02d}"
    
    if not season_dir.exists():
        return None
    
    # Look for label file matching episode
    episode_lower = episode.lower()
    for label_file in season_dir.glob("*.json"):
        if episode_lower in label_file.name.lower() or "filtered" in label_file.name.lower():
            try:
                with open(label_file, 'r') as f:
                    labels = json.load(f)
                return labels
            except:
                continue
    
    return None


def get_entry_timestamp(entry_id: str, episode: str, season: int, labels_dir: Path) -> Optional[float]:
    """Get pause_start timestamp for an entry from original labels."""
    labels = load_original_labels(episode, season, labels_dir)
    if not labels:
        return None
    
    # Find entry by ID
    for entry in labels:
        if entry.get("id") == entry_id or entry.get("entry_id") == entry_id:
            pause_start = entry.get("pause_start", "")
            if pause_start:
                return parse_timestamp(pause_start)
    
    return None


def extract_category_frames(
    mcq_dataset_path: str,
    videos_dir: str,
    labels_dir: str,
    output_dir: str,
    frames_per_category: int = 3
):
    """Extract representative frames for each spatial reasoning category."""
    
    dataset_path = "/scratch/XXXX-6.XXXX-7/grpo_dataset_updatedv2"
    model, processor = load_model("Qwen/Qwen3-VL-8B-Instruct", "./outputs/train_q3/checkpoint-250")
    dataset = load_from_disk(dataset_path)
    dataset = DoraGRPODataset(
            dataset=dataset,
            processor=processor,
            max_prompt_length=512,
            max_completion_length=256,
            use_frames=True,
            max_frames=4,
            system_prompt="You are a helpful visual reasoning assistant for kids. \nThink step by step and always give a final concise answer in the first sentence.",
        )
    search_dict = build_dataset_mcq_dict(dataset)

    mcq_path = Path(mcq_dataset_path)
    mcq_dict = build_mcq_index(mcq_path)
    videos_dir = Path(videos_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Extracting Representative Frames for Spatial Reasoning Categories")
    print("="*80)
    
    # Load MCQ dataset
    print(f"\nLoading MCQ dataset from {mcq_path}...")
    with open(mcq_path, 'r') as f:
        mcq_data = json.load(f)
    print(f"✓ Loaded {len(mcq_data)} entries")
    
    # Filter all categories we want to extract frames for
    target_categories = {
        'spatial_location': [],
        'object_selection': [],
        'navigation': [],
        'counting': [],
        'problem_solving': [],
        'recall_knowledge': [],
        'language': []
    }
    
    
    # Get video correlation dict (for mapping episodes to video files)
    print(f"\nLoading video correlation dictionary...")
    corr_dict = {}
    for season in range(1, 9):
        season_labels_dir = labels_dir / f"S{season}"
        if not season_labels_dir.exists():
            season_labels_dir = labels_dir / f"S{season:02d}"
        if not season_labels_dir.exists():
            continue
        
        season_videos_dir = videos_dir / f"S{season}"
        if not season_videos_dir.exists():
            season_videos_dir = videos_dir / f"S{season:02d}"
        if not season_videos_dir.exists():
            continue
        
        # Build correlation dict for this season
        for label_file in season_labels_dir.glob("*.json"):
            label_name = label_file.stem
            # Extract episode pattern from label filename
            if season <= 2:
                # Format: Dora.the.explorer.s01e01.avi
                ep_match = re.search(r's(\d+)e(\d+)', label_name, re.IGNORECASE)
                if ep_match:
                    ep_idx = ep_match.group(0).lower()
                    video_name = f"Dora.the.explorer.{ep_idx}.avi"
                    corr_dict[label_name] = str(season_videos_dir / video_name)
            elif season <= 5:
                # Format: 英文版第4季-01.Dora Starcatching Adventure_formatted_filtered.json
                video_name = label_name.split("_")[0] + ".avi"
                corr_dict[label_name] = str(season_videos_dir / video_name)
            else:
                # Format: *.mp4
                video_name = label_name.split("_")[0] + ".mp4"
                corr_dict[label_name] = str(season_videos_dir / video_name)
    
    print(f"✓ Loaded {len(corr_dict)} episode mappings")
    
    # Extract frames for each category
    extracted_frames = {}
    num_examples = len(eval_dataset)
    select = list(range(num_examples))
    random.shuffle(select)
    for i in tqdm(range(num_examples)):
        # for batch in loader:
        random_idx = select[i]
        item = dataset[random_idx]
        # item = {k: v[0] for k, v in batch.items()}
        messages = item["prompt"]
        question = item["question"]
        if question not in mcq_dict.keys():
            print("Skipped question ", question)
            continue
        entry = mcq_dict[question]
        category = entry.get('reasoning_category', '')
        if category in target_categories:
            target_categories[category].append(entry)
    
    for category, entries in target_categories.items():
        print(f"\n{'='*80}")
        print(f"Extracting frames for: {category}")
        print(f"{'='*80}")
        
        if not entries:
            print(f"  ⚠️  No entries found for {category}")
            continue
        
        # Select representative entries (first N with valid episode/video)
        selected_entries = []
        for entry in entries[:frames_per_category * 3]:  # Try more to find valid ones
            episode = entry.get('episode', '')
            entry_id = entry.get('entry_id', '')
            season = entry.get('season', 0)
            
            if not episode or not entry_id or not season:
                continue
            
            # Find video path
            video_path = find_video_path(episode, season, videos_dir, corr_dict)
            if not video_path or not Path(video_path).exists():
                continue
            
            # Get timestamp from original labels
            timestamp = get_entry_timestamp(entry_id, episode, season, labels_dir)
            if timestamp is None or timestamp <= 0:
                continue
            
            selected_entries.append({
                'entry': entry,
                'video_path': video_path,
                'timestamp': timestamp,
                'episode': episode,
                'entry_id': entry_id
            })
            
            if len(selected_entries) >= frames_per_category:
                break
        
        print(f"  Found {len(selected_entries)} valid entries with video/timestamps")
        
        # Extract frames
        category_frames = []
        for idx, item in enumerate(selected_entries[:frames_per_category], 1):
            episode = item['episode']
            entry_id = item['entry_id']
            video_path = item['video_path']
            timestamp = item['timestamp']
            question = item['entry'].get('question', '')
            answer = ""
            if question in search_dict.keys():
                message_idx = search_dict[question]
                message = eval_dataset[message_idx]
                answer = answer_question(model, processor, message)
            else:
                print("not found, ", question)
                continue
            
            # Create output filename
            frame_filename = f"{category}_{episode}_{entry_id}_frame{idx}.png"
            frame_path = output_dir / frame_filename
            
            print(f"  [{idx}/{frames_per_category}] Extracting from {Path(video_path).name} at {timestamp:.1f}s...")
            print(f"      Q: {question}...")
            
            if extract_frame_from_video(video_path, timestamp, frame_path):
                category_frames.append({
                    'path': str(frame_path),
                    'episode': episode,
                    'entry_id': entry_id,
                    'question': item['entry'].get('question', ''),
                    'answer': answer,
                    'timestamp': timestamp
                })
                print(f"      ✓ Saved: {frame_filename}")
            else:
                print(f"      ✗ Failed to extract frame")
        
        extracted_frames[category] = category_frames
    
    # Save metadata
    metadata_path = output_dir / "frame_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(extracted_frames, f, indent=2)
    print(f"\n✓ Saved frame metadata to {metadata_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("Extraction Summary")
    print(f"{'='*80}")
    for category, frames in extracted_frames.items():
        print(f"{category}: {len(frames)} frames extracted")
        for frame_info in frames:
            print(f"  - {Path(frame_info['path']).name}")
    print(f"{'='*80}")
    
    return extracted_frames


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract representative frames for spatial reasoning categories")
    parser.add_argument("--mcq-dataset", type=str, default="outputs/mcq_dataset/mcq_dataset.json",
                       help="Path to MCQ dataset JSON")
    parser.add_argument("--videos-dir", type=str, default="/mnt/data/dora/mp4",
                       help="Directory containing video files")
    parser.add_argument("--labels-dir", type=str, default="outputs/filtered_labels",
                       help="Directory containing original label files")
    parser.add_argument("--output-dir", type=str, default="outputs/mcq_dataset/category_frames",
                       help="Output directory for extracted frames")
    parser.add_argument("--frames-per-category", type=int, default=3,
                       help="Number of frames to extract per category")
    
    args = parser.parse_args()
    
    extract_category_frames(
        mcq_dataset_path=args.mcq_dataset,
        videos_dir=args.videos_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        frames_per_category=args.frames_per_category
    )