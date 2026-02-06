#!/usr/bin/env python3
"""
Convert PLM-STC (HuggingFace) + SA-V (Meta) → Preprocessing Format.

Transforms:
  Input:
    - /mnt/data/plm_stc/raw/rdcap/ (HF dataset with dense captions)
    - /mnt/data/plm_stc/raw/sa-v/ (videos + masklet JSONs with RLE masks)
  
  Output:
    - /mnt/data/plm_stc/formatted/
      ├── videos/ (symlinks to sa-v/)
      ├── masklets/ (decoded RLE → .npy arrays)
      └── annotations/train.json (question, answer, evidence_steps)
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from datasets import load_from_disk
import cv2


def decode_rle_mask(rle_dict: Dict, img_height: int, img_width: int) -> np.ndarray:
    """
    Decode RLE-encoded mask to binary numpy array.
    
    SA-V uses COCO-style RLE format:
    {
        "size": [height, width],
        "counts": "RLE_string"
    }
    
    Args:
        rle_dict: RLE dictionary with 'size' and 'counts'
        img_height: Image height
        img_width: Image width
    
    Returns:
        Binary mask array of shape (height, width)
    """
    try:
        from pycocotools import mask as mask_utils
        
        # Ensure size matches expected dimensions
        if 'size' in rle_dict:
            rle_height, rle_width = rle_dict['size']
            if rle_height != img_height or rle_width != img_width:
                print(f"  Warning: RLE size {rle_dict['size']} != expected {(img_height, img_width)}")
        
        # Decode RLE
        mask = mask_utils.decode(rle_dict)
        return mask.astype(np.uint8)
    
    except ImportError:
        print("ERROR: pycocotools not installed. Install with: pip install pycocotools")
        raise
    except Exception as e:
        print(f"  Warning: Failed to decode RLE mask: {e}")
        return np.zeros((img_height, img_width), dtype=np.uint8)


def load_sa_v_masklet(video_id: str, sa_v_dir: Path, masklet_id: int, 
                       prefer_manual: bool = True) -> Optional[Dict]:
    """
    Load masklet data from SA-V JSON files.
    
    SA-V format: Each JSON file contains ONE video's masklets.
    - 'masklet_id': List of object IDs (e.g., [0, 1, 2, 3, 4])
    - 'masklet': List of frames, each frame is list of RLE masks per object
    
    Args:
        video_id: Video ID without .mp4 extension (e.g., 'sav_000866')
        sa_v_dir: Directory containing SA-V data
        masklet_id: Masklet ID to extract (index into masklet_id list)
        prefer_manual: Try manual annotations first, fallback to auto
    
    Returns:
        Dictionary with 'masklet' (frame list) and 'masklet_id', or None if not found
    """
    # Try manual first (higher quality), then auto
    suffixes = ['_manual.json', '_auto.json'] if prefer_manual else ['_auto.json', '_manual.json']
    
    for suffix in suffixes:
        masklet_path = sa_v_dir / f"{video_id}{suffix}"
        if not masklet_path.exists():
            continue
        
        try:
            with open(masklet_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            # SA-V format: masklet_id is a list, masklet is frame-by-frame data
            available_ids = data.get('masklet_id', [])
            if not isinstance(available_ids, list):
                available_ids = [available_ids]
            
            # Check if requested masklet_id exists
            if masklet_id in available_ids:
                # Return the full data - we'll extract the specific object later
                return {
                    'masklet': data.get('masklet', []),
                    'masklet_id': masklet_id,
                    'available_ids': available_ids,
                    'video_width': data.get('video_width', 1920),
                    'video_height': data.get('video_height', 1080),
                }
        
        except Exception as e:
            # Silently continue to next suffix
            continue
    
    return None


def convert_masklet_to_numpy(masklet_data: Dict, video_path: Path, 
                              total_frames: int) -> np.ndarray:
    """
    Convert SA-V masklet (RLE per frame) to numpy array.
    
    SA-V format:
    - masklet_data['masklet']: List of frames (each frame = list of objects)
    - masklet_data['masklet'][frame_idx][object_idx]: RLE dict {'size': [H,W], 'counts': str}
    - masklet_data['masklet_id']: Which object index to extract
    
    Args:
        masklet_data: Dictionary with 'masklet' (frame list) and 'masklet_id'
        video_path: Path to video file (for dimensions, if needed)
        total_frames: Total number of frames in the video
    
    Returns:
        Numpy array of shape (num_frames, H, W) with binary masks
    """
    # Get dimensions from masklet_data or video (ensure integers)
    img_width = int(masklet_data.get('video_width', 1920))
    img_height = int(masklet_data.get('video_height', 1080))
    
    # Get which object ID to extract
    target_masklet_id = masklet_data['masklet_id']
    available_ids = masklet_data.get('available_ids', [])
    
    # Find index of target_masklet_id in available_ids
    try:
        object_idx = available_ids.index(target_masklet_id)
    except ValueError:
        # masklet_id not in available_ids
        return np.zeros((total_frames, img_height, img_width), dtype=np.uint8)
    
    # Initialize mask array (all zeros) - ensure all dimensions are integers
    masks = np.zeros((int(total_frames), int(img_height), int(img_width)), dtype=np.uint8)
    
    # Decode each frame's mask for the target object
    masklet_frames = masklet_data.get('masklet', [])
    
    for frame_idx, frame_objects in enumerate(masklet_frames):
        if frame_idx >= total_frames:
            break
        
        # frame_objects is a list of RLE dicts, one per object
        if not isinstance(frame_objects, list) or len(frame_objects) <= object_idx:
            continue
        
        rle_dict = frame_objects[object_idx]
        if rle_dict and isinstance(rle_dict, dict):
            # Parse size if it's a string
            if 'size' in rle_dict and isinstance(rle_dict['size'], str):
                # Size might be '[1920, 1080]' as string
                size_str = rle_dict['size'].strip('[]')
                img_width, img_height = map(int, map(float, size_str.split(',')))
                rle_dict = {'size': [int(img_height), int(img_width)], 'counts': rle_dict['counts']}
            
            masks[frame_idx] = decode_rle_mask(rle_dict, int(img_height), int(img_width))
    
    return masks


def frame_to_time(frame_idx: int, fps: float, total_frames: int) -> float:
    """Convert frame index to time in seconds."""
    duration = total_frames / fps if fps > 0 else 1.0
    return min(frame_idx / fps if fps > 0 else 0.0, duration)


def generate_question_answer(video_id: str, dense_captions: List[Dict]) -> tuple:
    """
    Generate question and answer from dense captions.
    
    For now, use simple template. Can be enhanced later.
    
    Args:
        video_id: Video identifier
        dense_captions: List of {start_frame, end_frame, caption} dicts
    
    Returns:
        (question, answer) tuple
    """
    # Simple question template
    question = "Describe what happens in this video and track the object's motion."
    
    # Answer is concatenation of all captions
    if dense_captions:
        captions_text = " ".join([cap['caption'] for cap in dense_captions])
        answer = captions_text
    else:
        answer = "No description available."
    
    return question, answer


def process_sample(sample: Dict, sa_v_dir: Path, output_masklets_dir: Path,
                   video_fps: Optional[float] = None) -> Optional[Dict]:
    """
    Process a single PLM-STC RDCap sample.
    
    Args:
        sample: RDCap sample with video, masklet_id, dense_captions, etc.
        sa_v_dir: Directory with SA-V videos and masklets
        output_masklets_dir: Output directory for converted .npy masklets
        video_fps: Video FPS (if known, otherwise extracted from video)
    
    Returns:
        Annotation dict for train.json, or None if processing failed
    """
    video_filename = sample['video']
    video_id = video_filename.replace('.mp4', '')
    masklet_id = sample['masklet_id']
    total_frames = sample['total_frames']
    dense_captions = sample.get('dense_captions', [])
    
    # Check if video exists
    video_path = sa_v_dir / video_filename
    if not video_path.exists():
        return None
    
    # Get video FPS
    if video_fps is None:
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 24.0
        cap.release()
    
    # Load masklet from SA-V
    masklet = load_sa_v_masklet(video_id, sa_v_dir, masklet_id)
    if masklet is None:
        return None
    
    # Convert masklet to numpy and save
    try:
        masks = convert_masklet_to_numpy(masklet, video_path, total_frames)
    except Exception as e:
        print(f"  Error converting masklet for {video_id}: {e}")
        return None
    
    # Create evidence steps from dense captions
    evidence_steps = []
    for idx, caption_entry in enumerate(dense_captions):
        start_frame = caption_entry['start_frame']
        end_frame = caption_entry['end_frame']
        caption = caption_entry['caption']
        
        # Convert frames to time
        t_s = frame_to_time(start_frame, video_fps, total_frames)
        t_e = frame_to_time(end_frame, video_fps, total_frames)
        
        # Save masklet for this step
        masklet_filename = f"{video_id}_{idx}.npy"
        masklet_path = output_masklets_dir / masklet_filename
        
        # Extract masks for this frame range
        step_masks = masks[start_frame:end_frame+1]
        np.save(str(masklet_path), step_masks)
        
        evidence_steps.append({
            "t_s": float(t_s),
            "t_e": float(t_e),
            "masklet_path": masklet_filename,
            "caption": caption
        })
    
    # Generate question and answer
    question, answer = generate_question_answer(video_id, dense_captions)
    
    return {
        "video_id": video_id,
        "question": question,
        "answer": answer,
        "evidence_steps": evidence_steps
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert PLM-STC + SA-V to preprocessing format"
    )
    parser.add_argument(
        "--input-annotations",
        type=str,
        default="/mnt/data/plm_stc/raw/rdcap",
        help="Path to HuggingFace RDCap dataset"
    )
    parser.add_argument(
        "--input-videos",
        type=str,
        default="/mnt/data/plm_stc/raw/sa-v",
        help="Path to SA-V videos and masklet JSONs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/data/plm_stc/formatted",
        help="Output directory for formatted dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)"
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Override video FPS (default: extract from video)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_annotations = Path(args.input_annotations)
    input_videos = Path(args.input_videos)
    output_dir = Path(args.output_dir)
    
    output_videos_dir = output_dir / "videos"
    output_masklets_dir = output_dir / "masklets"
    output_annotations_dir = output_dir / "annotations"
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    output_videos_dir.mkdir(exist_ok=True)
    output_masklets_dir.mkdir(exist_ok=True)
    output_annotations_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("PLM-STC + SA-V → Preprocessing Format Conversion")
    print("="*70)
    print(f"Input annotations: {input_annotations}")
    print(f"Input videos: {input_videos}")
    print(f"Output directory: {output_dir}")
    if args.limit:
        print(f"Limit: {args.limit} samples")
    print()
    
    # Load RDCap dataset
    print("Loading RDCap annotations...")
    try:
        rdcap = load_from_disk(str(input_annotations))
        print(f"✓ Loaded {len(rdcap)} samples")
    except Exception as e:
        print(f"ERROR: Failed to load annotations: {e}")
        return
    
    # Limit samples if requested
    if args.limit and args.limit < len(rdcap):
        rdcap = rdcap.select(range(args.limit))
        print(f"  Limited to first {len(rdcap)} samples")
    print()
    
    # Process samples
    print("Processing samples...")
    print("(Creating symlinks, converting masklets, generating annotations)")
    print()
    
    annotations = []
    successful = 0
    failed = 0
    
    for sample in tqdm(rdcap, desc="Converting"):
        video_filename = sample['video']
        video_id = video_filename.replace('.mp4', '')
        
        # Create symlink to video
        src_video = input_videos / video_filename
        dst_video = output_videos_dir / video_filename
        
        if src_video.exists() and not dst_video.exists():
            try:
                os.symlink(src_video, dst_video)
            except Exception as e:
                print(f"  Warning: Failed to create symlink for {video_filename}: {e}")
                failed += 1
                continue
        elif not src_video.exists():
            failed += 1
            continue
        
        # Process sample
        annotation = process_sample(
            sample,
            input_videos,
            output_masklets_dir,
            video_fps=args.video_fps
        )
        
        if annotation:
            annotations.append(annotation)
            successful += 1
        else:
            failed += 1
    
    print()
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print()
    
    # Save annotations
    if annotations:
        output_annotation_file = output_annotations_dir / "train.json"
        print(f"Saving annotations to {output_annotation_file}...")
        with open(output_annotation_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"✓ Saved {len(annotations)} annotations!")
        print()
    
    print("="*70)
    print("SUCCESS!")
    print("="*70)
    print()
    print("Dataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── videos/ ({len(list(output_videos_dir.glob('*.mp4')))} symlinks)")
    print(f"    ├── masklets/ ({len(list(output_masklets_dir.glob('*.npy')))} .npy files)")
    print(f"    └── annotations/train.json ({len(annotations)} samples)")
    print()
    print("Next step:")
    print(f"  python scripts/preprocess_plm_stc.py \\")
    print(f"      {output_dir} \\")
    print(f"      /mnt/data/plm_stc/preprocessed \\")
    print(f"      --split train --max-frames 8")
    print()


if __name__ == "__main__":
    main()
