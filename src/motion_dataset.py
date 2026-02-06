"""
Motion GRPO Dataset for Spatio-Temporal Reasoning.

PyTorch Dataset that formats PLM-STC data for GRPO training with 
spatio-temporal evidence chain prompts.
"""

from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from datasets import Dataset as HFDataset


class MotionGRPODataset(Dataset):
    """
    GRPO Dataset for Motion Reasoning with Spatio-Temporal Evidence Chains.
    
    Formats preprocessed PLM-STC data into GRPO-compatible format with
    chain-of-thought prompts requesting structured evidence.
    """
    
    def __init__(
        self,
        dataset: HFDataset,
        processor,
        max_frames: int = 16,
        system_prompt: str = "You are an expert at analyzing video motion and spatial relationships.",
    ):
        """
        Initialize Motion GRPO Dataset.
        
        Args:
            dataset: HuggingFace Dataset with preprocessed PLM-STC format:
                - video_id, video_path, frames, question, answer
                - gt_evidence_steps: [{t_s, t_e, bboxes, motion_desc, caption}]
            processor: AutoProcessor for the VLM model
            max_frames: Maximum number of frames to include
            system_prompt: System prompt for the model
        """
        super(MotionGRPODataset, self).__init__()
        self.dataset = dataset
        self.processor = processor
        self.max_frames = max_frames
        self.system_prompt = system_prompt
        
        # Disable resize in processor (handled by model)
        if hasattr(processor, 'image_processor'):
            processor.image_processor.do_resize = False
    
    def __len__(self):
        return len(self.dataset)
    
    def _build_chain_prompt(self, question: str, num_frames: int, frame_times: List[float]) -> str:
        """
        Build Think-then-Predict prompt for spatio-temporal reasoning.
        
        Uses the format proven to work in test_think_bbox_inference.py:
        - Explicit frame listing
        - Think: (x1,y1),(x2,y2) - rough estimate
        - Predict: (x1,y1),(x2,y2) - refined bbox
        - Motion: quantifiable descriptors
        
        Args:
            question: Motion-related question
            num_frames: Number of frames shown
            frame_times: List of frame timestamps in seconds
        
        Returns:
            Formatted prompt string
        """
        # Build frame time listing
        frame_list = "\n".join([
            f"  Frame {i+1} at t={t:.2f}s" 
            for i, t in enumerate(frame_times)
        ])
        
        prompt = f"""{question}

You are shown {num_frames} frames from the video at these times:
{frame_list}

For EACH frame, locate relevant objects and provide:

Step N: [time] Description
  Think: (x1,y1),(x2,y2) - rough estimate
  Predict: (x1,y1),(x2,y2) - refined bbox
  Motion: how the object moved

Coordinates use 0-1000 scale (0=left/top, 1000=right/bottom).

Example format:
Step 1: [0.0s] Ball on left side
  Think: (150,400),(250,600)
  Predict: (160,420),(240,580)
  Motion: Starting position

Step 2: [1.0s] Ball moved right
  Think: (450,390),(550,590)
  Predict: (460,410),(540,570)
  Motion: Moved 300 units right, velocity 150 units/s

CRITICAL: Each step must have DIFFERENT coordinates showing actual object position in that frame.
Do NOT copy coordinates between steps!

After all steps: Answer: your final answer"""
        
        return prompt
    
    def __getitem__(self, i) -> Dict[str, Any]:
        """
        Get item in GRPO format.
        
        Returns:
            Dictionary with:
            - prompt: Messages format for TRL (list of message dicts)
            - assistant: Ground truth answer (for reference)
            - gt_evidence_steps: Ground truth evidence (for reward computation)
            - question: Original question
            - answer: Ground truth answer
            - images: List of PIL Images (for compatibility)
        """
        item = self.dataset[i]
        
        question = item.get("question", "")
        answer = item.get("answer", "")
        frames = item.get("frames", [])
        gt_evidence_steps = item.get("gt_evidence_steps", [])
        
        # Convert frames to PIL Images
        images: List[Image.Image] = []
        original_width, original_height = None, None  # Capture original dimensions
        
        if frames:
            for frame_data in frames[:self.max_frames]:
                if isinstance(frame_data, dict):
                    # Frame stored as dict with 'image' key
                    img_array = frame_data.get("image")
                elif isinstance(frame_data, np.ndarray):
                    img_array = frame_data
                elif isinstance(frame_data, list):
                    # Frame stored as nested list (HF Datasets serialization)
                    img_array = frame_data
                elif isinstance(frame_data, Image.Image):
                    images.append(frame_data)
                    continue
                else:
                    continue
                
                if img_array is not None:
                    # Convert to PIL Image
                    if isinstance(img_array, list):
                        img_array = np.array(img_array, dtype=np.uint8)
                    elif isinstance(img_array, np.ndarray):
                        img_array = img_array.astype(np.uint8)
                    
                    if len(img_array.shape) == 3:
                        # Capture original dimensions BEFORE resize (from first frame)
                        if original_width is None:
                            original_height, original_width = img_array.shape[:2]
                        
                        img = Image.fromarray(img_array)
                        # Resize to consistent size for Qwen-VL
                        img = img.resize((448, 448), Image.Resampling.LANCZOS)
                        images.append(img)
        
        # Build chain prompt with frame timing info
        num_frames = len(images)
        # Extract frame times from gt_evidence_steps or generate uniform spacing
        frame_times = []
        if gt_evidence_steps and len(gt_evidence_steps) > 0:
            # Use ground truth step times
            for i in range(num_frames):
                if i < len(gt_evidence_steps):
                    frame_times.append(gt_evidence_steps[i].get("t_s", i * 1.0))
                else:
                    frame_times.append(i * 1.0)
        else:
            # Uniform spacing (assume 8s video with uniform frames)
            video_duration = 8.0
            frame_times = [i * video_duration / max(num_frames - 1, 1) for i in range(num_frames)]
        
        user_prompt = self._build_chain_prompt(question, num_frames, frame_times)
        
        # Build messages format (TRL standard format)
        # Format: [{"role": "user", "content": [{"type": "image", ...}, {"type": "text", ...}]}]
        user_content_list = []
        
        # Add images first
        if images:
            for img in images:
                user_content_list.append({"type": "image", "image": img})
        
        # Add text prompt
        user_content_list.append({"type": "text", "text": user_prompt})
        
        # Build messages (single user message)
        messages = [{"role": "user", "content": user_content_list}]
        
        # Get video dimensions from original frames (BEFORE resize)
        img_width, img_height = 1280, 720  # Default fallback
        if original_width is not None and original_height is not None:
            img_width, img_height = original_width, original_height
        
        # Return in TRL GRPO format
        return {
            "prompt": messages,  # Messages format for TRL
            "assistant": answer,  # Ground truth answer (reference)
            "gt_evidence_steps": gt_evidence_steps,  # For reward computation
            "question": question,  # For logging
            "answer": answer,  # For reward computation
            "images": images,  # For compatibility
            "img_width": img_width,  # For coordinate conversion in rewards
            "img_height": img_height,  # For coordinate conversion in rewards
        }


def make_motion_grpo_data_module(
    dataset: HFDataset,
    processor,
    max_frames: int = 16,
    system_prompt: str = "You are an expert at analyzing video motion and spatial relationships.",
    **kwargs
) -> Dict[str, Dataset]:
    """
    Make dataset for Motion GRPO training.
    
    Args:
        dataset: HuggingFace Dataset with preprocessed PLM-STC format
        processor: AutoProcessor for the VLM model
        max_frames: Maximum number of frames to include
        system_prompt: System prompt for the model
        **kwargs: Additional arguments (ignored for compatibility)
    
    Returns:
        Dictionary with train_dataset (and optionally eval_dataset)
    """
    grpo_dataset = MotionGRPODataset(
        dataset=dataset,
        processor=processor,
        max_frames=max_frames,
        system_prompt=system_prompt,
    )
    
    return dict(train_dataset=grpo_dataset, eval_dataset=None)
