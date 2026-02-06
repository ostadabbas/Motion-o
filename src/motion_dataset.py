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
    
    def _build_chain_prompt(self, question: str) -> str:
        """
        Build prompt requesting spatio-temporal evidence chain.
        
        Enhanced prompt following PyImageSearch best practices:
        1. Task-Specific Instruction (Analyze, Detect)
        2. Object Specification with attributes
        3. Output Requirements (JSON format with bbox_2d)
        4. Structured reasoning format
        
        Args:
            question: Motion-related question
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""{self.system_prompt}

**Task**: Analyze the video motion to answer this question: {question}

**Instructions**: Detect all relevant objects at each key moment and track their motion across frames.

For each evidence step, provide:
1. **Time Interval**: [start_time–end_time] in seconds (e.g., [2.1–3.4])
2. **Object Detection**: Detect objects and return bounding boxes in JSON format:
   ```json
   {{"bbox_2d": [x1, y1, x2, y2], "label": "object_name"}}
   ```
   Where coordinates are normalized between 0 and 1.
3. **Motion Description**: Describe how the object moved (direction, speed, displacement)
4. **What Happened**: Explain the event

**Output Format**:
Step 1: [t_s–t_e]
Objects detected:
```json
[{{"bbox_2d": [x1, y1, x2, y2], "label": "object1"}}, {{"bbox_2d": [x1, y1, x2, y2], "label": "object2"}}]
```
Motion: [centroid displacement, velocity, direction]
Description: [what happened]

Step 2: [t_s–t_e]
...

**Final Answer**: [your concise answer based on the motion evidence]"""
        
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
        if frames:
            for frame_data in frames[:self.max_frames]:
                if isinstance(frame_data, dict):
                    # Frame stored as dict with 'image' key
                    img_array = frame_data.get("image")
                elif isinstance(frame_data, np.ndarray):
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
                        img = Image.fromarray(img_array)
                        # Resize to consistent size for Qwen-VL
                        img = img.resize((448, 448), Image.Resampling.LANCZOS)
                        images.append(img)
        
        # Build chain prompt
        user_prompt = self._build_chain_prompt(question)
        
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
        
        # Return in TRL GRPO format
        return {
            "prompt": messages,  # Messages format for TRL
            "assistant": answer,  # Ground truth answer (reference)
            "gt_evidence_steps": gt_evidence_steps,  # For reward computation
            "question": question,  # For logging
            "answer": answer,  # For reward computation
            "images": images,  # For compatibility
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
