"""
Clean GRPO dataset class for Qwen-VL training.

Properly formats Dora dataset items for TRL's GRPOTrainer with vision-language support.
"""

from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from datasets import Dataset as HFDataset
import re


def truncate_transcript(transcript: str, max_length: int, question: str = "") -> str:
    """
    Truncate transcript from the beginning to keep recent context.
    
    Also removes the question from transcript if present to prevent answer leakage.
    
    Args:
        transcript: Full transcript text
        max_length: Maximum length in characters
        question: Current question (to remove from transcript)
        
    Returns:
        Truncated transcript
    """
    if not transcript:
        return ""
    
    # Remove question from transcript if present (to prevent leakage)
    if question:
        question_clean = question.strip().lower()
        # Try to find and remove question
        pattern = re.compile(re.escape(question_clean), re.IGNORECASE)
        transcript = pattern.sub("", transcript)
        # Clean up extra whitespace
        transcript = re.sub(r'\s+', ' ', transcript).strip()
    
    # Truncate from beginning if too long (keep end)
    if len(transcript) <= max_length:
        return transcript
    
    # Take last N characters
    truncated = transcript[-max_length:]
    # Add ellipsis at beginning
    if max_length > 10:
        truncated = "..." + truncated[3:]
    
    return truncated


class DoraGRPODataset(Dataset):
    """
    GRPO Dataset for Dora Q&A with Vision-Language support.
    
    Converts Dora dataset format to GRPO format expected by TRL's GRPOTrainer.
    Properly handles vision inputs (frames) and text (transcript, question).
    """
    
    def __init__(
        self,
        dataset: HFDataset,
        processor,
        max_prompt_length: int = 512,
        max_completion_length: int = 256,
        use_frames: bool = True,
        max_frames: int = 4,
        system_prompt: str = "You are a helpful visual reasoning assistant for kids.\nThink step by step, then give a final concise answer.",
    ):
        """
        Initialize Dora GRPO Dataset.
        
        Args:
            dataset: HuggingFace Dataset with Dora format (transcript, question, answer, frames)
            processor: AutoProcessor for the model
            max_prompt_length: Maximum prompt length in tokens
            max_completion_length: Maximum completion length in tokens
            use_frames: Whether to include frames/images
            max_frames: Maximum number of frames to include
            system_prompt: System prompt to prepend
        """
        super(DoraGRPODataset, self).__init__()
        self.dataset = dataset
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.use_frames = use_frames
        self.max_frames = max_frames
        self.system_prompt = system_prompt
        
        # Track which indices are accessed (for debugging)
        self._access_count = {}
        
        # Disable resize in processor (handled by model)
        if hasattr(processor, 'image_processor'):
            processor.image_processor.do_resize = False
    
    def __len__(self):
        return len(self.dataset)
    
    def get_access_stats(self):
        """Get statistics about which indices were accessed."""
        return {
            'access_count': dict(self._access_count),
            'indices_seen': sorted(self._seen_indices) if hasattr(self, '_seen_indices') else [],
            'total_accesses': sum(self._access_count.values()),
            'unique_indices': len(self._access_count),
        }
    
    def __getitem__(self, i) -> Dict[str, Any]:
        """
        Get item in GRPO format.
        
        Returns:
            Dictionary with:
            - prompt: Messages format for TRL (list of message dicts)
            - assistant: Ground truth answer string
            - images: List of PIL Images or None (for reward function access)
            - question: Question text (for reward function)
            - answer: Ground truth answer (for reward function)
        """
        item = self.dataset[i]
        
        transcript = item.get("transcript", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        # Track access count
        self._access_count[i] = self._access_count.get(i, 0) + 1
        
        # DEBUG: Log dataset item (track all indices to see iteration pattern)
        # Only log first call per index to reduce spam
        if not hasattr(self, '_seen_indices'):
            self._seen_indices = set()
        
        if i not in self._seen_indices:
            self._seen_indices.add(i)
            # print(f"\n[DEBUG DATASET] __getitem__ called for NEW index {i} (total seen: {len(self._seen_indices)}/{len(self.dataset)})")
            # print(f"  - Transcript length: {len(transcript)} chars")
            # print(f"  - Question: {question[:100]}")
            # print(f"  - Answer: {answer[:100]}")
            # print(f"  - Has frames key: {'frames' in item}")
            # if 'frames' in item:
            #     print(f"  - Frames type: {type(item['frames'])}")
            #     print(f"  - Frames length: {len(item['frames']) if item['frames'] else 0}")
        
        # Log summary every 50 accesses
        total_accesses = sum(self._access_count.values())
        if total_accesses % 50 == 0:
            print(f"\n[DEBUG DATASET] Access summary after {total_accesses} calls:")
            for idx in sorted(self._access_count.keys()):
                print(f"  - Index {idx}: {self._access_count[idx]} times")
            print(f"  - Indices seen: {sorted(self._seen_indices)} / {len(self.dataset)} total")
        
        # Convert frames to PIL Images
        images: Optional[List[Image.Image]] = None
        if self.use_frames and "frames" in item and item["frames"]:
            images = []
            for frame_data in item["frames"][:self.max_frames]:
                if isinstance(frame_data, dict):
                    img_array = frame_data.get("image")
                    if img_array is not None:
                        # Convert nested list to numpy array if needed
                        if isinstance(img_array, list):
                            img_array = np.array(img_array, dtype=np.uint8)
                        elif isinstance(img_array, np.ndarray):
                            img_array = img_array.astype(np.uint8)
                        
                        if len(img_array.shape) == 3:
                            img = Image.fromarray(img_array)
                            # Resize to consistent size (448x448 is common for Qwen-VL)
                            img = img.resize((448, 448), Image.Resampling.LANCZOS)
                            images.append(img)
                elif isinstance(frame_data, Image.Image):
                    img = frame_data.resize((448, 448), Image.Resampling.LANCZOS)
                    images.append(img)
            
            if not images:
                images = None
        
        # Truncate transcript to fit within token limit
        # We need to account for system prompt, question, and formatting
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            tokenizer = self.processor.tokenizer
            
            # Estimate space needed for non-transcript parts
            # System prompt + "Context: " + "\nQuestion: " + question + "\nAnswer:"
            non_transcript_text = f"{self.system_prompt}\n\nContext: \nQuestion: {question}\nAnswer:"
            non_transcript_tokens = len(tokenizer.encode(non_transcript_text, add_special_tokens=False))
            
            # Reserve space for transcript (with buffer)
            max_transcript_tokens = max(0, self.max_prompt_length - non_transcript_tokens - 50)  # 50 token buffer
            
            # Truncate transcript
            transcript_tokens = tokenizer.encode(transcript, add_special_tokens=False)
            if len(transcript_tokens) > max_transcript_tokens:
                # Keep last N tokens
                transcript_tokens = transcript_tokens[-max_transcript_tokens:]
                transcript = tokenizer.decode(transcript_tokens, skip_special_tokens=False)
                if max_transcript_tokens > 10:
                    transcript = "..." + transcript[3:]
        else:
            # Fallback: character-based truncation
            max_transcript_chars = 500
            transcript = truncate_transcript(transcript, max_transcript_chars, question)
        
        # Build messages format (TRL standard format)
        # Format: [{"role": "user", "content": [{"type": "image", "image": PIL.Image}, {"type": "text", "text": "..."}]}]
        user_content_list = []
        
        # Add images first
        if images:
            for img in images:
                user_content_list.append({"type": "image", "image": img})
        
        # Add text (system prompt + context + question)
        user_text = f"{self.system_prompt}\n\nContext: {transcript}\nQuestion: {question}\nAnswer:"
        user_content_list.append({"type": "text", "text": user_text})
        
        # Build messages (single user message, no system message - TRL standard)
        messages = [{"role": "user", "content": user_content_list}]
        
        # DEBUG: Log formatted data (only for first item)
        if i == 0:
            print(f"  - Number of images: {len(images) if images else 0}")
            print(f"  - User content list length: {len(user_content_list)}")
            print(f"  - Messages format: {type(messages)}")
            print(f"  - Messages[0] keys: {list(messages[0].keys()) if messages else []}")
            if messages and 'content' in messages[0]:
                print(f"  - Content items: {len(messages[0]['content'])}")
                for idx, content_item in enumerate(messages[0]['content'][:3]):
                    print(f"    - Content[{idx}] type: {type(content_item)}")
                    if isinstance(content_item, dict):
                        print(f"    - Content[{idx}] keys: {list(content_item.keys())}")
                        if 'text' in content_item:
                            print(f"    - Content[{idx}] text (first 150 chars): {content_item['text'][:150]}")
        
        # Return in TRL GRPO format
        return {
            "prompt": messages,  # Messages format for TRL
            "assistant": answer,  # Ground truth answer
            "images": images,  # PIL Images for reward function access
            "question": question,  # For reward function
            "answer": answer,  # For reward function
        }


def make_dora_grpo_data_module(
    dataset: HFDataset,
    processor,
    max_prompt_length: int = 512,
    max_completion_length: int = 256,
    use_frames: bool = True,
    max_frames: int = 4,
    system_prompt: str = "You are a helpful visual reasoning assistant for kids.\nThink step by step, then give a final concise answer.",
) -> Dict[str, Dataset]:
    """
    Make dataset for Dora GRPO training.
    
    Args:
        dataset: HuggingFace Dataset with Dora format
        processor: AutoProcessor for the model
        max_prompt_length: Maximum prompt length in tokens
        max_completion_length: Maximum completion length in tokens
        use_frames: Whether to include frames/images
        max_frames: Maximum number of frames to include
        system_prompt: System prompt to prepend
    
    Returns:
        Dictionary with train_dataset (and optionally eval_dataset)
    """
    grpo_dataset = DoraGRPODataset(
        dataset=dataset,
        processor=processor,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        use_frames=use_frames,
        max_frames=max_frames,
        system_prompt=system_prompt,
    )
    
    return dict(train_dataset=grpo_dataset, eval_dataset=None)

