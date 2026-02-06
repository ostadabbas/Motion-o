"""
Dora-specific GRPO Dataset for Vision-Language training.

Adapts Dora dataset format to work with QwenGRPOTrainer from examples structure.
Includes transcript truncation (from beginning, keep end).
"""

import copy
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from datasets import Dataset as HFDataset
import time

from src.ppo_trainer_simple import build_prompt


def truncate_transcript(transcript: str, target_length: int = 1337, question: str = "", all_questions: list = None) -> str:
    """
    Truncate transcript from the BEGINNING to keep recent context near question.
    Intelligently removes ALL questions (and their answers) from the transcript to avoid duplication/leakage.
    
    The most elegant approach: Remove ALL questions in the sequence (and any following answers) from the transcript
    BEFORE truncation, so the model only sees context, not previous Q&A pairs.
    
    Format: "...recent context near question" (removes old context from beginning, keeps end)
    
    Args:
        transcript: Full transcript text
        target_length: Target total length for transcript portion
        question: Current question text (for calculating space needed)
        all_questions: List of all questions in sequence (to remove all previous Q&A pairs)
    
    Returns:
        Truncated transcript with "..." prefix if truncated, with all questions/answers removed if found
    """
    import re
    
    # Step 1: Remove ALL questions (and their answers) from transcript to prevent answer leakage
    # This is critical when transcript contains sequential Q&A pairs
    if all_questions:
        # Find the earliest position of any question in the sequence
        earliest_question_pos = len(transcript)
        
        for q in all_questions:
            if not q:
                continue
            # Normalize question for flexible matching
            question_clean = q.strip().lower()
            question_no_punct = re.sub(r'[^\w\s]', '', question_clean)
            
            # Try multiple patterns for robustness
            patterns_to_try = [
                re.escape(q),  # Exact match
                re.escape(question_clean),  # Lowercase
                re.escape(question_no_punct),  # No punctuation
                question_clean.replace("'s", r"['\u2019]s"),  # Handle apostrophes
            ]
            
            for pattern in patterns_to_try:
                # Find all matches (case-insensitive)
                for match in re.finditer(pattern, transcript, re.IGNORECASE):
                    pos = match.start()
                    # Check if it's a good match (at sentence start or after punctuation)
                    if pos == 0 or transcript[max(0, pos-2):pos].strip() in ('', '.', '!', '?', '\n'):
                        if pos < earliest_question_pos:
                            earliest_question_pos = pos
        
        # If any question found, truncate transcript to end BEFORE the first question
        # This removes ALL questions and their answers from the transcript
        if earliest_question_pos < len(transcript):
            transcript = transcript[:earliest_question_pos].rstrip()
            # Clean up: remove trailing sentence fragments
            transcript = re.sub(r'[.!?]\s*$', '', transcript).strip()
    elif question:
        # Fallback: If all_questions not provided, just remove current question
        question_clean = question.strip().lower()
        question_no_punct = re.sub(r'[^\w\s]', '', question_clean)
        
        patterns_to_try = [
            re.escape(question),
            re.escape(question_clean),
            re.escape(question_no_punct),
            question_clean.replace("'s", r"['\u2019]s"),
        ]
        
        last_question_pos = -1
        for pattern in patterns_to_try:
            for match in re.finditer(pattern, transcript, re.IGNORECASE):
                pos = match.start()
                if pos > last_question_pos:
                    if pos == 0 or transcript[max(0, pos-2):pos].strip() in ('', '.', '!', '?', '\n'):
                        last_question_pos = pos
        
        if last_question_pos > 0:
            transcript = transcript[:last_question_pos].rstrip()
            transcript = re.sub(r'[.!?]\s*$', '', transcript).strip()
    
    # Step 2: Truncate from beginning if still too long (keep recent context)
    if len(transcript) <= target_length:
        return transcript
    
    # Take the last N characters (removes from beginning, keeps end)
    truncated = transcript[-target_length:]
    
    # Add ellipsis at the beginning to indicate truncation
    if target_length > 10:
        truncated = "..." + truncated[3:]
    
    return truncated


class DoraGRPODataset(Dataset):
    """
    GRPO Dataset for Dora Q&A with Vision-Language support.
    
    Converts Dora dataset format to GRPO format expected by QwenGRPOTrainer.
    Handles transcript truncation and frame/image processing.
    """
    
    def __init__(
        self,
        dataset: HFDataset,
        processor,
        target_prompt_length: int = 1337,
        use_frames: bool = True,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        max_frames: int = 4,
        visual_only: bool = False,
        no_context: bool = False,
    ):
        """
        Initialize Dora GRPO Dataset.
        
        Args:
            dataset: HuggingFace Dataset with Dora format (transcript, question, answer, frames)
            processor: AutoProcessor for the model
            target_prompt_length: Target prompt length for transcript truncation
            use_frames: Whether to include frames/images
            model_id: Model ID for determining image patch size
        """
        super(DoraGRPODataset, self).__init__()
        self.dataset = dataset
        self.processor = processor
        self.target_prompt_length = target_prompt_length
        self.use_frames = use_frames
        self.model_id = model_id
        self.max_frames = max_frames
        self.visual_only = visual_only
        self.no_context = no_context
        
        # Determine image patch size based on model
        if "Qwen3" in self.model_id:
            self.image_patch_size = 16
        else:
            self.image_patch_size = 14
        
        # Disable resize in processor (handled by model)
        if hasattr(processor, 'image_processor'):
            processor.image_processor.do_resize = False
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i) -> Dict[str, Any]:
        """
        Get item in GRPO format.
        
        OPTIMIZED: Uses preprocessed data from generation phase:
        - Transcript already has questions removed and is pre-truncated
        - Images are already resized to 448x448 (just convert array to PIL)
        - Minimal tokenization check (only if needed)
        
        Returns:
            Dictionary with:
            - prompt: Messages format for TRL
            - assistant: Assistant response string
            - images: List of PIL Images or None
            - question: Question (for reward function)
            - answer: Answer (for reward function)
        """
        t_start = time.time()
        item = self.dataset[i]
        j = i
        while "frames" not in item or len(item["frames"]) == 0:
            j -= 1
            item = self.dataset[j]
            print(f"Skipped 1 entry at position {j+1}, try to replace with position {j}")
            print("item content: ", item)
            assert False
        t_load = time.time() - t_start
        
        # Get preprocessed transcript (questions already removed, pre-truncated)
        transcript = item.get("transcript", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        # Convert pre-resized images (already 448x448) from arrays to PIL Images
        # This is fast since images are already the right size
        t_img_start = time.time()
        images: Optional[List[Image.Image]] = []
        if self.use_frames and "frames" in item and item["frames"]:
            images = []
            for frame_data in item["frames"][: self.max_frames]:
                if isinstance(frame_data, dict):
                    img_array = frame_data.get("image")
                    if img_array is not None:
                        # Convert list to numpy array (already 448x448 from preprocessing)
                        if isinstance(img_array, list):
                            img_array = np.array(img_array, dtype=np.uint8)
                        elif isinstance(img_array, np.ndarray):
                            img_array = img_array.astype(np.uint8)
                        
                        if len(img_array.shape) == 3:
                            # Fast conversion - no resizing needed (already 448x448)
                            img = Image.fromarray(img_array)
                            images.append(img)
                elif isinstance(frame_data, Image.Image):
                    # Already PIL Image (shouldn't happen with preprocessed data, but handle it)
                    if frame_data.size != (448, 448):
                        img = frame_data.resize((448, 448), Image.Resampling.LANCZOS)
                    else:
                        img = frame_data
                    images.append(img)
                else:
                    print(frame_data)
            if not images:
                images = [] # there is a problem
        t_img = time.time() - t_img_start
        
        system_prompt = "You are a helpful visual reasoning assistant for kids.\n Think step by step and always give a final concise answer in the first sentence."
        max_prompt_tokens = 512
        
        # Final tokenization check (transcript is already pre-truncated, but do a quick check)
        # This is much faster than before since transcript is already truncated
        t_token_start = time.time()
        transcript_for_prompt = transcript
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            tokenizer = self.processor.tokenizer
            
            # Build messages to check token count (only one iteration needed usually)
            def build_messages(curr_transcript: str):
                user_content_list = []
                if images:
                    for img in images:
                        user_content_list.append({"type": "image", "image": img})
                user_text = f"{system_prompt}\n\nContext: {curr_transcript}\nQuestion: {question}\nAnswer:"
                user_content_list.append({"type": "text", "text": user_text})
                msgs = [{"role": "user", "content": user_content_list}]
                return msgs
            
            # Quick check - transcript should already fit, but verify
            msgs = build_messages(transcript_for_prompt)
            if hasattr(self.processor, "apply_chat_template"):
                template_text = self.processor.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                tok_len = tokenizer(
                    template_text, add_special_tokens=False, return_tensors="pt"
                ).input_ids.shape[-1]
                
                # Only truncate if still too long (should be rare with pre-truncation)
                if tok_len > max_prompt_tokens:
                    # Trim from beginning (keep recent context)
                    t_tokens = tokenizer.encode(transcript_for_prompt, add_special_tokens=False)
                    excess = tok_len - max_prompt_tokens + 10
                    if len(t_tokens) > excess:
                        keep = len(t_tokens) - excess
                        t_tokens = t_tokens[-keep:]
                        transcript_for_prompt = tokenizer.decode(t_tokens, skip_special_tokens=False)
                        if keep > 10:
                            transcript_for_prompt = "..." + transcript_for_prompt[3:]
                    else:
                        transcript_for_prompt = ""
        t_token = time.time() - t_token_start
        
        # Build messages format (standard TRL format - COOKBOOK APPROACH)
        t_msg_start = time.time()
        user_content_list = []
        if images:
            for img in images:
                user_content_list.append({"type": "image", "image": img})

        if self.visual_only:
            user_text = f"{system_prompt}"
        elif self.no_context:
            user_text = f"{system_prompt}\n\nQuestion: {question}\nAnswer:"
        else:
            user_text = f"{system_prompt}\n\nContext: {transcript_for_prompt}\nQuestion: {question}\nAnswer:"
        user_content_list.append({"type": "text", "text": user_text})
        
        # NO system message - just user message (matches cookbook exactly)
        messages = [{"role": "user", "content": user_content_list}]
        t_msg = time.time() - t_msg_start
        
        # Return in standard TRL format
        data_dict = dict(
            prompt=messages,  # Messages format, not string
            assistant=answer,
            images=images,  # TRL expects "images" key
            question=question,
            answer=answer,
        )
        
        # Debug timing (only log first few items)
        if not hasattr(self, '_timing_logged'):
            self._timing_logged = set()
        if i not in self._timing_logged and len(self._timing_logged) < 5:
            total_time = time.time() - t_start
            print(f"[TIMING __getitem__ {i}] load={t_load*1000:.1f}ms, img={t_img*1000:.1f}ms, token={t_token*1000:.1f}ms, msg={t_msg*1000:.1f}ms, total={total_time*1000:.1f}ms")
            self._timing_logged.add(i)
        
        return data_dict


def make_dora_grpo_data_module(
    dataset: HFDataset,
    processor,
    target_prompt_length: int = 1337,
    use_frames: bool = True,
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    visual_only: bool = False,
    no_context: bool = False,
):
    """
    Make dataset for Dora GRPO training.
    
    Args:
        dataset: HuggingFace Dataset with Dora format
        processor: AutoProcessor for the model
        target_prompt_length: Target prompt length for transcript truncation
        use_frames: Whether to include frames/images
        model_id: Model ID
    
    Returns:
        Dictionary with train_dataset (and optionally eval_dataset)
    """
    grpo_dataset = DoraGRPODataset(
        dataset=dataset,
        processor=processor,
        target_prompt_length=target_prompt_length,
        use_frames=use_frames,
        model_id=model_id,
        visual_only=visual_only,
        no_context=no_context,
    )
    
    return dict(train_dataset=grpo_dataset, eval_dataset=None)

