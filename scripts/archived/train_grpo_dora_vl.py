#!/usr/bin/env python3
"""
GRPO training for Dora Q&A dataset with Vision-Language support.

Adapts the examples/ structure for VL training similar to train_finetune.py,
but using GRPO (Group Relative Policy Optimization) from TRL.

Key features:
- Supports VL training with frames/images
- Truncates transcript from beginning (keeps end near question)
- Uses QwenGRPOTrainer from examples structure
- Compatible with Dora dataset format
"""

import os
# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"[GRPO-VL] Set CUDA_VISIBLE_DEVICES=0 (before torch import)")

import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset, load_from_disk
import numpy as np
from PIL import Image
import sys
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForVision2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl.trainer import GRPOTrainer, GRPOConfig

# Import utilities from existing code
from src.ppo_trainer_simple import build_prompt, extract_final_answer, string_f1

# Import Dora-specific dataset
from scripts.dora_grpo_dataset import make_dora_grpo_data_module


def truncate_transcript(transcript: str, target_length: int = 1337, question: str = "") -> str:
    """
    Truncate transcript from the BEGINNING to keep recent context near question.
    
    Format: "...recent context near question" (removes old context from beginning, keeps end)
    
    Args:
        transcript: Full transcript text
        target_length: Target total length for transcript portion
        question: Question text (for calculating space needed)
    
    Returns:
        Truncated transcript with "..." prefix if truncated
    """
    if len(transcript) <= target_length:
        return transcript
    
    # Take the last N characters (removes from beginning, keeps end)
    truncated = transcript[-target_length:]
    
    # Add ellipsis at the beginning to indicate truncation
    if target_length > 10:
        truncated = "..." + truncated[3:]
    
    return truncated


def format_dora_item_for_grpo(item: Dict[str, Any], target_prompt_length: int = 1337, use_frames: bool = True) -> Dict[str, Any]:
    """
    Convert Dora dataset item to GRPO format (LLaVA-style conversations).
    
    Args:
        item: Dora dataset item with transcript, question, answer, frames
        target_prompt_length: Target prompt length for transcript truncation
        use_frames: Whether to include frames/images
    
    Returns:
        Dictionary in GRPO format with conversations, images, etc.
    """
    transcript = item.get("transcript", "")
    question = item.get("question", "")
    answer = item.get("answer", "")
    
    # Calculate space needed for question and answer parts
    # Approximate: "Context: " + transcript + "\nQuestion: " + question + "\nAnswer:"
    fixed_parts = len("Context: ") + len("\nQuestion: ") + len(question) + len("\nAnswer:")
    max_transcript_length = max(0, target_prompt_length - fixed_parts - 100)  # 100 buffer
    
    # Truncate transcript from beginning (keep end near question)
    truncated_transcript = truncate_transcript(transcript, max_transcript_length, question)
    
    # Build user prompt (similar to format_messages in finetune.py)
    user_content = f"Context: {truncated_transcript}\nQuestion: {question}\nAnswer:"
    
    # Build conversations in LLaVA format
    conversations = [
        {
            "from": "human",
            "value": user_content
        },
        {
            "from": "gpt",
            "value": answer
        }
    ]
    
    # Handle images/frames
    images = None
    if use_frames and "frames" in item:
        images = []
        for frame_data in item["frames"]:
            if isinstance(frame_data, dict):
                img_array = frame_data.get("image")
                if img_array is not None:
                    # Convert nested list to numpy array
                    img_array = np.array(img_array, dtype=np.uint8)
                    if len(img_array.shape) == 3:
                        img = Image.fromarray(img_array)
                        images.append(img)
            elif isinstance(frame_data, Image.Image):
                images.append(frame_data)
    
    # Convert to GRPO format
    result = {
        "conversations": conversations,
        "question": question,  # Keep for reward function
        "answer": answer,      # Keep for reward function
    }
    
    if images:
        result["image"] = images  # List of PIL Images
    else:
        result["image"] = None
    
    return result


def compute_dora_reward(*args, **kwargs) -> List[float]:
    """
    Reward function for Dora Q&A with VL support.
    
    GRPO calls this with individual fields as keyword arguments.
    We need to reconstruct the inputs from the individual fields.
    
    Args (all via kwargs):
        prompts: List of prompt strings
        completions: List of completion strings (model outputs)
        question, answer: Lists of individual fields (one per input)
        completion_ids: List of token ID lists (optional, not used here)
    
    Returns:
        List of reward scores (one per completion)
    """
    # Extract completions (required)
    completions = kwargs.get('completions', None)
    if completions is None and len(args) > 2:
        completions = args[2] if isinstance(args[2], list) else None
    
    if completions is None or len(completions) == 0:
        print(f"\n[REWARD DEBUG] ERROR: No completions provided!")
        return [0.0] * 2  # Fallback
    
    num_items = len(completions)
    
    # Reconstruct inputs from individual fields
    answer_list = kwargs.get('answer', [])
    question_list = kwargs.get('question', [])
    
    print(f"\n[REWARD DEBUG] compute_dora_reward called!")
    print(f"  - Number of inputs: {num_items}")
    print(f"  - Number of completions: {len(completions)}")
    
    # Get completion_ids if available for token counting
    completion_ids_list = kwargs.get('completion_ids', [])
    
    rewards = []
    for i, completion in enumerate(completions):
        # Get ground truth answer
        gold_answer = answer_list[i] if i < len(answer_list) else ""
        question = question_list[i] if i < len(question_list) else ""
        
        # Debug: Check what format completion is (first one only)
        if i == 0:
            print(f"  [DEBUG] Completion type: {type(completion).__name__}, sample: {repr(completion)[:150]}")
        
        # Handle completion format - GRPO passes message dicts for conversational models
        # Format: [{'role': 'assistant', 'content': 'text'}] or plain string
        completion_text = ""
        
        if isinstance(completion, list):
            # If it's a list, check what's inside
            if not completion:
                completion_text = ""
            elif isinstance(completion[0], dict):
                # Message dict format: [{'role': 'assistant', 'content': 'text'}]
                # Extract content from assistant message
                for msg in completion:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            completion_text = content
                            break
                        elif isinstance(content, list):
                            # Content might be a list (multimodal), extract text parts
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            completion_text = " ".join(text_parts)
                            break
                if not completion_text:
                    # Fallback: try to extract any content
                    completion_text = str(completion)
            elif isinstance(completion[0], int):
                # Token IDs - GRPO shouldn't pass these, but handle gracefully
                print(f"  WARNING: Completion {i} appears to be token IDs, not decoded text!")
                completion_text = ""  # Can't decode without tokenizer
            else:
                # List of strings/chars - join them (no spaces, might be character list)
                completion_text = "".join(str(c) for c in completion)
        elif isinstance(completion, str):
            # String format (expected for non-conversational)
            completion_text = completion
        elif isinstance(completion, dict):
            # Single dict - might be message format
            if completion.get("role") == "assistant":
                content = completion.get("content", "")
                if isinstance(content, str):
                    completion_text = content
                else:
                    completion_text = str(content)
            else:
                completion_text = str(completion)
        else:
            # Other type - convert to string
            completion_text = str(completion) if completion else ""
        
        # Get token count if available
        token_count = len(completion_ids_list[i]) if i < len(completion_ids_list) and completion_ids_list[i] else None
        char_count = len(completion_text) if completion_text else 0
        
        # Extract final answer from completion
        final_answer = extract_final_answer(completion_text)
        
        # Compute F1 score as reward
        reward = string_f1(final_answer, gold_answer)
        rewards.append(float(reward))
        
        # Detailed logging for all completions (not just first 2)
        print(f"\n  --- Completion {i} ---")
        print(f"  Question: {question}")
        print(f"  Gold answer: {gold_answer}")
        print(f"  Full completion ({char_count} chars, {token_count} tokens):")
        print(f"    {repr(completion_text[:200])}")  # Show processed text, not raw completion
        print(f"  Extracted final answer: {repr(final_answer)}")
        print(f"  Reward: {reward:.3f}")
    
    print(f"\n[REWARD DEBUG] Returning {len(rewards)} rewards: {rewards}")
    return rewards


def prepare_grpo_vl_dataset(dataset: Dataset, target_prompt_length: int = 1337, use_frames: bool = True) -> List[Dict]:
    """
    Prepare Dora dataset for GRPO VL training.
    
    Converts Dora format to GRPO format with transcript truncation.
    """
    formatted_data = []
    for i in range(len(dataset)):
        item = dataset[i]
        grpo_item = format_dora_item_for_grpo(item, target_prompt_length, use_frames)
        formatted_data.append(grpo_item)
    
    return formatted_data


def main():
    parser = argparse.ArgumentParser(description="Train Dora Q&A model with GRPO (VL support)")
    parser.add_argument("dataset_path", type=str, help="Path to dataset directory")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                       help="Model ID to use")
    parser.add_argument("--output-dir", type=str, default="./outputs/grpo_dora_vl",
                       help="Output directory")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Cache directory for models")
    
    # Training args
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--num-generations", type=int, default=4,
                       help="Number of generations per prompt (for GRPO). Minimum is 2.")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Max training steps")
    parser.add_argument("--save-steps", type=int, default=50,
                       help="Save checkpoint every N steps")
    
    # Model args
    parser.add_argument("--use-4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--use-lora", action="store_true",
                       help="Use LoRA fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=16,
                       help="LoRA rank")
    
    # GRPO specific
    parser.add_argument("--max-prompt-length", type=int, default=512,
                       help="Max prompt length in tokens (standard: 512, for transcript truncation)")
    parser.add_argument("--max-response-length", type=int, default=256,
                       help="Max response length (completion tokens, standard: 256)")
    parser.add_argument("--use-frames", action="store_true", default=True,
                       help="Use frames/images for VL training")
    parser.add_argument("--no-frames", action="store_false", dest="use_frames",
                       help="Disable frames (text-only training)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"[GRPO-VL] Loading dataset from {args.dataset_path}...")
    if Path(args.dataset_path).is_dir():
        dataset = load_from_disk(args.dataset_path)
    else:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=args.dataset_path)["train"]
    
    print(f"[GRPO-VL] Dataset loaded: {len(dataset)} examples")
    
    # Load processor (for VL models)
    print(f"[GRPO-VL] Loading processor from {args.model_id}...")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    processor.image_processor.do_resize = False
    print(f"[GRPO-VL] Processor loaded")
    
    # Load model
    print(f"[GRPO-VL] Loading model from {args.model_id}...")
    model_kwargs = {
        "torch_dtype": compute_dtype,
        "device_map": None,  # Don't use auto - manually move after loading
        "cache_dir": args.cache_dir,
        "trust_remote_code": True,
    }
    
    if args.use_4bit and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quantization_config
        print("[GRPO-VL] Using 4-bit quantization")
    
    # For VL models, we need AutoModelForVision2Seq
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, **model_kwargs)
    
    # Manually move to device
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda:0")
        print(f"[GRPO-VL] Model moved to cuda:0")
    
    model.config.use_cache = False
    
    # Fix model's generation_config: ensure max_length=None and num_assistant_tokens=None
    # The model's default generation_config has max_length=20 and num_assistant_tokens=20
    # which can override our settings when calling model.generate()
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.max_length = None
        model.generation_config.num_assistant_tokens = None
        print(f"[GRPO-VL] Fixed model's generation_config: set max_length=None and num_assistant_tokens=None")
    
    # Prepare for training
    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    
    # Enable gradient checkpointing for memory efficiency (even without 4-bit)
    # This is important for VLM training as it uses significant memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[GRPO-VL] Gradient checkpointing enabled")
    
    # Setup LoRA if requested
    peft_config = None
    if args.use_lora:
        print(f"[GRPO-VL] Setting up LoRA (rank={args.lora_rank})...")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Prepare dataset using Dora-specific dataset class
    print("[GRPO-VL] Preparing dataset for GRPO VL...")
    dataset_module = make_dora_grpo_data_module(
        dataset=dataset,
        processor=processor,
        target_prompt_length=args.max_prompt_length,
        use_frames=args.use_frames,
        model_id=args.model_id,
    )
    print(f"[GRPO-VL] Prepared dataset: {len(dataset_module['train_dataset'])} examples")
    
    # Create GRPO config
    generation_batch_size = min(args.batch_size * args.num_generations, len(dataset))
    if generation_batch_size % args.num_generations != 0:
        generation_batch_size = ((generation_batch_size + args.num_generations - 1) // args.num_generations) * args.num_generations
        generation_batch_size = min(generation_batch_size, len(dataset))
    
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        generation_batch_size=generation_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_response_length,  # Use command-line argument
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,  # Important for multi-GPU stability
    )
    
    # Create reward function
    reward_funcs = [compute_dora_reward]
    
    # Use standard GRPOTrainer with minimal customization for signature columns and image handling
    # TRL's GRPOTrainer handles VLM inputs automatically when dataset returns messages format
    class SimpleVLGRPOTrainer(GRPOTrainer):
        """Minimal subclass to set signature columns and disable truncation for VLM training."""
        def _set_signature_columns_if_needed(self):
            if self._signature_columns is None:
                self._signature_columns = ["prompt", "assistant", "images", "question", "answer"]
                print(f"[GRPO-VL] Set signature columns: {self._signature_columns}")
        
        def _generate_single_turn(self, prompts: list):
            """
            Override to disable truncation when images are present.
            This is the ONLY change needed - TRL's default truncates which breaks image tokens.
            """
            from trl.data_utils import is_conversational
            
            # Check if images are present in messages
            has_images = False
            if prompts and isinstance(prompts[0], list):
                for msg in prompts[0]:
                    if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                        for item in msg["content"]:
                            if isinstance(item, dict) and item.get("type") == "image":
                                has_images = True
                                break
                        if has_images:
                            break
            
            device = self.accelerator.device
            processor_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "padding_side": "left",
                "add_special_tokens": False,
            }
            
            # CRITICAL FIX: Disable truncation when images are present
            # This is the root cause of the "image token count mismatch" error
            if not has_images:
                processor_kwargs["max_length"] = self.max_prompt_length
                processor_kwargs["truncation"] = True
            
            # Apply chat template - standard approach (no system messages, so no bug!)
            # Cookbook approach: system prompt is in user text, not separate system message
            if is_conversational({"prompt": prompts[0]}):
                # Standard processing - works because we don't have system messages
                generate_inputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    **processor_kwargs,  # No truncation if has_images=True
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                generate_inputs = self.processing_class(text=prompts, **processor_kwargs)
            
            generate_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in generate_inputs.items()}
            
            # Generate - use standard generation
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from contextlib import nullcontext
            from trl.models import unwrap_model_for_generation
            from trl.extras.profiling import profiling_context
            
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                gen_config = self.generation_config if hasattr(self, 'generation_config') and self.generation_config else None
                
                # CRITICAL: Ensure sampling is enabled for diverse completions
                # Cookbook standard: do_sample=True, temperature=1.0, top_p=0.9
                # CRITICAL: Set generation kwargs with proper sampling for diversity
                # Cookbook standard: do_sample=True, temperature=1.0, top_p=0.9
                generate_kwargs = {
                    **generate_inputs,
                    "max_new_tokens": self.max_completion_length,  # Ensure proper length
                    "min_new_tokens": 1,  # Allow at least 1 token
                    "do_sample": True,  # Force sampling for diversity (CRITICAL)
                    "temperature": 1.0,  # Cookbook standard
                    "top_p": 0.9,  # Cookbook standard for better diversity
                    "eos_token_id": gen_config.eos_token_id if gen_config else self.eos_token_id,
                    "pad_token_id": gen_config.pad_token_id if gen_config else self.pad_token_id,
                }
                
                # Ensure model is in eval mode for generation (but sampling still works)
                # Don't use deterministic mode - we want diverse outputs
                if hasattr(unwrapped_model, 'eval'):
                    unwrapped_model.eval()
                
                prompt_completion_ids = unwrapped_model.generate(**generate_kwargs)
            
            # Extract completions (same as parent)
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            
            # Mask after EOS (same as parent)
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
            
            return prompt_ids, completion_ids, None, {}
    
    # Create trainer using standard GRPOTrainer
    print("[GRPO-VL] Creating GRPOTrainer (standard TRL implementation)...")
    trainer = SimpleVLGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset_module["train_dataset"],
        processing_class=processor,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
    )
    
    # Ensure generation config is set correctly for diverse sampling and proper length
    # Cookbook standard: do_sample=True, temperature=1.0, top_p=0.9
    from transformers import GenerationConfig
    if hasattr(trainer, 'generation_config') and trainer.generation_config is not None:
        trainer.generation_config.max_length = None
        trainer.generation_config.num_assistant_tokens = None
        trainer.generation_config.max_new_tokens = args.max_response_length  # Explicitly set from max_completion_length
        trainer.generation_config.do_sample = True
        trainer.generation_config.temperature = 1.0
        trainer.generation_config.top_p = 0.9  # Cookbook standard for diversity
        print(f"\n[GRPO-VL] Set generation_config: max_length=None, num_assistant_tokens=None, max_new_tokens={args.max_response_length}, do_sample=True, temperature=1.0, top_p=0.9")
    
    # Also fix model's generation config
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.max_length = None
        model.generation_config.num_assistant_tokens = None
        model.generation_config.max_new_tokens = args.max_response_length  # Explicitly set
        model.generation_config.do_sample = True
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 0.9  # Cookbook standard
    
    # Log generation config after trainer initialization
    print("\n[GRPO-VL] Generation Configuration:")
    print(f"  - max_completion_length: {grpo_config.max_completion_length}")
    print(f"  - max_prompt_length: {grpo_config.max_prompt_length}")
    if hasattr(trainer, 'generation_config') and trainer.generation_config is not None:
        print(f"  - generation_config.max_new_tokens: {getattr(trainer.generation_config, 'max_new_tokens', 'N/A')}")
        print(f"  - generation_config.max_length: {getattr(trainer.generation_config, 'max_length', 'N/A')}")
        print(f"  - generation_config.num_assistant_tokens: {getattr(trainer.generation_config, 'num_assistant_tokens', 'N/A')}")
        print(f"  - generation_config.do_sample: {getattr(trainer.generation_config, 'do_sample', 'N/A')}")
        print(f"  - generation_config.temperature: {getattr(trainer.generation_config, 'temperature', 'N/A')}")
        print(f"  - generation_config.eos_token_id: {getattr(trainer.generation_config, 'eos_token_id', 'N/A')}")
    
    # Train
    print("\n[GRPO-VL] Starting training...")
    print(f"[GRPO-VL] Using frames: {args.use_frames}")
    print(f"[GRPO-VL] Num generations: {args.num_generations}")
    
    trainer.train()
    
    # Save
    print(f"[GRPO-VL] Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    print(f"[GRPO-VL] Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

