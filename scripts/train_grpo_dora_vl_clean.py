#!/usr/bin/env python3
"""
Clean GRPO training for Dora Q&A with Vision-Language support.
Follows cookbook pattern exactly - no workarounds, straightforward implementation.
"""

import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_from_disk
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl.trainer import GRPOTrainer, GRPOConfig

from scripts.dora_grpo_dataset import make_dora_grpo_data_module
from src.ppo_trainer_simple import extract_final_answer, string_f1


# Minimal trainer subclass - only fixes truncation issue for images
class MinimalVLGRPOTrainer(GRPOTrainer):
    """Minimal subclass - only fixes truncation issue for images."""
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "assistant", "images", "question", "answer"]
    
    def _generate_single_turn(self, prompts: list):
        """
        Minimal override: disable truncation when images are present.
        This is the ONLY necessary fix - TRL's default truncates which breaks image tokens.
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
        
        # CRITICAL: Disable truncation when images are present
        if not has_images:
            self.max_prompt_length = 512
            processor_kwargs["max_length"] = self.max_prompt_length
            processor_kwargs["truncation"] = True
        
        # Apply chat template (standard TRL approach)
        if is_conversational({"prompt": prompts[0]}):
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
        
        # Generate using standard TRL logic
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
            
            # Get EOS and PAD tokens from processor/tokenizer if available
            eos_token_id = self.eos_token_id
            pad_token_id = self.pad_token_id
            if hasattr(self.processing_class, 'tokenizer') and self.processing_class.tokenizer is not None:
                tokenizer = self.processing_class.tokenizer
                if tokenizer.eos_token_id is not None:
                    eos_token_id = tokenizer.eos_token_id
                if tokenizer.pad_token_id is not None:
                    pad_token_id = tokenizer.pad_token_id
            
            generate_kwargs = {
                **generate_inputs,
                "max_new_tokens": self.max_completion_length,
                "min_new_tokens": 1,  # Allow at least 1 token (some models need this)
                "do_sample": True,  # CRITICAL: Enable sampling for diversity
                "temperature": 1.0,  # Cookbook standard - higher = more diverse
                "top_p": 0.9,  # Cookbook standard
                "top_k": 50,  # Additional diversity control
                "eos_token_id": eos_token_id,
                "pad_token_id": pad_token_id,
            }
            
            # Ensure model is in eval mode for generation (sampling still works)
            if hasattr(unwrapped_model, 'eval'):
                unwrapped_model.eval()
            
            # Generate with sampling
            prompt_completion_ids = unwrapped_model.generate(**generate_kwargs)
        
        # Extract completions (standard TRL approach)
        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # Debug: Check what was generated (first batch only, first step)
        if hasattr(self, '_debug_step_count'):
            self._debug_step_count += 1
        else:
            self._debug_step_count = 1
        
        if self._debug_step_count == 1 and len(completion_ids) > 0:
            # Decode first few completions to see what's being generated
            if hasattr(self.processing_class, 'tokenizer') and self.processing_class.tokenizer is not None:
                tokenizer = self.processing_class.tokenizer
                print(f"\n[DEBUG] Generated {len(completion_ids)} completions:")
                for i, comp_ids in enumerate(completion_ids[:4]):
                    comp_text = tokenizer.decode(comp_ids[:20], skip_special_tokens=False)  # First 20 tokens
                    print(f"  Completion {i}: {len(comp_ids)} tokens, first 20: {repr(comp_text[:100])}")
        
        # Mask after EOS (standard TRL approach)
        # Get EOS token ID from tokenizer if available
        eos_token_id_for_mask = self.eos_token_id
        if hasattr(self.processing_class, 'tokenizer') and self.processing_class.tokenizer is not None:
            if self.processing_class.tokenizer.eos_token_id is not None:
                eos_token_id_for_mask = self.processing_class.tokenizer.eos_token_id
        
        is_eos = completion_ids == eos_token_id_for_mask
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
        
        # Debug: Log completion lengths
        if self._debug_step_count == 1:
            completion_lengths = [len(c) for c in completion_ids]
            print(f"[DEBUG] Completion lengths: {completion_lengths}")
            if len(set(completion_lengths)) == 1:
                print(f"[DEBUG] WARNING: All completions have same length ({completion_lengths[0]} tokens)")
        
        return prompt_ids, completion_ids, None, {}


def compute_dora_reward(*args, **kwargs) -> List[float]:
    """
    Reward function for Dora Q&A.
    Cookbook pattern: simple reward function that receives completions and returns scores.
    """
    completions = kwargs.get('completions', [])
    if not completions:
        return [0.0]
    
    answer_list = kwargs.get('answer', [])
    rewards = []
    
    for i, completion in enumerate(completions):
        # Extract text from completion (handles both string and message dict formats)
        if isinstance(completion, list) and completion and isinstance(completion[0], dict):
            # Message dict format: extract content
            completion_text = ""
            for msg in completion:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    completion_text = content if isinstance(content, str) else str(content)
                    break
        elif isinstance(completion, str):
            completion_text = completion
        else:
            completion_text = str(completion) if completion else ""
        
        # Get ground truth
        gold_answer = answer_list[i] if i < len(answer_list) else ""
        
        # Extract final answer and compute F1
        final_answer = extract_final_answer(completion_text)
        reward = string_f1(final_answer, gold_answer)
        rewards.append(float(reward))
    
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Clean GRPO training for Dora VLM")
    parser.add_argument("dataset_path", type=str, help="Path to dataset")
    parser.add_argument("--output-dir", type=str, default="./outputs/grpo_dora_vl", help="Output directory")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model ID")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--use-frames", action="store_true", help="Use frames/images")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    parser.add_argument("--visual-only", action="store_true", help="Use only frames")
    parser.add_argument("--no-context", action="store_true", help="Use no context")
    parser.add_argument("--num-generations", type=int, default=4, help="Number of generations per prompt")
    parser.add_argument("--max-steps", type=int, default=100, help="Max training steps")
    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-prompt-length", type=int, default=512, help="Max prompt length")
    parser.add_argument("--max-response-length", type=int, default=256, help="Max response length")
    parser.add_argument("--dataloader-num-workers", type=int, default=8, help="Number of DataLoader workers (default: 8)")
    parser.add_argument("--generation-batch-size", type=int, default=None, help="Generation batch size (default: batch_size * num_generations)")
    parser.add_argument("--kl-beta", type=float, default=0.0, help="This controls how far the policy drifts from the reference model.")
    parser.add_argument("--reward-weights", type=float, default=1.0, help="If your reward model is sharp or overconfident, lowering reward scale often stabilizes training.")

    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    # Load dataset
    print(f"[GRPO] Loading dataset from {args.dataset_path}...")
    if Path(args.dataset_path).is_dir():
        dataset = load_from_disk(args.dataset_path)
    else:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=args.dataset_path)["train"]
    print(f"[GRPO] Dataset loaded: {len(dataset)} examples")
    
    # Load processor
    print(f"[GRPO] Loading processor from {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir, trust_remote_code=True)
    processor.image_processor.do_resize = False
    print(f"[GRPO] Processor loaded")
    
    # Load model
    print(f"[GRPO] Loading model from {args.model_id}...")
    model_kwargs = {
        "torch_dtype": compute_dtype,
        "device_map": "auto",
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
        print(f"[GRPO] Using 4-bit quantization")
    
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    print(f"[GRPO] Model loaded")
    
    # Setup LoRA if requested
    peft_config = None
    if args.use_lora:
        print(f"[GRPO] Setting up LoRA...")
        if args.use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print(f"[GRPO] LoRA setup complete")
    
    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print(f"[GRPO] Gradient checkpointing enabled")
    
    # Prepare dataset
    print(f"[GRPO] Preparing dataset...")
    dataset_module = make_dora_grpo_data_module(
        dataset=dataset,
        processor=processor,
        target_prompt_length=args.max_prompt_length,
        use_frames=args.use_frames,
        model_id=args.model_id,
        visual_only=args.visual_only,
        no_context=args.no_context,
    )
    print(f"[GRPO] Dataset prepared: {len(dataset_module['train_dataset'])} examples")
    
    # Create GRPO config (cookbook pattern)
    # OPTIMIZATION: For H200, use batch_size * num_generations to fully utilize GPU during generation
    if args.generation_batch_size is not None:
        generation_batch_size = args.generation_batch_size
    else:
        generation_batch_size = args.batch_size * args.num_generations
    if generation_batch_size % args.num_generations != 0:
        generation_batch_size = ((generation_batch_size + args.num_generations - 1) // args.num_generations) * args.num_generations
    print(f"[GRPO] Generation batch size: {generation_batch_size} (batch_size={args.batch_size}, num_generations={args.num_generations})")
    
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type='cosine',
        warmup_steps=10,
        per_device_train_batch_size=args.batch_size,
        generation_batch_size=generation_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_response_length,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,  # PERFORMANCE: Use 8+ workers for parallel data loading (was 4)
        dataloader_pin_memory=True,  # PERFORMANCE: Pin memory for faster GPU transfer
        ddp_find_unused_parameters=False,
        beta=args.kl_beta, # KL coefficient.  DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning use a value of 0.001
        reward_weights=[args.reward_weights]
    )
    
    # Create trainer with minimal override for image truncation fix
    print(f"[GRPO] Creating GRPOTrainer (minimal override for image truncation fix)...")
    trainer = MinimalVLGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset_module["train_dataset"],
        processing_class=processor,
        reward_funcs=[compute_dora_reward],
        peft_config=peft_config,
    )
    
    # Set generation config (cookbook standard)
    if hasattr(trainer, 'generation_config') and trainer.generation_config is not None:
        trainer.generation_config.max_length = None
        trainer.generation_config.num_assistant_tokens = None
        trainer.generation_config.max_new_tokens = args.max_response_length
        trainer.generation_config.do_sample = True
        trainer.generation_config.temperature = 1.0
        trainer.generation_config.top_p = 0.9
        print(f"[GRPO] Generation config: do_sample=True, temperature=1.0, top_p=0.9, max_new_tokens={args.max_response_length}")
    
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.max_length = None
        model.generation_config.num_assistant_tokens = None
        model.generation_config.max_new_tokens = args.max_response_length
        model.generation_config.do_sample = True
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 0.9
    
    # Start training
    print(f"\n[GRPO] Starting training...")
    print(f"[GRPO] Config: batch_size={args.batch_size}, num_generations={args.num_generations}")
    print(f"[GRPO] Config: max_steps={args.max_steps}, max_response_length={args.max_response_length}")

    trainer.train()
    
    print(f"\n[GRPO] Training complete!")


if __name__ == "__main__":
    main()

# python scripts/train_grpo_dora_vl_clean.py /projects/XXXX-1/dora/grpo_dataset     --output-dir ./outputs/grpo_dora_vl     --use-frames    --use-lora     --num-generations 8     --max-steps 100     --batch-size 8     --gradient-accumulation-steps 4     --learning-rate 1e-5     --max-prompt-length 512     --max-response-length 256     --save-steps 50 --dataloader-num-workers 4