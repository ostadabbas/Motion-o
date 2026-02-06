#!/usr/bin/env python3
"""
GRPO training for Motion Reasoning with Spatio-Temporal Evidence Chains.

Trains a VLM to generate verifiable evidence chains with bounding boxes and
motion descriptors using geometric rewards (spatial, temporal, motion, caption).
"""

import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"  # Use V100s, skip GTX 745

import argparse
import torch
from pathlib import Path
from typing import List
from datasets import load_from_disk
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl.trainer import GRPOTrainer, GRPOConfig

from src.motion_dataset import make_motion_grpo_data_module
from src.geometric_reward import compute_geometric_reward


# Minimal trainer subclass - only fixes truncation issue for images
class MinimalVLGRPOTrainer(GRPOTrainer):
    """Minimal subclass - only fixes truncation issue for images."""
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Add gt_evidence_steps for motion reward
            self._signature_columns = ["prompt", "assistant", "images", "question", "answer", "gt_evidence_steps"]
    
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
                "min_new_tokens": 1,
                "do_sample": True,  # CRITICAL: Enable sampling for diversity
                "temperature": 1.0,  # Cookbook standard - higher = more diverse
                "top_p": 0.9,
                "top_k": 50,
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
                for i, comp_ids in enumerate(completion_ids[:2]):
                    comp_text = tokenizer.decode(comp_ids[:50], skip_special_tokens=False)
                    print(f"  Completion {i}: {len(comp_ids)} tokens, first 50: {repr(comp_text[:150])}")
        
        # Mask after EOS (standard TRL approach)
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
        
        return prompt_ids, completion_ids, None, {}


def main():
    parser = argparse.ArgumentParser(description="GRPO training for Motion Reasoning")
    parser.add_argument("dataset_path", type=str, help="Path to preprocessed PLM-STC dataset")
    parser.add_argument("--output-dir", type=str, default="./outputs/motion_grpo", help="Output directory")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Model ID")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    parser.add_argument("--num-generations", type=int, default=4, help="Number of generations per prompt")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-prompt-length", type=int, default=512, help="Max prompt length")
    parser.add_argument("--max-response-length", type=int, default=512, help="Max response length (evidence chains are longer)")
    parser.add_argument("--max-frames", type=int, default=16, help="Maximum frames per video")
    parser.add_argument("--dataloader-num-workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--generation-batch-size", type=int, default=None, help="Generation batch size")
    parser.add_argument("--kl-beta", type=float, default=0.01, help="KL divergence coefficient")
    parser.add_argument("--reward-weights", type=float, default=1.0, help="Reward scaling factor")
    
    # Motion-specific reward weights
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second for motion computation")
    parser.add_argument("--lambda-spatial", type=float, default=0.25, help="Weight for spatial reward (bbox IoU)")
    parser.add_argument("--lambda-temporal", type=float, default=0.15, help="Weight for temporal reward (interval IoU)")
    parser.add_argument("--lambda-motion", type=float, default=0.35, help="Weight for motion reward (trajectory)")
    parser.add_argument("--lambda-caption", type=float, default=0.20, help="Weight for caption reward (text similarity)")
    parser.add_argument("--lambda-format", type=float, default=0.05, help="Weight for format reward (parseability)")
    parser.add_argument("--debug-reward", action="store_true", help="Enable debug logging in reward function")

    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    # Load dataset
    print(f"[MOTION_GRPO] Loading dataset from {args.dataset_path}...")
    if Path(args.dataset_path).is_dir():
        dataset = load_from_disk(args.dataset_path)
    else:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=args.dataset_path)["train"]
    print(f"[MOTION_GRPO] Dataset loaded: {len(dataset)} examples")
    
    # Load processor
    print(f"[MOTION_GRPO] Loading processor from {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir, trust_remote_code=True)
    processor.image_processor.do_resize = False
    print(f"[MOTION_GRPO] Processor loaded")
    
    # Load model
    print(f"[MOTION_GRPO] Loading model from {args.model_id}...")
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
        print(f"[MOTION_GRPO] Using 4-bit quantization")
    
    # Select correct model class based on model ID
    mid = args.model_id.lower()
    if "qwen3" in mid:
        model_cls = Qwen3VLForConditionalGeneration
    elif "qwen2.5" in mid or "qwen2_5" in mid:
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        model_cls = Qwen2VLForConditionalGeneration
    
    model = model_cls.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    print(f"[MOTION_GRPO] Model loaded: {model_cls.__name__}")
    
    # Setup LoRA if requested
    peft_config = None
    if args.use_lora:
        print(f"[MOTION_GRPO] Setting up LoRA...")
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
        print(f"[MOTION_GRPO] LoRA setup complete")
    
    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print(f"[MOTION_GRPO] Gradient checkpointing enabled")
    
    # Prepare dataset
    print(f"[MOTION_GRPO] Preparing motion dataset...")
    dataset_module = make_motion_grpo_data_module(
        dataset=dataset,
        processor=processor,
        max_frames=args.max_frames,
    )
    print(f"[MOTION_GRPO] Dataset prepared: {len(dataset_module['train_dataset'])} examples")
    
    # Create GRPO config
    if args.generation_batch_size is not None:
        generation_batch_size = args.generation_batch_size
    else:
        generation_batch_size = args.batch_size * args.num_generations
    if generation_batch_size % args.num_generations != 0:
        generation_batch_size = ((generation_batch_size + args.num_generations - 1) // args.num_generations) * args.num_generations
    print(f"[MOTION_GRPO] Generation batch size: {generation_batch_size} (batch_size={args.batch_size}, num_generations={args.num_generations})")
    
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
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        beta=args.kl_beta,
        reward_weights=[args.reward_weights]
    )
    
    # Create reward function with motion-specific parameters
    def motion_reward_wrapper(*reward_args, **kwargs):
        # Inject motion-specific parameters from outer scope args
        kwargs['fps'] = args.fps
        kwargs['lambda_s'] = args.lambda_spatial
        kwargs['lambda_t'] = args.lambda_temporal
        kwargs['lambda_m'] = args.lambda_motion
        kwargs['lambda_c'] = args.lambda_caption
        kwargs['lambda_f'] = args.lambda_format
        kwargs['debug'] = args.debug_reward
        return compute_geometric_reward(*reward_args, **kwargs)
    
    # Create trainer with minimal override for image truncation fix
    print(f"[MOTION_GRPO] Creating GRPOTrainer...")
    print(f"[MOTION_GRPO] Reward weights: spatial={args.lambda_spatial}, temporal={args.lambda_temporal}, motion={args.lambda_motion}, caption={args.lambda_caption}")
    trainer = MinimalVLGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset_module["train_dataset"],
        processing_class=processor,
        reward_funcs=[motion_reward_wrapper],
        peft_config=peft_config,
    )
    
    # Set generation config
    if hasattr(trainer, 'generation_config') and trainer.generation_config is not None:
        trainer.generation_config.max_length = None
        trainer.generation_config.num_assistant_tokens = None
        trainer.generation_config.max_new_tokens = args.max_response_length
        trainer.generation_config.do_sample = True
        trainer.generation_config.temperature = 1.0
        trainer.generation_config.top_p = 0.9
        print(f"[MOTION_GRPO] Generation config: do_sample=True, temperature=1.0, top_p=0.9, max_new_tokens={args.max_response_length}")
    
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.max_length = None
        model.generation_config.num_assistant_tokens = None
        model.generation_config.max_new_tokens = args.max_response_length
        model.generation_config.do_sample = True
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 0.9
    
    # Start training
    print(f"\n[MOTION_GRPO] Starting training...")
    print(f"[MOTION_GRPO] Config: batch_size={args.batch_size}, num_generations={args.num_generations}")
    print(f"[MOTION_GRPO] Config: max_steps={args.max_steps}, max_response_length={args.max_response_length}")

    trainer.train()
    
    print(f"\n[MOTION_GRPO] Training complete!")


if __name__ == "__main__":
    main()
