#!/usr/bin/env python3
"""
GRPO training script for Qwen2-VL-2B-Instruct on Dora Q&A dataset.

Clean, straightforward implementation using TRL's GRPOTrainer with vision-language support.
"""

import os
# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print(f"[GRPO-VL] Set CUDA_VISIBLE_DEVICES=1 (before torch import)")

import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict, load_from_disk
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl.trainer import GRPOTrainer, GRPOConfig

# Import our clean implementations
from src.grpo_dataset import make_dora_grpo_data_module
from src.grpo_reward import compute_dora_reward


class VLGRPOTrainer(GRPOTrainer):
    """
    Custom GRPO Trainer that properly handles vision-language inputs.
    
    Key fix: Disables truncation when images are present to avoid image token count mismatches.
    """
    
    def _set_signature_columns_if_needed(self):
        """Set signature columns for GRPO training."""
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "assistant", "images", "question", "answer"]
            print(f"[GRPO-VL] Set signature columns: {self._signature_columns}")
    
    def training_step(self, model, inputs, num_items_in_batch):
        """
        Override training step to ensure model is in train mode and gradients are enabled.
        """
        # Ensure model is in train mode
        model.train()
        
        # DEBUG: Track which examples are being used
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1
        
        # Call parent training step - GRPOTrainer requires num_items_in_batch
        result = super().training_step(model, inputs, num_items_in_batch)
        
        # DEBUG: Log loss to verify gradients are flowing
        # During gradient accumulation, intermediate steps may return None or tensor
        # Only the final step returns a dict with loss
        if self._step_count <= 10:
            loss_info = None
            result_type = type(result).__name__
            
            if isinstance(result, dict):
                loss_val = result.get('loss', None)
                if loss_val is not None:
                    if hasattr(loss_val, 'item'):  # Tensor
                        loss_info = f"{loss_val.item():.4f}"
                    else:
                        loss_info = f"{loss_val:.4f}"
            elif hasattr(result, 'loss'):
                loss_val = result.loss
                if hasattr(loss_val, 'item'):  # Tensor
                    loss_info = f"{loss_val.item():.4f}"
                else:
                    loss_info = f"{loss_val:.4f}"
            elif result is None:
                loss_info = "None (grad accumulation)"
            elif hasattr(result, 'item'):  # Direct tensor
                loss_info = f"{result.item():.4f}"
            
            if loss_info:
                print(f"\n[DEBUG TRAINING] Step {self._step_count}, result_type={result_type}, loss={loss_info}")
            else:
                print(f"\n[DEBUG TRAINING] Step {self._step_count}, result_type={result_type}, loss=N/A")
        
        return result
    
    def _generate_single_turn(self, prompts: list):
        """
        Override to disable truncation when images are present.
        
        This is critical for VLM training - truncation breaks image token alignment.
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
            processor_kwargs["max_length"] = self.max_prompt_length
            processor_kwargs["truncation"] = True
        
        # Apply chat template
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
        
        # Generate with proper sampling for diversity
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
            # Generation config with sampling enabled
            # CRITICAL: Force sampling parameters to ensure diversity
            generate_kwargs = {
                **generate_inputs,
                "max_new_tokens": self.max_completion_length,
                "min_new_tokens": 1,
                "do_sample": True,  # Enable sampling for diversity
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": 50,  # Add top_k for more diversity
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
            }
            
            # Override any model-level generation config that might disable sampling
            if hasattr(unwrapped_model, 'generation_config') and unwrapped_model.generation_config is not None:
                unwrapped_model.generation_config.do_sample = True
                unwrapped_model.generation_config.temperature = 1.0
                unwrapped_model.generation_config.top_p = 0.9
            
            # Model should be in eval mode for generation (but sampling still works)
            if hasattr(unwrapped_model, 'eval'):
                unwrapped_model.eval()
            
            prompt_completion_ids = unwrapped_model.generate(**generate_kwargs)
        
        # DEBUG: Log generation results
        if not hasattr(self, '_debug_generation_count'):
            self._debug_generation_count = 0
        self._debug_generation_count += 1
        
        if self._debug_generation_count <= 2:
            print(f"\n[DEBUG GENERATION] Generation step {self._debug_generation_count}")
            print(f"  - prompt_completion_ids shape: {prompt_completion_ids.shape}")
            print(f"  - generate_inputs keys: {list(generate_inputs.keys())}")
            if 'input_ids' in generate_inputs:
                print(f"  - input_ids shape: {generate_inputs['input_ids'].shape}")
            if 'pixel_values' in generate_inputs:
                print(f"  - pixel_values shape: {generate_inputs['pixel_values'].shape if hasattr(generate_inputs['pixel_values'], 'shape') else 'N/A'}")
            if 'image_grid_thw' in generate_inputs:
                print(f"  - image_grid_thw: {generate_inputs['image_grid_thw']}")
        
        # Extract completions
        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # DEBUG: Log extracted completions
        if self._debug_generation_count <= 2:
            print(f"  - prompt_length: {prompt_length}")
            print(f"  - completion_ids shape: {completion_ids.shape}")
            print(f"  - Number of completions: {completion_ids.shape[0]}")
            
            # Decode first completion to see what was generated
            if hasattr(self.processing_class, 'tokenizer') and self.processing_class.tokenizer is not None:
                tokenizer = self.processing_class.tokenizer
                for comp_idx in range(min(2, completion_ids.shape[0])):
                    comp_text = tokenizer.decode(completion_ids[comp_idx], skip_special_tokens=False)
                    print(f"  - Completion {comp_idx} (decoded, first 200 chars): {repr(comp_text[:200])}")
        
        # Mask after EOS
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
        
        # DEBUG: Log final format
        if self._debug_generation_count <= 2:
            print(f"  - Returning {len(completion_ids)} completion_ids (as lists)")
            print(f"  - First completion_ids length: {len(completion_ids[0]) if completion_ids else 0}")
        
        return prompt_ids, completion_ids, None, {}


def main():
    parser = argparse.ArgumentParser(description="Train Qwen2-VL-2B-Instruct with GRPO on Dora dataset")
    parser.add_argument(
        "dataset_path",
        type=str,
        default="./outputs/dataset",
        nargs="?",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Model ID to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/grpo_vl",
        help="Output directory for checkpoints and final model"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for models"
    )
    
    # Training args
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of generations per prompt (minimum 2 for GRPO)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Log every N steps"
    )
    
    # Model args
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA fine-tuning"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    
    # GRPO specific
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
        help="Max prompt length in tokens"
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=256,
        help="Max completion length in tokens"
    )
    parser.add_argument(
        "--use-frames",
        action="store_true",
        default=True,
        help="Use frames/images for VL training"
    )
    parser.add_argument(
        "--no-frames",
        action="store_false",
        dest="use_frames",
        help="Disable frames (text-only training)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=4,
        help="Maximum number of frames to include"
    )
    
    args = parser.parse_args()
    
    # Validate num_generations
    if args.num_generations < 2:
        print("⚠️  WARNING: num_generations must be at least 2 for GRPO. Setting to 2.")
        args.num_generations = 2
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print("\n" + "=" * 80)
    print("GRPO TRAINING FOR QWEN-VL")
    print("=" * 80)
    print(f"Dataset: {args.dataset_path}")
    print(f"Model: {args.model_id}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Use frames: {args.use_frames}")
    print("=" * 80 + "\n")
    
    # Load dataset
    print(f"[1/6] Loading dataset from {args.dataset_path}...")
    dataset_path_obj = Path(args.dataset_path)
    try:
        if dataset_path_obj.is_dir():
            dataset = load_from_disk(str(args.dataset_path))
        else:
            from datasets import load_dataset
            dataset = load_dataset("json", data_files=str(args.dataset_path))["train"]
    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Handle DatasetDict
    if isinstance(dataset, DatasetDict):
        dataset = dataset.get("train", list(dataset.values())[0])
    
    print(f"✓ Dataset loaded: {len(dataset)} examples")
    
    # Load processor
    print(f"[2/6] Loading processor from {args.model_id}...")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    processor.image_processor.do_resize = False
    print("✓ Processor loaded")
    
    # Load model
    print(f"[3/6] Loading model from {args.model_id}...")
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
        print("  Using 4-bit quantization")
    
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, **model_kwargs)
    
    # Manually move to device
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda:0")
        print(f"  Model moved to cuda:0")
    
    model.config.use_cache = False
    
    # Fix generation config
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.max_length = None
        model.generation_config.num_assistant_tokens = None
        print("  Fixed generation config")
    
    # Prepare for training
    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")
    
    # Setup LoRA if requested
    peft_config = None
    if args.use_lora:
        print(f"[4/6] Setting up LoRA (rank={args.lora_rank})...")
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
        print("✓ LoRA configured")
    else:
        print("[4/6] Skipping LoRA (full fine-tuning)")
        # For full fine-tuning, ensure all parameters require gradients
        for param in model.parameters():
            param.requires_grad = True
    
    # CRITICAL: Ensure model is in train mode and parameters require gradients
    model.train()
    print("  Model set to train mode")
    
    # Verify gradients are enabled
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Prepare dataset
    print(f"[5/6] Preparing dataset for GRPO...")
    
    # FIX: Ensure dataset is properly accessible and not cached incorrectly
    # Convert to list if needed to ensure proper iteration
    if hasattr(dataset, 'to_list'):
        dataset_list = dataset.to_list()
        print(f"  Converted dataset to list: {len(dataset_list)} examples")
        # Recreate as Dataset to ensure proper indexing
        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_list(dataset_list)
    
    dataset_module = make_dora_grpo_data_module(
        dataset=dataset,
        processor=processor,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        use_frames=args.use_frames,
        max_frames=args.max_frames,
    )
    print(f"✓ Dataset prepared: {len(dataset_module['train_dataset'])} examples")
    
    # Verify dataset iteration works
    print(f"  Testing dataset access...")
    for test_idx in range(min(4, len(dataset_module['train_dataset']))):
        test_item = dataset_module['train_dataset'][test_idx]
        print(f"    Index {test_idx}: question='{test_item.get('question', '')[:50]}...'")
    
    # Create GRPO config
    # FIX: Ensure we use all examples, not just first batch
    # generation_batch_size should be large enough to cover all examples
    generation_batch_size = args.batch_size * args.num_generations
    if generation_batch_size % args.num_generations != 0:
        generation_batch_size = ((generation_batch_size + args.num_generations - 1) // args.num_generations) * args.num_generations
    
    # Don't limit to dataset size - let GRPO iterate through all examples
    print(f"  Generation batch size: {generation_batch_size}")
    print(f"  Dataset size: {len(dataset_module['train_dataset'])}")
    
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        generation_batch_size=generation_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,  # May help with dataset iteration
        dataloader_drop_last=False,  # Don't drop last batch - use all examples
    )
    
    # Create reward function
    reward_funcs = [compute_dora_reward]
    
    # Create trainer
    print(f"[6/6] Creating GRPOTrainer...")
    trainer = VLGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset_module["train_dataset"],
        processing_class=processor,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
    )
    
    # Set generation config
    if hasattr(trainer, 'generation_config') and trainer.generation_config is not None:
        trainer.generation_config.max_length = None
        trainer.generation_config.num_assistant_tokens = None
        trainer.generation_config.max_new_tokens = args.max_completion_length
        trainer.generation_config.do_sample = True
        trainer.generation_config.temperature = 1.0
        trainer.generation_config.top_p = 0.9
    
    # CRITICAL: Ensure model is in train mode after trainer creation
    # GRPO trainer might set model to eval mode, we need to override
    trainer.model.train()
    if hasattr(trainer.model, 'module'):  # For wrapped models
        trainer.model.module.train()
    
    print("✓ Trainer created")
    print("  Model explicitly set to train mode")
    
    # Print configuration
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Num generations: {args.num_generations}")
    print(f"  Max prompt length: {args.max_prompt_length} tokens")
    print(f"  Max completion length: {args.max_completion_length} tokens")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print("=" * 80 + "\n")
    
    # Train
    print("Starting training...")
    try:
        trainer.train()
        
        # Print dataset access statistics
        if hasattr(dataset_module['train_dataset'], 'get_access_stats'):
            stats = dataset_module['train_dataset'].get_access_stats()
            print(f"\n[DEBUG] Dataset Access Statistics:")
            print(f"  - Total accesses: {stats['total_accesses']}")
            print(f"  - Unique indices used: {stats['unique_indices']}/{len(dataset_module['train_dataset'])}")
            print(f"  - Indices seen: {stats['indices_seen']}")
            print(f"  - Access count per index: {stats['access_count']}")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        # Print stats even if interrupted
        if hasattr(dataset_module['train_dataset'], 'get_access_stats'):
            stats = dataset_module['train_dataset'].get_access_stats()
            print(f"\n[DEBUG] Dataset Access Statistics (before interruption):")
            print(f"  - Total accesses: {stats['total_accesses']}")
            print(f"  - Unique indices used: {stats['unique_indices']}/{len(dataset_module['train_dataset'])}")
            print(f"  - Indices seen: {stats['indices_seen']}")
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model saved to: {args.output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

