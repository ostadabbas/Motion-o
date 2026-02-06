#!/usr/bin/env python3
"""
Simple GRPO training for Dora Q&A dataset (text-only).

Adapted from examples/train_grpo.py but simplified for text-only Q&A.
Uses TRL's GRPOTrainer which is cleaner than patching PPO.
"""

import os
# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
# This must be done before any CUDA libraries are loaded
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"[GRPO] Set CUDA_VISIBLE_DEVICES=0 (before torch import)")

import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset, load_from_disk

from transformers import (
    AutoTokenizer,
    AutoProcessor,  # Add processor for Qwen models
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl.trainer import GRPOTrainer, GRPOConfig

# Import utilities from existing code
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ppo_trainer_simple import build_prompt, extract_final_answer, string_f1


def compute_dora_reward(*args, **kwargs) -> List[float]:
    """
    Reward function for Dora Q&A.
    
    GRPO calls this with individual fields as keyword arguments, not a single 'inputs' dict.
    We need to reconstruct the inputs from the individual fields.
    
    Args (all via kwargs):
        prompts: List of prompt strings
        completions: List of completion strings (model outputs)
        question, answer, transcript: Lists of individual fields (one per input)
        completion_ids: List of token ID lists (optional, not used here)
    
    Returns:
        List of reward scores (one per completion)
    """
    # GRPO passes individual fields, not a single 'inputs' dict
    # Extract completions (required)
    completions = kwargs.get('completions', None)
    if completions is None and len(args) > 2:
        completions = args[2] if isinstance(args[2], list) else None
    
    if completions is None or len(completions) == 0:
        print(f"\n[REWARD DEBUG] ERROR: No completions provided!")
        return [0.0] * 2  # Fallback
    
    num_items = len(completions)
    
    # Reconstruct inputs from individual fields
    # GRPO passes: question, answer, transcript, etc. as separate lists
    answer_list = kwargs.get('answer', [])
    question_list = kwargs.get('question', [])
    transcript_list = kwargs.get('transcript', [])
    
    # Build inputs list from individual fields
    inputs = []
    for i in range(num_items):
        inp = {
            "question": question_list[i] if i < len(question_list) else "",
            "answer": answer_list[i] if i < len(answer_list) else "",
            "transcript": transcript_list[i] if i < len(transcript_list) else "",
        }
        # Add any other fields that might be useful
        for key in ['video_path', 'segment_id', 'confidence']:
            if key in kwargs and i < len(kwargs[key]):
                inp[key] = kwargs[key][i]
        inputs.append(inp)
    
    print(f"\n[REWARD DEBUG] compute_dora_reward called!")
    print(f"  - Number of inputs: {len(inputs)}")
    print(f"  - Number of completions: {len(completions)}")
    
    rewards = []
    for i, (inp, completion) in enumerate(zip(inputs, completions)):
        # Get ground truth answer from input
        gold_answer = inp.get("answer", "") or ""
        
        # Extract final answer from completion
        final_answer = extract_final_answer(completion)
        
        # Compute F1 score as reward
        reward = string_f1(final_answer, gold_answer)
        rewards.append(float(reward))
        
        if i < 2:  # Debug first 2
            print(f"  - Completion {i}: gold='{gold_answer}', pred='{final_answer[:50]}...', reward={reward:.3f}")
    
    print(f"[REWARD DEBUG] Returning {len(rewards)} rewards: {rewards}")
    return rewards


def prepare_grpo_dataset(dataset: Dataset, tokenizer, max_length: int = 1024) -> Dataset:
    """
    Prepare Dora dataset for GRPO training.
    
    Each item needs:
    - "prompt": The input prompt (will be tokenized by GRPO)
    - All original fields (question, answer, transcript) for reward function
    """
    def format_item(item):
        # Build prompt using existing function
        prompt = build_prompt(item)
        
        # Return item with prompt field + all original fields
        return {
            "prompt": prompt,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "transcript": item.get("transcript", ""),
            # Keep any other fields
            **{k: v for k, v in item.items() if k not in ["prompt", "question", "answer", "transcript"]}
        }
    
    # Apply formatting
    formatted_data = [format_item(dataset[i]) for i in range(len(dataset))]
    return Dataset.from_list(formatted_data)


def main():
    parser = argparse.ArgumentParser(description="Train Dora Q&A model with GRPO")
    parser.add_argument("dataset_path", type=str, help="Path to dataset directory")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Model ID to use")
    parser.add_argument("--output-dir", type=str, default="./outputs/grpo_dora",
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
    parser.add_argument("--num-generations", type=int, default=2,
                       help="Number of generations per prompt (for GRPO). Minimum is 2 (required by GRPO).")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Max training steps")
    parser.add_argument("--save-steps", type=int, default=50,
                       help="Save checkpoint every N steps")
    
    # Model args
    parser.add_argument("--use-4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Explicitly disable 4-bit quantization (use if you see CUDA errors)")
    parser.add_argument("--use-lora", action="store_true",
                       help="Use LoRA fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=16,
                       help="LoRA rank")
    
    # GRPO specific
    parser.add_argument("--max-prompt-length", type=int, default=1024,
                       help="Max prompt length")
    parser.add_argument("--max-response-length", type=int, default=128,
                       help="Max response length")
    
    args = parser.parse_args()
    
    # CUDA_VISIBLE_DEVICES was set at the top of the file before torch import
    # Verify it's set correctly
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"[GRPO] CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Verify GPU setup
    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        print(f"[GRPO] Number of visible GPUs: {num_gpus}")
        if num_gpus > 1:
            print(f"[GRPO] WARNING: {num_gpus} GPUs visible - this may cause DataParallel issues")
            print(f"[GRPO] If you see DataParallel warnings, set CUDA_VISIBLE_DEVICES=0 before running")
    
    print(f"[GRPO] Loading dataset from {args.dataset_path}...")
    if Path(args.dataset_path).is_dir():
        dataset = load_from_disk(args.dataset_path)
    else:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=args.dataset_path)["train"]
    
    print(f"[GRPO] Dataset loaded: {len(dataset)} examples")
    
    # Load tokenizer
    # For text-only models, use AutoTokenizer directly
    # GRPOTrainer expects a tokenizer (not processor) for text-only models
    print(f"[GRPO] Loading tokenizer from {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"[GRPO] Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    # Prepare dataset
    print("[GRPO] Preparing dataset for GRPO...")
    train_dataset = prepare_grpo_dataset(dataset, tokenizer, max_length=args.max_prompt_length)
    print(f"[GRPO] Prepared dataset: {len(train_dataset)} examples")
    
    # Load model
    print(f"[GRPO] Loading model from {args.model_id}...")
    # Use same approach as PPO: device_map=None, then manually move
    # This avoids issues with device_map="auto" and quantization
    model_kwargs = {
        "torch_dtype": compute_dtype,  # Use torch_dtype (same as PPO)
        "device_map": None,  # Don't use auto - manually move after loading
        "cache_dir": args.cache_dir,
        "trust_remote_code": True,
    }
    
    # Add quantization if requested
    # Note: bitsandbytes can have CUDA compatibility issues
    # If you see "Error named symbol not found" errors, use --no-4bit flag
    if args.no_4bit:
        args.use_4bit = False
        print("[GRPO] 4-bit quantization explicitly disabled")
    
    if args.use_4bit and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quantization_config
        print("[GRPO] Using 4-bit quantization")
        print("[GRPO] Note: If you see 'Error named symbol not found' CUDA errors,")
        print("[GRPO]       run with --no-4bit flag to disable quantization")
    
    print("[GRPO] Loading model...")
    model_loaded = False
    if args.use_4bit:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
            model_loaded = True
            print("[GRPO] Model loaded successfully with 4-bit quantization")
        except Exception as e:
            print(f"[GRPO] Error loading with 4-bit quantization: {e}")
            print("[GRPO] Retrying without quantization...")
            # Remove quantization and retry
            model_kwargs.pop("quantization_config", None)
            args.use_4bit = False
    
    if not model_loaded:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
        print("[GRPO] Model loaded successfully (full precision)")
    
    # Manually move to device (same as PPO approach)
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda:0")
        print(f"[GRPO] Model moved to cuda:0")
        # Ensure model is not wrapped in DataParallel (can cause hangs)
        if isinstance(model, torch.nn.DataParallel):
            print("[GRPO] WARNING: Model is wrapped in DataParallel, unwrapping...")
            model = model.module
    
    model.config.use_cache = False
    print(f"[GRPO] Model config: use_cache={model.config.use_cache}")
    
    # Prepare for training
    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    
    # Setup LoRA if requested
    peft_config = None
    if args.use_lora:
        print(f"[GRPO] Setting up LoRA (rank={args.lora_rank})...")
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
    
    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[GRPO] Gradient checkpointing enabled")
    
    # Create GRPO config
    # GRPO requires: generation_batch_size must be divisible by num_generations
    # For small datasets, don't exceed dataset size
    generation_batch_size = min(args.batch_size, len(train_dataset))
    if generation_batch_size % args.num_generations != 0:
        # Round up to nearest multiple of num_generations
        generation_batch_size = ((generation_batch_size + args.num_generations - 1) // args.num_generations) * args.num_generations
        generation_batch_size = min(generation_batch_size, len(train_dataset))  # Don't exceed dataset size
        print(f"[GRPO] Adjusting generation_batch_size to {generation_batch_size} (must be divisible by num_generations={args.num_generations}, max={len(train_dataset)})")
    
    # Create GRPO config
    # Note: Can't set both generation_batch_size and steps_per_generation
    # We set generation_batch_size explicitly to handle the divisibility constraint
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        generation_batch_size=generation_batch_size,  # Set explicitly (must be divisible by num_generations)
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=1,  # Log every step to see loss immediately
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_response_length,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        remove_unused_columns=False,
        # Explicitly disable multi-GPU to avoid DataParallel issues
        dataloader_num_workers=0,  # Disable multiprocessing to avoid hangs
        ddp_find_unused_parameters=False,
    )
    
    print(f"[GRPO] Config: batch_size={args.batch_size}, generation_batch_size={generation_batch_size}")
    print(f"[GRPO] Config: num_generations={args.num_generations}")
    print(f"[GRPO] Config: max_steps={args.max_steps}, dataset_size={len(train_dataset)}")
    
    # Create reward function
    reward_funcs = [compute_dora_reward]
    
    # Create trainer
    # Use base GRPOTrainer (QwenGRPOTrainer is only needed for vision inputs)
    # For text-only models, use tokenizer as processing_class
    print("[GRPO] Creating GRPOTrainer...")
    
    # Check model device before creating trainer
    print(f"[GRPO] Model device before trainer: {next(model.parameters()).device}")
    print(f"[GRPO] Model type: {type(model)}")
    
    # Create a custom trainer class that sets signature columns correctly
    # This is critical - GRPO needs to know which columns to keep
    class TextOnlyGRPOTrainer(GRPOTrainer):
        def _set_signature_columns_if_needed(self):
            # Override to set signature columns for text-only data
            # This prevents GRPO from trying to remove columns it needs
            if self._signature_columns is None:
                # For text-only, we need: prompt (for generation) and all fields for reward function
                self._signature_columns = ["prompt", "question", "answer", "transcript"]
                print(f"[GRPO] Set signature columns: {self._signature_columns}")
    
    trainer = TextOnlyGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,  # Use tokenizer for text-only models
        reward_funcs=reward_funcs,
        peft_config=peft_config,
    )
    
    # Check if model got wrapped after trainer creation
    if hasattr(trainer, 'model'):
        trainer_model = trainer.model
        # Unwrap DataParallel if it got wrapped (this can cause hangs)
        if isinstance(trainer_model, torch.nn.DataParallel):
            print(f"[GRPO] WARNING: Trainer model is wrapped in DataParallel, unwrapping...")
            trainer_model = trainer_model.module
            trainer.model = trainer_model
        print(f"[GRPO] Trainer model device: {next(trainer_model.parameters()).device if hasattr(trainer_model, 'parameters') else 'N/A'}")
    
    # Monkey-patch BOTH _generate_and_score_completions AND _generate
    # _generate_and_score_completions is what gets called during training
    print("[GRPO] Monkey-patching generation methods...")
    
    # Patch _generate_and_score_completions (the main entry point)
    original_generate_and_score = trainer._generate_and_score_completions
    def debug_generate_and_score(self, inputs):
        print(f"\n[DEBUG] ===== _generate_and_score_completions CALLED =====")
        print(f"[DEBUG] Number of inputs: {len(inputs)}")
        if inputs:
            print(f"[DEBUG] First input keys: {list(inputs[0].keys())}")
        import time
        import sys
        sys.stdout.flush()
        start = time.time()
        print(f"[DEBUG] Starting _generate_and_score_completions at {time.strftime('%H:%M:%S')}...")
        sys.stdout.flush()
        try:
            result = original_generate_and_score(inputs)
            elapsed = time.time() - start
            print(f"[DEBUG] ===== _generate_and_score_completions COMPLETED in {elapsed:.1f}s =====")
            sys.stdout.flush()
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f"[DEBUG] ===== _generate_and_score_completions FAILED after {elapsed:.1f}s =====")
            print(f"[DEBUG] Error: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
    
    # Patch _generate (called from within _generate_and_score_completions)
    original_generate = trainer._generate
    def debug_generate(self, prompts):
        print(f"\n[DEBUG GENERATE] ===== _generate CALLED =====")
        print(f"[DEBUG GENERATE] Number of prompts: {len(prompts)}")
        if prompts:
            print(f"[DEBUG GENERATE] First prompt length: {len(prompts[0])} chars")
        import time
        import sys
        sys.stdout.flush()
        start = time.time()
        try:
            result = original_generate(prompts)
            elapsed = time.time() - start
            print(f"[DEBUG GENERATE] ===== _generate COMPLETED in {elapsed:.1f}s =====")
            sys.stdout.flush()
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f"[DEBUG GENERATE] ===== _generate FAILED after {elapsed:.1f}s =====")
            print(f"[DEBUG GENERATE] Error: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
    
    # Also patch training_step to see if it's being called
    # Note: GRPOTrainer.training_step signature is (self, model, inputs, num_items_in_batch)
    # num_items_in_batch is REQUIRED, not optional
    original_training_step = trainer.training_step
    def debug_training_step(self, model, inputs, num_items_in_batch):
        print(f"\n[DEBUG TRAINING] ===== training_step CALLED =====")
        print(f"[DEBUG TRAINING] Input type: {type(inputs)}")
        if isinstance(inputs, dict):
            print(f"[DEBUG TRAINING] Input keys: {list(inputs.keys())}")
        elif isinstance(inputs, list):
            print(f"[DEBUG TRAINING] Input is list with {len(inputs)} items")
        print(f"[DEBUG TRAINING] num_items_in_batch: {num_items_in_batch}")
        import time
        import sys
        sys.stdout.flush()
        start = time.time()
        try:
            # GRPOTrainer requires num_items_in_batch - always pass it
            result = original_training_step(model, inputs, num_items_in_batch)
            elapsed = time.time() - start
            
            # Extract loss from result if available
            loss_info = "N/A"
            if isinstance(result, dict):
                loss_info = result.get('loss', 'N/A')
                if loss_info != 'N/A' and isinstance(loss_info, (int, float, torch.Tensor)):
                    if isinstance(loss_info, torch.Tensor):
                        loss_info = loss_info.item()
                    loss_info = f"{loss_info:.4f}"
            elif isinstance(result, (int, float, torch.Tensor)):
                # Result might be the loss directly
                if isinstance(result, torch.Tensor):
                    loss_info = f"{result.item():.4f}"
                else:
                    loss_info = f"{result:.4f}"
            
            print(f"[DEBUG TRAINING] ===== training_step COMPLETED in {elapsed:.1f}s, Loss: {loss_info} =====")
            sys.stdout.flush()
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f"[DEBUG TRAINING] ===== training_step FAILED after {elapsed:.1f}s =====")
            print(f"[DEBUG TRAINING] Error: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
    
    # Bind all methods
    import types
    trainer._generate_and_score_completions = types.MethodType(debug_generate_and_score, trainer)
    trainer._generate = types.MethodType(debug_generate, trainer)
    trainer.training_step = types.MethodType(debug_training_step, trainer)
    print("[GRPO] All training and generation methods patched successfully")
    
    # Train
    print("[GRPO] Starting training...")
    print("[GRPO] Debug: About to call trainer.train()")
    print(f"[GRPO] Debug: Dataset has {len(train_dataset)} examples")
    print(f"[GRPO] Debug: First example keys: {list(train_dataset[0].keys())}")
    print(f"[GRPO] Debug: First example prompt length: {len(train_dataset[0].get('prompt', ''))}")
    
    # Add callback to see progress
    from transformers import TrainerCallback
    import sys
    
    class DebugCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            print(f"[DEBUG CALLBACK] Step {state.global_step} beginning...")
            sys.stdout.flush()
        def on_step_end(self, args, state, control, **kwargs):
            # Loss is logged in on_log, not here, so check log_history
            loss_info = "N/A"
            if state.log_history:
                last_log = state.log_history[-1]
                # GRPO logs loss as 'loss' key in the log dict
                loss_info = last_log.get('loss', 'N/A')
                if loss_info != 'N/A' and isinstance(loss_info, (int, float)):
                    loss_info = f"{loss_info:.4f}"
            print(f"[DEBUG CALLBACK] Step {state.global_step} ended. Loss: {loss_info}")
            sys.stdout.flush()
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                # Extract key metrics from logs
                loss = logs.get('loss', 'N/A')
                reward = logs.get('reward', 'N/A')
                if loss != 'N/A':
                    print(f"[DEBUG CALLBACK] Step {state.global_step} - Loss: {loss:.4f}, Reward: {reward:.4f}" if reward != 'N/A' else f"[DEBUG CALLBACK] Step {state.global_step} - Loss: {loss:.4f}")
                else:
                    print(f"[DEBUG CALLBACK] Logs at step {state.global_step}: {logs}")
                sys.stdout.flush()
        def on_train_begin(self, args, state, control, **kwargs):
            print("[DEBUG CALLBACK] Training beginning...")
            sys.stdout.flush()
        def on_prediction_step(self, args, state, control, **kwargs):
            print("[DEBUG CALLBACK] Prediction step...")
            sys.stdout.flush()
        def on_save(self, args, state, control, **kwargs):
            print(f"[DEBUG CALLBACK] Saving checkpoint at step {state.global_step}...")
            sys.stdout.flush()
        def on_train_batch_begin(self, args, state, control, **kwargs):
            print(f"[DEBUG CALLBACK] Training batch beginning at step {state.global_step}...")
            sys.stdout.flush()
        def on_train_batch_end(self, args, state, control, **kwargs):
            print(f"[DEBUG CALLBACK] Training batch ended at step {state.global_step}...")
            sys.stdout.flush()
    
    trainer.add_callback(DebugCallback())
    
    print("[GRPO] About to call trainer.train()...")
    print(f"[GRPO] Note: First step will generate completions (this can take 2-5 minutes)")
    print("[GRPO] Watch for '[DEBUG GENERATE]' messages to see when generation starts")
    print("[GRPO] Watch for '[REWARD DEBUG]' messages to see when generation completes")
    print("[GRPO] If you don't see '[DEBUG GENERATE]' within 30 seconds, generation may be stuck")
    sys.stdout.flush()
    
    # Test if we can manually trigger generation (this helps debug hangs)
    print("\n[GRPO] Testing manual generation to verify it works...")
    try:
        # Get a simple test prompt from the dataset
        test_item = train_dataset[0]
        test_prompt = test_item.get("prompt", "Test prompt")
        print(f"[GRPO] Test prompt length: {len(test_prompt)} chars")
        print(f"[GRPO] Attempting manual generation (this may take 30-60 seconds)...")
        sys.stdout.flush()
        
        # Try to generate manually using the model
        with torch.no_grad():
            # Tokenize the prompt
            inputs = tokenizer(test_prompt, return_tensors="pt", padding=True).to(device)
            print(f"[GRPO] Tokenized input shape: {inputs['input_ids'].shape}")
            sys.stdout.flush()
            
            # Generate
            print(f"[GRPO] Starting generation...")
            sys.stdout.flush()
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
            print(f"[GRPO] Generation completed! Output shape: {outputs.shape}")
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[GRPO] Generated text (first 200 chars): {decoded[:200]}...")
            sys.stdout.flush()
        print("[GRPO] Manual generation test PASSED - model can generate\n")
    except Exception as e:
        print(f"[GRPO] Manual generation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("[GRPO] Continuing anyway - this might indicate the issue...\n")
    sys.stdout.flush()
    
    # Test if we can iterate over the dataset before training
    print("\n[GRPO] Testing dataset iteration...")
    try:
        from torch.utils.data import DataLoader
        test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        print(f"[GRPO] Created test DataLoader with {len(test_loader)} batches")
        sys.stdout.flush()
        
        # Try to get the first batch
        print("[GRPO] Attempting to get first batch from DataLoader...")
        sys.stdout.flush()
        first_batch = next(iter(test_loader))
        print(f"[GRPO] First batch retrieved! Keys: {list(first_batch.keys())}")
        sys.stdout.flush()
        print("[GRPO] Dataset iteration test PASSED\n")
    except Exception as e:
        print(f"[GRPO] Dataset iteration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("[GRPO] Continuing anyway...\n")
    sys.stdout.flush()
    
    try:
        import time
        start_time = time.time()
        print(f"\n[GRPO] ===== STARTING TRAINING at {time.strftime('%H:%M:%S')} =====\n")
        sys.stdout.flush()
        
        # Add a timeout mechanism - if nothing happens in 60 seconds, print a warning
        import threading
        training_started = threading.Event()
        def timeout_watcher():
            if not training_started.wait(timeout=60):
                print("\n[GRPO] WARNING: No training step activity detected in 60 seconds!")
                print("[GRPO] This suggests the training loop is stuck in data loading or initialization")
                print("[GRPO] Check if DataLoader is hanging or if there's a multiprocessing issue\n")
                sys.stdout.flush()
        
        watcher_thread = threading.Thread(target=timeout_watcher, daemon=True)
        watcher_thread.start()
        
        # Patch training_step to set the event when it's called
        # Note: GRPOTrainer.training_step signature is (self, model, inputs, num_items_in_batch)
        # num_items_in_batch is REQUIRED, not optional
        import types
        original_training_step_patched = trainer.training_step
        def training_step_with_event(self, model, inputs, num_items_in_batch):
            training_started.set()
            # Always pass num_items_in_batch - it's required by GRPOTrainer
            return original_training_step_patched(model, inputs, num_items_in_batch)
        trainer.training_step = types.MethodType(training_step_with_event, trainer)
        
        trainer.train()
        training_started.set()  # In case it completes before timeout
        elapsed = time.time() - start_time
        print(f"\n[GRPO] ===== TRAINING COMPLETED in {elapsed/60:.1f} minutes =====\n")
    except KeyboardInterrupt:
        print("\n[GRPO] Training interrupted by user")
        raise
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        print(f"\n[GRPO] ===== TRAINING FAILED after {elapsed:.1f}s =====\n")
        print(f"[GRPO] Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Save
    print(f"[GRPO] Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"[GRPO] Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()


