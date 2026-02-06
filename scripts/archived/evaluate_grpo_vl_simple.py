#!/usr/bin/env python3
"""
Simple evaluation script for GRPO-trained Vision-Language models on Dora Q&A dataset.
Aligned with the training pipeline structure.
"""
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_from_disk
from peft import PeftModel
from src.grpo_dataset import DoraGRPODataset
from src.grpo_reward import extract_final_answer, tokenize_answer, compute_f1_score
from src.ppo_trainer_simple import string_f1
from torch.utils.data import DataLoader
import random


def load_model_and_processor(model_id: str, checkpoint_path: str = None, use_lora: bool = False):
    """Load model and processor, with optional LoRA weights."""
    print(f"\n[1/3] Loading processor from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("✓ Processor loaded")
    
    print(f"\n[2/3] Loading model from {model_id}...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load checkpoint weights if specified
    if checkpoint_path:
        import os
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        
        if use_lora and os.path.exists(adapter_config_path):
            # Load LoRA weights
            print(f"  Loading LoRA weights from {checkpoint_path}...")
            model = PeftModel.from_pretrained(model, checkpoint_path)
            model = model.merge_and_unload()  # Merge LoRA weights for evaluation
            print("✓ LoRA weights loaded and merged")
        else:
            # Load full model checkpoint (reload from checkpoint path instead of base model)
            print(f"  Loading model checkpoint from {checkpoint_path}...")
            model = AutoModelForVision2Seq.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            print("✓ Model checkpoint loaded")
    
    model.eval()
    print("✓ Model loaded and set to eval mode")
    
    return model, processor


def generate_answer(model, processor, messages: List[Dict[str, Any]], max_new_tokens: int = 256):
    """
    Generate answer from messages (same format as training).
    
    Args:
        model: The model
        processor: The processor
        messages: Messages in TRL format (same as training)
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated text (full completion)
    """
    # Extract images from messages (same format as training dataset)
    images = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    images.append(item["image"])
    
    # Build chat template text (same as training)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Prepare inputs using processor (same pattern as model_loader.py)
    inputs = processor(
        text=[text],
        images=images if images else None,
        padding=True,
        return_tensors="pt",
    )
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate (same sampling params as training)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
        )
    
    # Decode only newly generated tokens (skip prompt)
    input_len = inputs["input_ids"].shape[-1]
    generated_ids_trimmed = generated_ids[:, input_len:]
    
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    
    return response


def evaluate_model(
    model,
    processor,
    dataset: DoraGRPODataset,
    max_examples: int = None,
    max_new_tokens: int = 256,
):
    """
    Evaluate model on dataset.
    
    Returns:
        List of evaluation results with input, output, ground truth, and metrics
    """
    results = []
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    # loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # indices = torch.randperm(len(dataset))
    # shuffled_dataset = [dataset[i] for i in indices]

    print(f"\n[3/3] Evaluating on {num_examples} examples...")
    print("=" * 80)
    
    for i in range(num_examples):
    # for batch in loader:
        random_idx = random.randint(0, num_examples)
        item = dataset[random_idx]
        # item = {k: v[0] for k, v in batch.items()}
        messages = item["prompt"]
        question = item["question"]
        gold_answer = item["answer"]
        
        print(f"\n[Example {i+1}/{num_examples}]")
        print(f"Question: {question}")
        print(f"Gold Answer: {gold_answer}")
        
        # Generate prediction
        try:
            generated_text = generate_answer(model, processor, messages, max_new_tokens)
            
            # Extract final answer (same logic as reward function)
            pred_answer = extract_final_answer(generated_text)
            
            # Compute F1 score (same logic as reward function)
            pred_tokens = tokenize_answer(pred_answer)
            gold_tokens = tokenize_answer(gold_answer)
            f1_score = string_f1(pred_tokens, gold_tokens)
            
            print(f"Generated: {generated_text[:200]}..." if len(generated_text) > 200 else f"Generated: {generated_text}")
            print(f"Predicted Answer: {pred_answer}")
            print(f"F1 Score: {f1_score:.3f}")
            
            results.append({
                "index": i,
                "question": question,
                "gold_answer": gold_answer,
                "generated_text": generated_text,
                "predicted_answer": pred_answer,
                "f1_score": f1_score,
            })
            
        except Exception as e:
            print(f"ERROR generating answer: {e}")
            results.append({
                "index": i,
                "question": question,
                "gold_answer": gold_answer,
                "generated_text": "",
                "predicted_answer": "",
                "f1_score": 0.0,
                "error": str(e),
            })
    
    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print evaluation summary."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    if not results:
        print("No results to summarize.")
        return
    
    # Compute statistics
    f1_scores = [r["f1_score"] for r in results if "error" not in r]
    num_errors = sum(1 for r in results if "error" in r)
    
    if f1_scores:
        mean_f1 = sum(f1_scores) / len(f1_scores)
        max_f1 = max(f1_scores)
        min_f1 = min(f1_scores)
        perfect_matches = sum(1 for f1 in f1_scores if f1 == 1.0)
        
        print(f"\nTotal Examples: {len(results)}")
        print(f"Errors: {num_errors}")
        print(f"Successful Evaluations: {len(f1_scores)}")
        print(f"\nF1 Score Statistics:")
        print(f"  Mean: {mean_f1:.3f}")
        print(f"  Max:  {max_f1:.3f}")
        print(f"  Min:  {min_f1:.3f}")
        print(f"  Perfect Matches (F1=1.0): {perfect_matches}/{len(f1_scores)} ({100*perfect_matches/len(f1_scores):.1f}%)")
    else:
        print("No successful evaluations.")
    
    # Show detailed results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    
    for r in results:
        print(f"\n[Example {r['index']+1}]")
        print(f"  Question: {r['question']}")
        print(f"  Gold:     {r['gold_answer']}")
        print(f"  Pred:     {r['predicted_answer']}")
        print(f"  F1:       {r['f1_score']:.3f}")
        if "error" in r:
            print(f"  ERROR:    {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GRPO-trained Qwen2-VL-2B-Instruct on Dora dataset"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        default="./outputs/dataset",
        nargs="?",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (for LoRA, specify the checkpoint directory)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Base model ID",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Load LoRA weights from checkpoint",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=4,
        help="Maximum number of frames to use per example",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
        help="Maximum prompt length in tokens",
    )
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None and args.use_lora:
        # Try to find latest checkpoint in output directory
        output_dir = Path("./outputs/grpo_vl")
        if output_dir.exists():
            checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
            if checkpoints:
                checkpoint_path = str(checkpoints[-1])
                print(f"Found checkpoint: {checkpoint_path}")
    
    # Load dataset
    print(f"\n[0/3] Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    print(f"✓ Dataset loaded: {len(dataset)} examples")
    
    # Load model and processor
    model, processor = load_model_and_processor(
        args.model_id,
        checkpoint_path=checkpoint_path,
        use_lora=args.use_lora,
    )
    
    # Prepare dataset (same format as training)
    eval_dataset = DoraGRPODataset(
        dataset=dataset,
        processor=processor,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_new_tokens,
        use_frames=True,
        max_frames=args.max_frames,
        system_prompt="You are a helpful visual reasoning assistant for kids.\nThink step by step, then give a final concise answer.",
    )
    
    # Evaluate
    results = evaluate_model(
        model,
        processor,
        eval_dataset,
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Print summary
    print_summary(results)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()

# python scripts/evaluate_grpo_vl_simple.py /projects/XXXX-1/dora/grpo_dataset_updatedv2/ --checkpoint ./outputs/grpo_dora_vl_no_context/checkpoint-450  --use-lora