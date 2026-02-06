#!/usr/bin/env python3
"""
Evaluation script for PPO-trained text-only models on Dora Q&A dataset.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, Dataset, DatasetDict
from src.eval_utils import simple_accuracy
from src.ppo_trainer_simple import build_prompt, extract_final_answer


def evaluate_ppo_model(
    model_path: str,
    dataset: Dataset,
    max_examples: int = None,
    device: str = "cuda",
    cache_dir: str = None,
) -> Dict[str, Any]:
    """
    Evaluate PPO-trained text-only model on dataset.
    
    Args:
        model_path: Path to trained model directory
        dataset: Evaluation dataset
        max_examples: Maximum number of examples to evaluate
        device: Device to use
        cache_dir: Cache directory for models
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda:0")
    model.eval()
    
    print("Model loaded successfully!")
    
    predictions = []
    ground_truth = []
    correct = 0
    total = 0
    
    max_examples = max_examples or len(dataset)
    
    print(f"\nEvaluating on {min(max_examples, len(dataset))} examples...")
    print("="*80)
    
    with torch.no_grad():
        for i, item in enumerate(dataset):
            if i >= max_examples:
                break
            
            # Build prompt (same format as training)
            prompt = build_prompt(item)
            
            # Use chat template if available (for Qwen models)
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": "You are a helpful visual reasoning assistant for kids."},
                    {"role": "user", "content": f"Context: {item.get('transcript', '')}\nQuestion: {item.get('question', '')}\nAnswer (think step by step, then say 'Final answer: <answer>'):"}
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode only newly generated tokens
            input_len = inputs["input_ids"].shape[-1]
            generated_tokens = output[0][input_len:]
            full_prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Extract final answer from prediction (same as training)
            prediction = extract_final_answer(full_prediction)
            
            # Get ground truth
            answer = item.get("answer", "")
            
            # Compute accuracy on extracted final answer
            acc = simple_accuracy(prediction, answer)
            correct += acc
            total += 1
            
            # Print detailed example
            print(f"\n--- Example {i+1} ---")
            print(f"Transcript: {item.get('transcript', '')[:150]}...")
            print(f"Question: {item.get('question', '')}")
            print(f"Ground truth: {answer}")
            print(f"Full prediction:\n{full_prediction}")
            print(f"Extracted final answer: {prediction}")
            print(f"Accuracy: {acc}")
            print("-"*80)
            
            predictions.append(full_prediction)
            ground_truth.append(answer)
            
            if (i + 1) % 10 == 0:
                print(f"\nProgress: {i + 1}/{min(max_examples, len(dataset))} examples evaluated")
                print(f"Current accuracy: {correct/total:.4f} ({correct}/{total})")
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions,
        "ground_truth": ground_truth
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO-trained text-only model on Dora Q&A dataset"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset directory or file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./outputs/ppo_dora",
        help="Path to trained PPO model directory"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    try:
        dataset = load_from_disk(args.dataset_path) if Path(args.dataset_path).is_dir() else Dataset.load_from_disk(args.dataset_path)
    except:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=args.dataset_path)["train"]
    
    # Handle DatasetDict
    if isinstance(dataset, DatasetDict):
        eval_dataset = dataset.get("train", dataset.get("test", None))
    else:
        eval_dataset = dataset
    
    if eval_dataset is None:
        raise ValueError("No evaluation dataset found")
    
    print(f"Evaluation dataset: {len(eval_dataset)} examples")
    
    # Evaluate
    results = evaluate_ppo_model(
        model_path=args.model_path,
        dataset=eval_dataset,
        max_examples=args.max_examples,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    
    # Print results
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print("="*80)
    
    # Print some example predictions
    print("\nSample Predictions:")
    print("="*80)
    for i in range(min(5, len(results['predictions']))):
        print(f"\nExample {i+1}:")
        print(f"  Ground truth: {results['ground_truth'][i]}")
        print(f"  Full prediction:\n  {results['predictions'][i]}")
        print()


if __name__ == "__main__":
    main()

