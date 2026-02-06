#!/usr/bin/env python3
"""
Evaluation script for trained models on Dora Q&A dataset.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import VLM
from src.eval_utils import simple_accuracy, normalize_text
from datasets import load_from_disk, Dataset, DatasetDict
import json
import numpy as np
from PIL import Image


def evaluate_model(vlm: VLM,
                   dataset: Dataset,
                   max_examples: int = None,
                   use_frames: bool = True) -> Dict[str, Any]:
    """
    Evaluate model on dataset.
    
    Args:
        vlm: VLM model instance
        dataset: Evaluation dataset
        max_examples: Maximum number of examples to evaluate
        use_frames: Whether to use frames (True) or text only (False)
        
    Returns:
        Dictionary with evaluation metrics
    """
    predictions = []
    ground_truth = []
    correct = 0
    total = 0
    
    max_examples = max_examples or len(dataset)
    
    for i, item in enumerate(dataset):
        if i >= max_examples:
            break
        
        # Extract frames if available and using frames
        frames = []
        if use_frames and "frames" in item:
            for frame_data in item["frames"]:
                if isinstance(frame_data, dict):
                    img_array = frame_data.get("image")
                    if img_array is not None:
                        if isinstance(img_array, np.ndarray):
                            img = Image.fromarray(img_array.astype(np.uint8))
                        else:
                            img = Image.fromarray(np.array(img_array).astype(np.uint8))
                        frames.append((img, frame_data.get("timestamp", 0.0)))
        
        # Debug: print frame info
        if i == 0:  # Only for first example
            print(f"\n[DEBUG] Number of frames: {len(frames)}")
            if frames:
                print(f"[DEBUG] Frame timestamps: {[f[1] for f in frames[:5]]}...")  # First 5
            print(f"[DEBUG] Transcript length: {len(item.get('transcript', ''))}")
            print(f"[DEBUG] Transcript preview: {item.get('transcript', '')[:200]}...")
        
        # Generate prediction
        system_prompt = "You are a helpful visual reasoning assistant for kids."
        user_prompt = (
            f"Context: {item.get('transcript', '')}\n"
            f"Question: {item['question']}\n"
            f"Answer:"
        )
        
        try:
            if frames:
                prediction = vlm.generate(
                    frames=frames,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=32,
                    temperature=0.3,
                    top_p=0.9
                )
            else:
                # Text-only mode
                prediction = vlm.generate(
                    frames=[],
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=32,
                    temperature=0.3,
                    top_p=0.9
                )
            
            answer = item.get("answer", "")
            
            # Compute accuracy
            acc = simple_accuracy(prediction, answer)
            correct += acc
            total += 1

            # Debug: print detailed example info
            print("\n--- Example", i, "---")
            print("Question:", item.get("question", ""))
            print("Ground truth:", answer)
            print("Prediction:", prediction)
            
            predictions.append(prediction)
            ground_truth.append(answer)
            
            if (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{min(max_examples, len(dataset))} examples...")
        
        except Exception as e:
            print(f"Error evaluating example {i}: {e}")
            continue
    
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
        description="Evaluate trained model on Dora Q&A dataset"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset directory or file"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Model ID or path to trained model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model directory (overrides model-id)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test", "val"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Evaluate in text-only mode (no frames)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for evaluation results JSON"
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
        eval_dataset = dataset.get(args.split, dataset.get("test", None))
        if eval_dataset is None and args.split == "val":
            eval_dataset = dataset.get("validation", None)
    else:
        eval_dataset = dataset
    
    if eval_dataset is None:
        raise ValueError(f"Dataset split '{args.split}' not found")
    
    print(f"Evaluation dataset: {len(eval_dataset)} examples")
    
    # Load model
    model_path = args.model_path or args.model_id
    print(f"Loading model from {model_path}...")
    vlm = VLM(
        model_id=model_path,
        device=args.device,
        dtype="auto",
        load_4bit=True,
        cache_dir=args.cache_dir
    )
    
    # Evaluate
    print(f"\nEvaluating model ({'text-only' if args.text_only else 'with frames'})...")
    results = evaluate_model(
        vlm,
        eval_dataset,
        max_examples=args.max_examples,
        use_frames=not args.text_only
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print("="*50)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON (remove predictions/ground_truth if too long)
        save_results = {
            "accuracy": results["accuracy"],
            "correct": results["correct"],
            "total": results["total"],
            "model_id": model_path,
            "split": args.split,
            "text_only": args.text_only,
        }
        
        # Optionally include predictions
        if len(results["predictions"]) <= 100:
            save_results["predictions"] = results["predictions"]
            save_results["ground_truth"] = results["ground_truth"]
        
        with open(output_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

