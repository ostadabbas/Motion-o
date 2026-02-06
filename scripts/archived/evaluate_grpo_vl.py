#!/usr/bin/env python3
"""
Evaluation script for GRPO-trained Vision-Language models on Dora Q&A dataset.
"""
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_from_disk, Dataset, DatasetDict
from src.eval_utils import simple_accuracy
from src.ppo_trainer_simple import extract_final_answer
from scripts.dora_grpo_dataset import truncate_transcript


def build_vl_prompt(item: Dict[str, Any], target_prompt_length: int = 1337, dataset=None, processor=None) -> Tuple[str, List[Image.Image], str]:
    """
    Build prompt for VL model with images, matching training format.
    
    Returns:
        Tuple of (prompt_text, images_list)
    """
    transcript = item.get("transcript", "") or ""
    question = item.get("question", "") or ""
    
    # Remove ALL questions (and their answers) from transcript to prevent answer leakage
    # This is critical for sequential Q&A pairs
    import re
    transcript_for_prompt = transcript
    if question and dataset:
        # Collect all questions from dataset to remove all Q&A pairs
        all_questions = []
        try:
            for j in range(len(dataset) if hasattr(dataset, '__len__') else 0):
                other_q = dataset[j].get("question", "").strip() if hasattr(dataset, '__getitem__') else ""
                if other_q and other_q not in all_questions:
                    all_questions.append(other_q)
        except:
            all_questions = [question] if question else []
        
        # Find the earliest position of any question in the sequence
        earliest_question_pos = len(transcript_for_prompt)
        
        for q in all_questions:
            if not q:
                continue
            # Try multiple patterns for robust matching
            patterns_to_try = [
                re.escape(q),
                re.escape(q.strip().lower()),
                re.escape(re.sub(r'[^\w\s]', '', q.strip().lower())),
            ]
            
            for pattern_str in patterns_to_try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                for match in pattern.finditer(transcript_for_prompt):
                    pos = match.start()
                    # Check if it's a good match (at sentence boundary)
                    if pos == 0 or transcript_for_prompt[max(0, pos-2):pos].strip() in ('', '.', '!', '?', '\n'):
                        if pos < earliest_question_pos:
                            earliest_question_pos = pos
                        break  # Found a match for this question, move to next
        
        # Truncate transcript to end BEFORE the first question
        # This removes ALL questions and their answers from the transcript
        if earliest_question_pos < len(transcript_for_prompt):
            transcript_for_prompt = transcript_for_prompt[:earliest_question_pos].rstrip()
            # Clean up: remove trailing sentence fragments
            transcript_for_prompt = re.sub(r'[.!?]\s*$', '', transcript_for_prompt).strip()
    elif question:
        # Fallback: just remove current question if dataset not available
        pattern = re.compile(re.escape(question), re.IGNORECASE)
        transcript_for_prompt = re.sub(pattern, "", transcript_for_prompt)
        transcript_for_prompt = re.sub(r"\s{2,}", " ", transcript_for_prompt).strip()
    
    # Extract multiple images
    images = []
    max_frames = 4
    if "frames" in item and item["frames"]:
        for frame_data in item["frames"][:max_frames]:
            if isinstance(frame_data, dict):
                img_array = np.array(frame_data.get("image"), dtype=np.uint8)
                if len(img_array.shape) == 3:
                    img = Image.fromarray(img_array).resize((448, 448), Image.Resampling.LANCZOS)
                    images.append(img)
            elif isinstance(frame_data, Image.Image):
                img = frame_data.resize((448, 448), Image.Resampling.LANCZOS)
                images.append(img)
    
    system_prompt = "You are a helpful visual reasoning assistant for kids.\nThink step by step, then give a final concise answer."
    max_prompt_tokens = 512  # standard prompt budget

    # Fit transcript so that final tokenized prompt (with vision tokens) stays within budget
    truncated_transcript = transcript_for_prompt
    template_text = None
    if processor and hasattr(processor, "tokenizer") and processor.tokenizer is not None and hasattr(processor, "apply_chat_template"):
        tokenizer = processor.tokenizer

        def build_messages(curr_transcript: str) -> List[Dict]:
            user_content_list = []
            if images:
                for img in images:
                    user_content_list.append({"type": "image", "image": img})
            user_text = f"Context: {curr_transcript}\nQuestion: {question}\nAnswer:"
            user_content_list.append({"type": "text", "text": user_text})

            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": user_content_list})
            return msgs

        curr_transcript = truncated_transcript
        for _ in range(5):
            msgs = build_messages(curr_transcript)
            candidate_template = processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            tok_len = processor.tokenizer(
                candidate_template, add_special_tokens=False, return_tensors="pt"
            ).input_ids.shape[-1]
            template_text = candidate_template
            if tok_len <= max_prompt_tokens:
                truncated_transcript = curr_transcript
                break
            excess = tok_len - max_prompt_tokens + 10
            t_tokens = tokenizer.encode(curr_transcript, add_special_tokens=False)
            if len(t_tokens) <= excess:
                curr_transcript = ""
            else:
                keep = len(t_tokens) - excess
                t_tokens = t_tokens[-keep:]
                curr_transcript = tokenizer.decode(t_tokens, skip_special_tokens=False)
                if keep > 10:
                    curr_transcript = "..." + curr_transcript[3:]
        truncated_transcript = curr_transcript
    else:
        # Fallback to character-based: rough approximation of 512-token budget
        target_length = 1500
        if len(truncated_transcript) > target_length:
            truncated_transcript = truncated_transcript[-target_length:]
            if target_length > 10:
                truncated_transcript = "..." + truncated_transcript[3:]

    # Build prompt text (with placeholder tokens like training)
    user_content = f"Context: {truncated_transcript}\nQuestion: {question}\nAnswer:"
    prompt_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    if template_text is None and processor and hasattr(processor, "apply_chat_template"):
        user_content_list = []
        if images:
            for img in images:
                user_content_list.append({"type": "image", "image": img})
        user_content_list.append({"type": "text", "text": user_content})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content_list})
        template_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif template_text is None:
        template_text = prompt_text

    return prompt_text, images, template_text


def evaluate_grpo_vl_model(
    model_path: str,
    dataset: Dataset,
    max_examples: int = None,
    device: str = "cuda",
    cache_dir: str = None,
    target_prompt_length: int = 1337,
) -> Dict[str, Any]:
    """
    Evaluate GRPO-trained VL model on dataset.
    
    Args:
        model_path: Path to trained model directory
        dataset: Evaluation dataset
        max_examples: Maximum number of examples to evaluate
        device: Device to use
        cache_dir: Cache directory for models
        target_prompt_length: Target prompt length for transcript truncation
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading model from {model_path}...")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    processor.image_processor.do_resize = False
    
    # Load model
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda:0")  # Will use GPU 1 due to CUDA_VISIBLE_DEVICES=1
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
            
            # Build prompt with images (already templated and trimmed inside build_vl_prompt)
            prompt_text, images, template_text = build_vl_prompt(item, target_prompt_length, dataset=dataset, processor=processor)
            
            # DEBUG: Print prompt details
            print(f"\n[DEBUG Example {i+1}]")
            print(f"  Question: {item.get('question', '')}")
            print(f"  Prompt text length: {len(prompt_text)}")
            print(f"  Prompt text (first 200 chars): {prompt_text[:200]}")
            print(f"  Number of images: {len(images)}")
            if images:
                print(f"  Image size: {images[0].size}")
            print(f"  Template text length: {len(template_text)}")
            print(f"  Template text (last 200 chars): {template_text[-200:]}")
            
            # Process with images
            processor_kwargs = dict(
                text=[template_text],
                images=[images] if images else None,
                return_tensors="pt",
                padding=True,
                do_resize=False,
            )
            inputs = processor(**processor_kwargs)
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            print(f"  Input IDs shape: {inputs['input_ids'].shape}")
            print(f"  Has pixel_values: {'pixel_values' in inputs}")
            if 'pixel_values' in inputs:
                print(f"  Pixel values shape: {inputs['pixel_values'].shape}")
            
            # Decode input to see what model sees
            input_text = processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
            print(f"  Decoded input (last 300 chars): {input_text[-300:]}")
            
            # Generate
            output = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=128,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                **{k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}
            )
            
            # Decode only newly generated tokens
            input_len = inputs["input_ids"].shape[-1]
            generated_tokens = output[0][input_len:]
            full_prediction = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
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
            print(f"Has images: {len(images) > 0}")
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
        description="Evaluate GRPO-trained VL model on Dora Q&A dataset"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset directory or file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./outputs/grpo_dora_vl",
        help="Path to trained GRPO VL model directory"
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
    parser.add_argument(
        "--target-prompt-length",
        type=int,
        default=1337,
        help="Target prompt length for transcript truncation"
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
    results = evaluate_grpo_vl_model(
        model_path=args.model_path,
        dataset=eval_dataset,
        max_examples=args.max_examples,
        device=args.device,
        cache_dir=args.cache_dir,
        target_prompt_length=args.target_prompt_length,
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

