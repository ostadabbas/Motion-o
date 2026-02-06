#!/usr/bin/env python3
"""
Generate multiple VQA visualization examples and save them in a directory.
"""
import json
import sys
from pathlib import Path
import subprocess
from evaluate_grpo_vl_simple import load_model_and_processor, generate_answer
from src.grpo_dataset import DoraGRPODataset
from src.grpo_reward import extract_final_answer, tokenize_answer
from src.ppo_trainer_simple import string_f1
from tqdm import tqdm
from datasets import load_from_disk
from collections import defaultdict
import json
import random

def load_model(model_base, ckpt_pth=None):
    if ckpt_pth is not None:
        model, processor = load_model_and_processor(
                model_base,
                checkpoint_path=ckpt_pth,
                use_lora=True,
            )
    else:
        model, processor = load_model_and_processor(
                model_base,
                use_lora=True,
            )
    return model, processor

def answer_question(model, processor, item):
    max_new_tokens = 256
    messages = item["prompt"]
    generated_text = generate_answer(model, processor, messages, max_new_tokens)
    pred_answer = extract_final_answer(generated_text)

def main():
    if len(sys.argv) < 4:
        print("Usage: python generate_vqa_examples.py <video_path> <labels_path> <output_dir> [num_examples]")
        print("\nExample:")
        print("  python generate_vqa_examples.py \\")
        print("    /mnt/data/dora/mp4/S1/Dora.the.explorer.s01e01.avi \\")
        print("    outputs/filtered_labels/S1/Dora.the.Explorer.S01E01.WEBRip.Amazon_formatted_filtered.json \\")
        print("    outputs/vqa_examples \\")
        print("    10")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    labels_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    num_examples = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    if not labels_path.exists():
        print(f"❌ Labels file not found: {labels_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    if not isinstance(labels, list):
        print(f"❌ Labels file is not a JSON array")
        sys.exit(1)
    
    # Limit to available entries
    num_examples = min(num_examples, len(labels))
    
    print(f"Generating {num_examples} visualization examples...")
    print(f"Output directory: {output_dir}\n")
    
    # Generate visualizations
    script_path = Path(__file__).parent / "visualize_vqa_structure.py"
    
    for i in range(num_examples):
        entry = labels[i]
        question = entry.get("question_text", "")[:50]
        
        output_file = output_dir / f"vqa_example_{i:03d}.png"
        
        print(f"[{i+1}/{num_examples}] Entry {i}: {question}...")
        
        # Call visualization script
        cmd = [
            sys.executable,
            str(script_path),
            str(video_path),
            str(labels_path),
            "--entry-index", str(i),
            "--output", str(output_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"  ✓ Saved: {output_file.name}")
            else:
                print(f"  ❌ Error: {result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  Timeout")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✓ Generated {num_examples} examples in {output_dir}")

if __name__ == "__main__":
    main()
