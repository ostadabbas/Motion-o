#!/usr/bin/env python3
"""
Dataset verification script for Dora GRPO training.

Inspects dataset structure, verifies frames are properly formatted,
and tests data loading to ensure everything is ready for training.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_from_disk, Dataset, DatasetDict


def inspect_dataset_structure(dataset_path: str) -> bool:
    """Inspect the dataset structure and verify required fields."""
    print("=" * 80)
    print("DATASET STRUCTURE INSPECTION")
    print("=" * 80)
    
    # Load dataset
    dataset_path_obj = Path(dataset_path)
    if not dataset_path_obj.exists():
        print(f"❌ ERROR: Dataset path does not exist: {dataset_path}")
        return False
    
    try:
        if dataset_path_obj.is_dir():
            dataset = load_from_disk(str(dataset_path))
        else:
            from datasets import load_dataset
            dataset = load_dataset("json", data_files=str(dataset_path))["train"]
    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Handle DatasetDict
    if isinstance(dataset, DatasetDict):
        print(f"📁 Dataset is a DatasetDict with splits: {list(dataset.keys())}")
        # Use train split if available, otherwise first split
        dataset = dataset.get("train", list(dataset.values())[0])
    
    print(f"✓ Dataset loaded successfully")
    print(f"  - Total examples: {len(dataset)}")
    print(f"  - Dataset type: {type(dataset).__name__}")
    
    if len(dataset) == 0:
        print("❌ ERROR: Dataset is empty!")
        return False
    
    # Inspect first example
    print("\n" + "-" * 80)
    print("INSPECTING FIRST EXAMPLE")
    print("-" * 80)
    
    first_item = dataset[0]
    print(f"Keys in dataset item: {list(first_item.keys())}")
    
    # Check required fields
    required_fields = ["transcript", "question", "answer"]
    missing_fields = [f for f in required_fields if f not in first_item]
    
    if missing_fields:
        print(f"❌ ERROR: Missing required fields: {missing_fields}")
        return False
    
    print(f"✓ All required fields present")
    
    # Inspect each field
    print("\nField Details:")
    for key in first_item.keys():
        value = first_item[key]
        value_type = type(value).__name__
        
        if key == "frames":
            if value is None:
                print(f"  - {key}: None (no frames)")
            elif isinstance(value, list):
                print(f"  - {key}: list with {len(value)} items")
                if len(value) > 0:
                    frame_item = value[0]
                    if isinstance(frame_item, dict):
                        print(f"    - Frame item type: dict with keys: {list(frame_item.keys())}")
                        if "image" in frame_item:
                            img_data = frame_item["image"]
                            if isinstance(img_data, np.ndarray):
                                print(f"    - Image array shape: {img_data.shape}, dtype: {img_data.dtype}")
                            elif isinstance(img_data, list):
                                print(f"    - Image is nested list (will convert to numpy array)")
                    elif isinstance(frame_item, Image.Image):
                        print(f"    - Frame item type: PIL.Image, size: {frame_item.size}")
            else:
                print(f"  - {key}: {value_type} (unexpected type)")
        elif key in ["transcript", "question", "answer"]:
            text_len = len(str(value)) if value else 0
            print(f"  - {key}: {value_type}, length: {text_len} chars")
            if text_len > 0:
                preview = str(value)[:100].replace("\n", "\\n")
                print(f"    Preview: {preview}...")
        else:
            print(f"  - {key}: {value_type}")
    
    return True


def test_frame_conversion(dataset_path: str) -> bool:
    """Test converting frames from numpy arrays to PIL Images."""
    print("\n" + "=" * 80)
    print("TESTING FRAME CONVERSION")
    print("=" * 80)
    
    # Load dataset
    dataset_path_obj = Path(dataset_path)
    try:
        if dataset_path_obj.is_dir():
            dataset = load_from_disk(str(dataset_path))
        else:
            from datasets import load_dataset
            dataset = load_dataset("json", data_files=str(dataset_path))["train"]
    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset: {e}")
        return False
    
    if isinstance(dataset, DatasetDict):
        dataset = dataset.get("train", list(dataset.values())[0])
    
    # Find an example with frames
    example_with_frames = None
    for i in range(min(10, len(dataset))):
        item = dataset[i]
        if "frames" in item and item["frames"] and len(item["frames"]) > 0:
            example_with_frames = item
            print(f"✓ Found example with frames at index {i}")
            break
    
    if example_with_frames is None:
        print("⚠️  WARNING: No examples with frames found in first 10 examples")
        print("   This is okay if you're doing text-only training")
        return True
    
    frames = example_with_frames["frames"]
    print(f"  - Number of frames: {len(frames)}")
    
    # Test conversion
    converted_images = []
    for i, frame_data in enumerate(frames[:3]):  # Test first 3 frames
        try:
            if isinstance(frame_data, dict):
                img_array = frame_data.get("image")
                if img_array is not None:
                    # Convert nested list to numpy array if needed
                    if isinstance(img_array, list):
                        img_array = np.array(img_array, dtype=np.uint8)
                    elif isinstance(img_array, np.ndarray):
                        img_array = img_array.astype(np.uint8)
                    else:
                        print(f"  ❌ Frame {i}: Unexpected image type: {type(img_array)}")
                        return False
                    
                    # Convert to PIL Image
                    if len(img_array.shape) == 3:
                        img = Image.fromarray(img_array)
                        converted_images.append(img)
                        print(f"  ✓ Frame {i}: Converted to PIL.Image, size: {img.size}, mode: {img.mode}")
                    else:
                        print(f"  ❌ Frame {i}: Invalid array shape: {img_array.shape}")
                        return False
            elif isinstance(frame_data, Image.Image):
                converted_images.append(frame_data)
                print(f"  ✓ Frame {i}: Already PIL.Image, size: {frame_data.size}")
            else:
                print(f"  ❌ Frame {i}: Unexpected frame type: {type(frame_data)}")
                return False
        except Exception as e:
            print(f"  ❌ Frame {i}: Conversion error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n✓ Successfully converted {len(converted_images)} frames to PIL Images")
    return True


def test_data_formatting(dataset_path: str, model_id: str = "Qwen/Qwen2-VL-2B-Instruct") -> bool:
    """Test formatting data for GRPO training."""
    print("\n" + "=" * 80)
    print("TESTING DATA FORMATTING FOR GRPO")
    print("=" * 80)
    
    # Load dataset
    dataset_path_obj = Path(dataset_path)
    try:
        if dataset_path_obj.is_dir():
            dataset = load_from_disk(str(dataset_path))
        else:
            from datasets import load_dataset
            dataset = load_dataset("json", data_files=str(dataset_path))["train"]
    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset: {e}")
        return False
    
    if isinstance(dataset, DatasetDict):
        dataset = dataset.get("train", list(dataset.values())[0])
    
    # Load processor
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        processor.image_processor.do_resize = False
        print(f"✓ Processor loaded: {model_id}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load processor: {e}")
        return False
    
    # Test formatting one example
    item = dataset[0]
    transcript = item.get("transcript", "")
    question = item.get("question", "")
    answer = item.get("answer", "")
    
    print(f"\nTesting format for example:")
    print(f"  - Transcript length: {len(transcript)} chars")
    print(f"  - Question: {question[:80]}...")
    print(f"  - Answer: {answer[:80]}...")
    
    # Convert frames
    images = None
    if "frames" in item and item["frames"]:
        images = []
        for frame_data in item["frames"][:4]:  # Max 4 frames
            if isinstance(frame_data, dict):
                img_array = frame_data.get("image")
                if img_array is not None:
                    if isinstance(img_array, list):
                        img_array = np.array(img_array, dtype=np.uint8)
                    elif isinstance(img_array, np.ndarray):
                        img_array = img_array.astype(np.uint8)
                    if len(img_array.shape) == 3:
                        img = Image.fromarray(img_array)
                        images.append(img)
            elif isinstance(frame_data, Image.Image):
                images.append(frame_data)
        if not images:
            images = None
    
    # Build messages format
    system_prompt = "You are a helpful visual reasoning assistant for kids.\nThink step by step, then give a final concise answer."
    
    user_content_list = []
    if images:
        for img in images:
            user_content_list.append({"type": "image", "image": img})
        print(f"  - Added {len(images)} images to prompt")
    
    # Truncate transcript if needed (simple version for testing)
    max_transcript_chars = 500
    transcript_for_prompt = transcript
    if len(transcript_for_prompt) > max_transcript_chars:
        transcript_for_prompt = "..." + transcript_for_prompt[-(max_transcript_chars-3):]
        print(f"  - Truncated transcript to {len(transcript_for_prompt)} chars")
    
    user_text = f"{system_prompt}\n\nContext: {transcript_for_prompt}\nQuestion: {question}\nAnswer:"
    user_content_list.append({"type": "text", "text": user_text})
    
    messages = [{"role": "user", "content": user_content_list}]
    
    print(f"\n✓ Built messages format:")
    print(f"  - Messages type: {type(messages).__name__}")
    print(f"  - Number of messages: {len(messages)}")
    print(f"  - First message role: {messages[0]['role']}")
    print(f"  - Content items: {len(messages[0]['content'])}")
    
    # Test apply_chat_template
    try:
        template_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"✓ apply_chat_template (no tokenize) works")
        print(f"  - Template length: {len(template_text)} chars")
        
        # Count tokens
        if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
            tokens = processor.tokenizer(template_text, add_special_tokens=False, return_tensors="pt")
            num_tokens = tokens.input_ids.shape[-1]
            print(f"  - Token count: {num_tokens}")
    except Exception as e:
        print(f"❌ ERROR: apply_chat_template failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n✓ Data formatting test PASSED")
    return True


def print_sample_examples(dataset_path: str, num_examples: int = 3):
    """Print sample examples for manual inspection."""
    print("\n" + "=" * 80)
    print(f"SAMPLE EXAMPLES (showing {num_examples})")
    print("=" * 80)
    
    # Load dataset
    dataset_path_obj = Path(dataset_path)
    try:
        if dataset_path_obj.is_dir():
            dataset = load_from_disk(str(dataset_path))
        else:
            from datasets import load_dataset
            dataset = load_dataset("json", data_files=str(dataset_path))["train"]
    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset: {e}")
        return
    
    if isinstance(dataset, DatasetDict):
        dataset = dataset.get("train", list(dataset.values())[0])
    
    for i in range(min(num_examples, len(dataset))):
        item = dataset[i]
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {item.get('question', 'N/A')}")
        print(f"Answer: {item.get('answer', 'N/A')}")
        transcript = item.get('transcript', '')
        print(f"Transcript (first 200 chars): {transcript[:200]}...")
        if "frames" in item and item["frames"]:
            print(f"Frames: {len(item['frames'])} frames available")
        else:
            print(f"Frames: None")


def main():
    parser = argparse.ArgumentParser(description="Verify Dora dataset for GRPO training")
    parser.add_argument(
        "dataset_path",
        type=str,
        default="./outputs/dataset",
        nargs="?",
        help="Path to dataset directory or file"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Model ID for testing processor"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample examples to print"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("DORA DATASET VERIFICATION")
    print("=" * 80)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Model ID: {args.model_id}")
    print("=" * 80 + "\n")
    
    # Run all tests
    results = []
    
    # Test 1: Structure inspection
    results.append(("Structure Inspection", inspect_dataset_structure(args.dataset_path)))
    
    # Test 2: Frame conversion
    results.append(("Frame Conversion", test_frame_conversion(args.dataset_path)))
    
    # Test 3: Data formatting
    results.append(("Data Formatting", test_data_formatting(args.dataset_path, args.model_id)))
    
    # Print samples
    print_sample_examples(args.dataset_path, args.samples)
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    print("=" * 80)
    if all_passed:
        print("✅ ALL VERIFICATION TESTS PASSED")
        print("Dataset is ready for GRPO training!")
    else:
        print("❌ SOME VERIFICATION TESTS FAILED")
        print("Please fix the issues before proceeding with training.")
    print("=" * 80)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

