#!/usr/bin/env python3
"""
Quick test to verify SFT model generates motion tags
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from training.data_loader import SYSTEM_PROMPT
import torch

# Load SFT checkpoint
model_path = "outputs/sft_full_slurm_631"
print(f"Loading model from: {model_path}")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# Load a sample from the dataset
data_path = "/mnt/data/stgr/json_data/STGR-SFT-subset.json"
print(f"Loading test sample from: {data_path}")

with open(data_path, 'r') as f:
    data = json.load(f)

# Find a sample with tracking data (key_items with multiple frames)
test_sample = None
for item in data:
    if len(item.get('key_items', {})) >= 2:
        test_sample = item
        print(f"\nFound sample with tracking data:")
        print(f"  Video: {item.get('video', 'N/A')}")
        print(f"  Key items: {len(item.get('key_items', {}))} frames")
        print(f"  Question: {item['conversations'][0]['value'][:100]}...")
        break

if not test_sample:
    print("ERROR: No samples with tracking data found!")
    sys.exit(1)

# Prepare input
video_path = test_sample['video']
question = test_sample['conversations'][0]['value']
system_prompt = SYSTEM_PROMPT.get(test_sample.get('task_type', 'temporal-spatial free-form QA'))

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": question},
        ],
    }
]

# Prepare for inference
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Generate
print("\n" + "="*80)
print("GENERATING OUTPUT...")
print("="*80)

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,  # Greedy decoding for consistency
    )

# Decode
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("\n" + "="*80)
print("MODEL OUTPUT:")
print("="*80)
print(output_text)
print("="*80)

# Check for motion tags
has_motion_tags = "<motion>" in output_text and "</motion>" in output_text
has_think_tags = "<think>" in output_text and "</think>" in output_text
has_obj_tags = "<obj>" in output_text and "</obj>" in output_text
has_box_tags = "<box>" in output_text and "</box>" in output_text
has_time_tags = "<t>" in output_text and "</t>" in output_text

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)
print(f"✓ Has <think> tags: {has_think_tags}")
print(f"✓ Has <obj> tags: {has_obj_tags}")
print(f"✓ Has <box> tags: {has_box_tags}")
print(f"✓ Has <t> tags: {has_time_tags}")
print(f"{'✓' if has_motion_tags else '✗'} Has <motion> tags: {has_motion_tags}")

if has_motion_tags:
    # Extract motion content
    import re
    motion_matches = re.findall(r'<motion>(.*?)</motion>', output_text, re.DOTALL)
    print(f"\nFound {len(motion_matches)} <motion> tag(s):")
    for i, motion in enumerate(motion_matches, 1):
        print(f"  {i}. {motion.strip()}")
else:
    print("\n⚠️  WARNING: Model did NOT generate <motion> tags!")
    print("   This means SFT did not learn the motion format.")

print("\n" + "="*80)
print("GROUND TRUTH (for comparison):")
print("="*80)
if 'reasoning_process' in test_sample:
    gt = test_sample['reasoning_process'][:500]
    print(gt + "..." if len(test_sample['reasoning_process']) > 500 else gt)
    print(f"\n✓ Ground truth has <motion>: {'<motion>' in test_sample['reasoning_process']}")
