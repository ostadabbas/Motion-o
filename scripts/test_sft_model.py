#!/usr/bin/env python3
"""
Quick test to verify SFT model generates v2 motion tags.

Tests on a temporal-spatial sample with confirmed motion tags in GT.
Mimics the training collate_fn for input preparation.
"""
import json
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from training.train_sft_v2 import prepare_dataset
from training.vision_process import process_vision_info
from training.data_loader_v2 import SYSTEM_PROMPT
from configs.data_root import DATA_ROOT

STR_KF_ROOT = os.path.join(DATA_ROOT, "videos/stgr/temporal_grounding/kfs")
STR_PLM_KF_ROOT = os.path.join(DATA_ROOT, "videos/stgr/plm/kfs")


def find_motion_sample(data, require_moving=True):
    """Find first temporal-spatial sample with v2 motion tags."""
    for i, s in enumerate(data):
        if s.get('task') != 'temporal-spatial free-form QA':
            continue
        rp = s.get('reasoning_process', '')
        if '<motion ' not in rp:
            continue
        tags = re.findall(r'<motion\s+[^/]*?/>', rp)
        if not tags:
            continue
        # If require_moving, skip samples where all tags are STAT
        if require_moving and all('dir="STAT"' in t for t in tags):
            continue
        return i, s, tags
    return None, None, None


def prepare_input(example, processor):
    """Prepare input mimicking training collate_fn. System+User only."""
    prepared = prepare_dataset(example)
    messages = prepared["messages"][:2]  # System + User

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    if prepared["task"] == "temporal-spatial free-form QA":
        w, h = video_inputs[0].size(3), video_inputs[0].size(2)
        image_size = (w, h)

        kf_root = STR_PLM_KF_ROOT if prepared['source'] == "STR_plm_rdcap" else STR_KF_ROOT
        key_frames = []
        for kf in prepared["key_frames"]:
            img = Image.open(os.path.join(kf_root, kf["path"])).convert('RGB').resize(image_size)
            arr = torch.from_numpy(np.transpose(np.array(img), (2, 0, 1)))
            key_frames.append((kf["time"], arr))

        frame_prompt = ""
        refined = []
        kf_idx = ori_idx = 0
        frame_idx = 1
        while ori_idx < len(video_inputs[0]):
            t = int(ori_idx / video_kwargs['fps'][0])
            if kf_idx < len(key_frames) and t >= key_frames[kf_idx][0]:
                refined.append(key_frames[kf_idx][1])
                t = key_frames[kf_idx][0]
                frame_prompt += f"Frame {frame_idx} at {t}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                kf_idx += 1
            else:
                refined.append(video_inputs[0][ori_idx])
                t = round(ori_idx / video_kwargs['fps'][0], 1)
                frame_prompt += f"Frame {frame_idx} at {t}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                ori_idx += 1
            frame_idx += 1

        text = text.replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)
        inputs = processor(
            text=[text], images=[torch.stack(refined)], videos=None,
            return_tensors="pt", padding=True, do_resize=False
        )
    else:
        frame_prompt = ""
        ori_idx = 0
        while ori_idx < len(video_inputs[0]):
            t = round(ori_idx / video_kwargs['fps'][0], 1)
            frame_prompt += f"Frame {ori_idx+1} at {t}s: <|vision_start|><|image_pad|><|vision_end|>\n"
            ori_idx += 1
        frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
        text = text.replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)
        inputs = processor(
            text=[text], images=video_inputs, videos=None,
            return_tensors="pt", padding=True, do_resize=False
        )

    return inputs


def main():
    print("=" * 70)
    print("SFT v2 Motion Tag Test")
    print("=" * 70)

    # --- Config ---
    # model_path = "outputs/motiono_sft_v2_4456532/merged"
    # model_path = "outputs/motiono_sft_v2_4482512/merged"
    # model_path = "outputs/motiono_sft_dense_4553555/merged"
    model_path = "outputs/open-o3_motion_sft_4666166/merged"
    dataset_path = os.path.join(DATA_ROOT, "json_data/STGR-SFT-filtered-motion-densebbox.json")

    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # --- Load model ---
    print(f"\n1. Loading model: {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("   ✅ Model loaded")

    # --- Find sample ---
    print(f"\n2. Finding temporal-spatial sample with motion tags...")
    with open(dataset_path) as f:
        data = json.load(f)

    idx, sample, gt_tags = find_motion_sample(data, require_moving=True)
    if sample is None:
        print("   ❌ No suitable sample found!")
        sys.exit(1)

    print(f"   Sample #{idx}")
    print(f"   Task: {sample['task']}")
    print(f"   Q: {sample['question'][:80]}...")
    print(f"   GT motion tags ({len(gt_tags)}):")
    for t in gt_tags:
        print(f"     {t}")

    # Show GT reasoning (first 300 chars) for reference
    gt_rp = sample.get('reasoning_process', '')
    print(f"\n   GT reasoning (first 300 chars):")
    print(f"   {gt_rp[:300]}...")

    # --- Prepare input ---
    print(f"\n3. Preparing input...")
    try:
        inputs = prepare_input(sample, processor)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print(f"   ✅ Input prepared ({inputs['input_ids'].shape[1]} tokens)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # --- Generate ---
    print(f"\n4. Generating (max_new_tokens=512)...")
    torch.cuda.empty_cache()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512,
            do_sample=False, temperature=None, top_p=None,
        )
    gen_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = processor.decode(gen_ids, skip_special_tokens=True)
    print("   ✅ Done")

    # --- Display ---
    print("\n" + "=" * 70)
    print("GENERATED RESPONSE:")
    print("=" * 70)
    print(response)
    print("=" * 70)

    # --- Analyze ---
    print("\n5. Tag analysis:")
    checks = {
        "<think>":    '<think>' in response,
        "<answer>":   '<answer>' in response,
        "<obj>":      '<obj>' in response,
        "<box>":      '<box>' in response,
        "<t>...</t>s": bool(re.search(r'<t>[\d.]+</t>s', response)),
        "<motion ../>": bool(re.search(r'<motion\s+[^/]*?/>', response)),
        "<motion>..v1": '<motion>' in response and '</motion>' in response,
    }
    for tag, found in checks.items():
        print(f"   {tag:20s} {'✅' if found else '❌'}")

    # Extract any motion tags found
    v2_tags = re.findall(r'<motion\s+[^/]*?/>', response)
    v1_tags = re.findall(r'<motion>([^<]+)</motion>', response)

    if v2_tags:
        print(f"\n   v2 motion tags ({len(v2_tags)}):")
        for t in v2_tags:
            print(f"     {t}")
    if v1_tags:
        print(f"\n   v1 motion tags ({len(v1_tags)}):")
        for t in v1_tags:
            print(f"     <motion>{t}</motion>")

    # --- Verdict ---
    print("\n" + "=" * 70)
    has_grounding = checks["<obj>"] and checks["<box>"] and checks["<t>...</t>s"]
    has_motion = checks["<motion ../>"] or checks["<motion>..v1"]

    if has_grounding and has_motion:
        print("🎉 FULL SUCCESS — grounding + motion tags")
    elif has_grounding:
        print("⚠️  PARTIAL — grounding works, motion tags missing")
        print("   Motion may emerge during GRPO, or needs more SFT epochs")
    elif has_motion:
        print("⚠️  PARTIAL — motion tags present but grounding missing")
    else:
        print("❌ FAIL — no grounding or motion tags")
        print("   SFT didn't learn the format. Try:")
        print("   1. learning_rate 1e-5 (current 1e-6 too low)")
        print("   2. num_train_epochs 2-3")
        print("   3. Verify base Open-o3 SFT works first")

    return has_grounding and has_motion


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(1)
    success = main()
    sys.exit(0 if success else 1)