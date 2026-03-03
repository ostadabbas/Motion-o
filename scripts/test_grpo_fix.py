#!/usr/bin/env python3
"""
Smoke-test the GRPO vision-token fix using REAL video data from the dataset.

Loads the merged model, picks real video samples from the training JSON,
processes them through the exact same pipeline as compute_loss, runs
generation, and checks for vision-token leaks + logps forward-pass crashes.

Usage (single GPU):
    python scripts/test_grpo_fix.py \
        --model outputs/open-o3_motion_sft_4666166/merged \
        --dataset /scratch/bai.xiang/Open-o3-Video/json_data/STGR-RL-filtered-motion-densebbox.json \
        --rounds 10 \
        --max_new_tokens 256
"""

import argparse
import copy
import json
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from configs.data_root import DATA_ROOT

ROOT = os.path.join(DATA_ROOT, "videos")
STR_DATA = os.path.join(ROOT, "stgr/temporal_grounding/videos")
STR_PLM_DATA = os.path.join(ROOT, "stgr/plm/videos")
TVG_ROOT = os.path.join(ROOT, "tvg_r1")
GENERAL_VIDEO_ROOT = os.path.join(ROOT, "videor1")
VIDEO_ESPRESSO_ROOT = ROOT

import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from training.vision_process import process_vision_info

VISION_SPECIAL_IDS = [151652, 151653, 151654, 151655, 151656]
VISION_NAMES = {
    151652: "<|vision_start|>",
    151653: "<|vision_end|>",
    151654: "<|vision_pad|>",
    151655: "<|image_pad|>",
    151656: "<|video_pad|>",
}

SYSTEM_PROMPTS = {
    "temporal-spatial free-form QA": (
        "A conversation between user and assistant. The user provides a video and asks a question, "
        "and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind "
        "and then provide the user with the answer. The reasoning process and answer are enclosed within "
        "<think> </think> and <answer> </answer> tags, respectively."
    ),
    "temporal QA": (
        "A conversation between user and assistant. The user provides a video and asks a question, "
        "and the Assistant determines the precise time period that answers the question."
    ),
    "General video QA MCQ": (
        "A conversation between user and assistant. The user provides a video and asks a multiple-choice question."
    ),
    "General video QA Free-form": (
        "A conversation between user and assistant. The user provides a video and asks a question."
    ),
}


def get_video_root(source):
    if source == "videoespresso_train_video":
        return VIDEO_ESPRESSO_ROOT
    elif "STR_plm" in source:
        return STR_PLM_DATA
    elif "STR" in source:
        return STR_DATA
    elif "TVG" in source:
        return TVG_ROOT
    elif "videor1" in source:
        return GENERAL_VIDEO_ROOT
    return STR_DATA


def build_prompt_and_inputs(sample, processor):
    """Replicate the exact prompt-building logic from grpo_trainer.compute_loss."""
    task = sample.get("task", "temporal-spatial free-form QA")
    source = sample.get("source", "STR")
    system_msg = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["temporal-spatial free-form QA"])

    video_root = get_video_root(source)
    video_path = os.path.join(video_root, sample["video_path"])

    if not os.path.isfile(video_path):
        return None, None, None

    input_msg = [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": sample.get("question", "What is happening in this video?")}
        ]},
    ]

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(input_msg, return_video_kwargs=True)
    except Exception as e:
        print(f"  Skipping sample: video processing failed: {e}")
        return None, None, None

    if video_inputs is None:
        return None, None, None

    text = processor.apply_chat_template(input_msg, tokenize=False, add_generation_prompt=True)

    # Convert video frames to individual images (same as compute_loss multi_image path)
    frame_prompt = ""
    for idx in range(len(video_inputs[0])):
        t = round(idx / video_kwargs["fps"][0], 1)
        frame_prompt += f"Frame {idx + 1} at {t}s: <|vision_start|><|image_pad|><|vision_end|>\n"
    frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
    text = text.replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)

    inputs = processor(
        text=[text],
        images=[video_inputs[0]],
        videos=None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    return inputs, video_path, task


def check_vision_tokens(token_ids, prompt_length):
    completion = token_ids[:, prompt_length:]
    found = {}
    for vid in VISION_SPECIAL_IDS:
        count = (completion == vid).sum().item()
        if count > 0:
            found[VISION_NAMES[vid]] = count
    return found


def run_logps_forward(model, prompt_completion_ids, pixel_values, image_grid_thw, prompt_length):
    """Simulate _get_per_token_logps — the exact crash path."""
    n_gen = prompt_completion_ids.shape[0]
    kwargs = {
        "pixel_values": pixel_values.repeat(n_gen, 1),
        "image_grid_thw": image_grid_thw.repeat(n_gen, 1),
    }
    with torch.no_grad():
        logits = model(prompt_completion_ids, **kwargs).logits
    logits = logits[:, :-1, :]
    ids = prompt_completion_ids[:, 1:]
    per_token_logps = []
    for lr, ir in zip(logits, ids):
        lp = lr.log_softmax(dim=-1)
        per_token_logps.append(torch.gather(lp, 1, ir.unsqueeze(1)).squeeze(1))
    return torch.stack(per_token_logps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, help="Path to training JSON")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:  {device}")
    print(f"Model:   {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Rounds:  {args.rounds}, gens/round: {args.num_generations}, max_tokens: {args.max_new_tokens}")
    print()

    # Load dataset
    with open(args.dataset) as f:
        data = json.load(f)
    # Filter to video samples only
    video_samples = [s for s in data if s.get("video_path")]
    random.shuffle(video_samples)
    print(f"Loaded {len(video_samples)} video samples from dataset")
    print()

    # Load model
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.max_pixels = 401408
    processor.image_processor.min_pixels = 3136
    pad_token_id = processor.tokenizer.pad_token_id
    print("Model loaded.\n")

    gen_unsafe = GenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=True, top_p=0.95,
        temperature=1.0, num_return_sequences=args.num_generations,
        pad_token_id=pad_token_id,
    )
    gen_safe = GenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=True, top_p=0.95,
        temperature=0.7, num_return_sequences=args.num_generations,
        pad_token_id=pad_token_id,
        suppress_tokens=VISION_SPECIAL_IDS,
    )

    sample_idx = 0
    stats = {"unsafe_leak": 0, "safe_leak": 0, "safe_pass": 0, "safe_fail": 0,
             "tested": 0, "skipped": 0}

    print("=" * 60)
    print("Running tests on real video data")
    print("=" * 60)

    while stats["tested"] < args.rounds and sample_idx < len(video_samples):
        sample = video_samples[sample_idx]
        sample_idx += 1

        inputs, vpath, task = build_prompt_and_inputs(sample, processor)
        if inputs is None:
            stats["skipped"] += 1
            continue

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Truncate prompt
        if inputs["input_ids"].shape[1] > args.max_prompt_length:
            inputs["input_ids"] = inputs["input_ids"][:, -args.max_prompt_length:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -args.max_prompt_length:]

        prompt_length = inputs["input_ids"].shape[1]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        round_num = stats["tested"] + 1
        vname = os.path.basename(vpath)
        n_frames = image_grid_thw.shape[0]
        print(f"\n--- Round {round_num}/{args.rounds}: {vname} ({task}, {n_frames} frames, {prompt_length} prompt tokens) ---")

        # TEST 1: Without fix
        try:
            with torch.no_grad():
                out_unsafe = model.generate(**inputs, generation_config=gen_unsafe)
            leaked = check_vision_tokens(out_unsafe, prompt_length)
            if leaked:
                stats["unsafe_leak"] += 1
                print(f"  [NO FIX] LEAKED: {leaked}")
            else:
                print(f"  [NO FIX] clean")
        except Exception as e:
            stats["unsafe_leak"] += 1
            print(f"  [NO FIX] CRASHED during generation: {e}")

        # TEST 2: With fix + logps forward
        try:
            with torch.no_grad():
                out_safe = model.generate(**inputs, generation_config=gen_safe)
            leaked = check_vision_tokens(out_safe, prompt_length)
            if leaked:
                stats["safe_leak"] += 1
                print(f"  [  FIX ] LEAKED despite suppress: {leaked}")
            else:
                print(f"  [  FIX ] clean generation")

            # Logps forward pass (the crash site)
            logps = run_logps_forward(model, out_safe, pixel_values, image_grid_thw, prompt_length)
            stats["safe_pass"] += 1
            print(f"  [  FIX ] logps forward PASSED (shape: {list(logps.shape)})")

        except Exception as e:
            stats["safe_fail"] += 1
            print(f"  [  FIX ] CRASHED: {e}")

        stats["tested"] += 1

        # Clear GPU memory between rounds
        torch.cuda.empty_cache()

    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"RESULTS ({stats['tested']} rounds tested, {stats['skipped']} skipped)")
    print(f"{'=' * 60}")
    print(f"  Without fix:  {stats['unsafe_leak']}/{stats['tested']} rounds leaked vision tokens")
    print(f"  With fix:     {stats['safe_leak']}/{stats['tested']} leaks, "
          f"{stats['safe_pass']} logps passed, {stats['safe_fail']} logps crashed")
    print()

    if stats["safe_fail"] == 0 and stats["safe_leak"] == 0:
        print("VERDICT: Fix is working — zero leaks, zero crashes on real video data.")
    elif stats["safe_fail"] > 0:
        print("VERDICT: Logps STILL crashed with fix — there may be a second issue.")
    elif stats["safe_leak"] > 0:
        print("VERDICT: Vision tokens leaked despite suppress_tokens — unexpected.")
    print("=" * 60)


if __name__ == "__main__":
    main()
