#!/usr/bin/env python3
"""
Evaluate a model's ability to predict <motion/> tag attributes (dir, speed, scale).

Loads the SFT dataset, runs inference on samples that contain ground-truth motion
tags in their reasoning_process, parses predicted motion tags, and computes
per-attribute accuracy.

Usage:
    python scripts/eval_motion_tags.py \
        --model_path outputs/.../merged \
        --dataset_json /scratch/bai.xiang/Open-o3-Video/json_data/STGR-SFT-motion-mixed.json \
        --output_file evaluation/logs/motion_tags_eval.json \
        [--max_samples 200] [--batch_size 4]
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


MOTION_SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user provides a video and asks a question, "
    "and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind "
    "and then provide the user with the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual "
    "evidence from the video. When you mention any related object, person, or specific visual element "
    "in the reasoning process, you must strictly follow the following format: "
    "`<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. "
    "After the last observation of each object, you MUST describe its motion trajectory using a self-closing "
    "motion tag with discrete attributes: "
    '`<motion obj="object_name" dir="DIRECTION" speed="SPEED" scale="SCALE"/>` '
    "where DIRECTION is one of {N, NE, E, SE, S, SW, W, NW, STAT}, "
    "SPEED is one of {stationary, slow, moderate, fast}, "
    "and SCALE is one of {approaching, stable, receding}. "
    "The answer part only requires a text response; tags like <obj>, <box>, <t> are not needed."
)

MOTION_TAG_RE = re.compile(
    r'<motion\s+'
    r'obj="(?P<obj>[^"]+)"\s+'
    r'dir="(?P<dir>[^"]+)"\s+'
    r'speed="(?P<speed>[^"]+)"\s+'
    r'scale="(?P<scale>[^"]+)"\s*/>'
)

VALID_DIRS = {"N", "NE", "E", "SE", "S", "SW", "W", "NW", "STAT"}
VALID_SPEEDS = {"stationary", "slow", "moderate", "fast"}
VALID_SCALES = {"approaching", "stable", "receding"}


def parse_motion_tags(text: str) -> list[dict]:
    """Extract all <motion .../> tags from text, returning list of attribute dicts."""
    tags = []
    for m in MOTION_TAG_RE.finditer(text):
        tags.append({
            "obj": m.group("obj").strip().lower(),
            "dir": m.group("dir").strip(),
            "speed": m.group("speed").strip().lower(),
            "scale": m.group("scale").strip().lower(),
        })
    return tags


def obj_similarity(a: str, b: str) -> float:
    """Fuzzy similarity between two object names."""
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.9
    return SequenceMatcher(None, a, b).ratio()


def match_tags(gt_tags: list[dict], pred_tags: list[dict], threshold: float = 0.5):
    """
    Greedily match predicted tags to GT tags by object name similarity.
    Returns list of (gt_tag, pred_tag_or_None) pairs.
    """
    used_pred = set()
    matches = []
    for gt in gt_tags:
        best_idx, best_sim = -1, threshold
        for j, pred in enumerate(pred_tags):
            if j in used_pred:
                continue
            sim = obj_similarity(gt["obj"], pred["obj"])
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        if best_idx >= 0:
            used_pred.add(best_idx)
            matches.append((gt, pred_tags[best_idx]))
        else:
            matches.append((gt, None))
    return matches


def build_vllm_input(processor, sample: dict, video_max_pixels: int, video_max_frames: int):
    """Build a single vLLM input dict from a dataset sample."""
    video_path = sample["video_path_full"]
    question = sample["question"]

    messages = [
        {"role": "system", "content": MOTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": video_max_pixels,
                    "max_frames": video_max_frames,
                },
                {"type": "text", "text": question},
            ],
        },
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    mm_data = {"video": video_inputs[0].numpy()} if video_inputs else {}
    return {"prompt": prompt, "multi_modal_data": mm_data}


def main():
    parser = argparse.ArgumentParser(description="Evaluate motion tag accuracy")
    parser.add_argument("--model_path", required=True, help="Path to merged model")
    parser.add_argument("--dataset_json", required=True, help="Path to SFT dataset JSON")
    parser.add_argument("--output_file", default=None, help="Where to save detailed results JSON")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation samples")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--video_max_pixels", type=int, default=2097152)
    parser.add_argument("--video_max_frames", type=int, default=16)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    # ── Load dataset ──
    print(f"Loading dataset: {args.dataset_json}")
    with open(args.dataset_json) as f:
        data = json.load(f)

    samples = [s for s in data if MOTION_TAG_RE.search(s.get("reasoning_process", ""))]
    print(f"  Total samples: {len(data)}, with motion tags: {len(samples)}")

    # Filter to samples with accessible videos
    valid_samples = []
    for s in samples:
        vpath = s.get("video_path_full", "")
        if vpath and os.path.isfile(vpath):
            valid_samples.append(s)
    print(f"  With accessible video files: {len(valid_samples)}")

    if args.max_samples and len(valid_samples) > args.max_samples:
        valid_samples = valid_samples[: args.max_samples]
        print(f"  Truncated to: {len(valid_samples)} samples")

    if not valid_samples:
        print("ERROR: No valid samples found. Check video paths.")
        sys.exit(1)

    # ── Init model ──
    print(f"\nLoading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 32, "video": 10},
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        repetition_penalty=1.05,
        max_tokens=args.max_tokens,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"

    # ── Run inference in batches ──
    print(f"\nRunning inference on {len(valid_samples)} samples (batch_size={args.batch_size})...")
    all_results = []
    counters = {
        "total_gt_tags": 0,
        "matched_tags": 0,
        "dir_correct": 0,
        "speed_correct": 0,
        "scale_correct": 0,
        "all_correct": 0,
        "no_pred_tags": 0,
        "skipped_errors": 0,
    }
    dir_confusion = defaultdict(Counter)
    speed_confusion = defaultdict(Counter)
    scale_confusion = defaultdict(Counter)

    t_start = time.time()

    for batch_start in range(0, len(valid_samples), args.batch_size):
        batch = valid_samples[batch_start : batch_start + args.batch_size]
        batch_idx = batch_start // args.batch_size + 1
        total_batches = (len(valid_samples) + args.batch_size - 1) // args.batch_size

        # Build inputs, skip samples that fail (e.g. corrupted video)
        inputs = []
        batch_valid = []
        for s in batch:
            try:
                inp = build_vllm_input(
                    processor, s, args.video_max_pixels, args.video_max_frames
                )
                inputs.append(inp)
                batch_valid.append(s)
            except Exception as e:
                print(f"  WARN: Skipping {s['id']} — {e}")
                counters["skipped_errors"] += 1

        if not inputs:
            continue

        outputs = llm.generate(inputs, sampling_params=sampling_params)

        for s, out in zip(batch_valid, outputs):
            pred_text = out.outputs[0].text
            gt_tags = parse_motion_tags(s["reasoning_process"])
            pred_tags = parse_motion_tags(pred_text)

            counters["total_gt_tags"] += len(gt_tags)

            if not pred_tags:
                counters["no_pred_tags"] += 1

            matches = match_tags(gt_tags, pred_tags)
            sample_result = {
                "id": s["id"],
                "question": s["question"],
                "gt_tags": gt_tags,
                "pred_tags": pred_tags,
                "matches": [],
            }

            for gt, pred in matches:
                match_info = {"gt": gt, "pred": pred}
                if pred is not None:
                    counters["matched_tags"] += 1
                    d_ok = gt["dir"] == pred["dir"]
                    s_ok = gt["speed"] == pred["speed"]
                    sc_ok = gt["scale"] == pred["scale"]
                    counters["dir_correct"] += int(d_ok)
                    counters["speed_correct"] += int(s_ok)
                    counters["scale_correct"] += int(sc_ok)
                    counters["all_correct"] += int(d_ok and s_ok and sc_ok)
                    dir_confusion[gt["dir"]][pred["dir"]] += 1
                    speed_confusion[gt["speed"]][pred["speed"]] += 1
                    scale_confusion[gt["scale"]][pred["scale"]] += 1
                    match_info["dir_ok"] = d_ok
                    match_info["speed_ok"] = s_ok
                    match_info["scale_ok"] = sc_ok
                sample_result["matches"].append(match_info)

            all_results.append(sample_result)

        elapsed = time.time() - t_start
        rate = (batch_start + len(batch)) / elapsed if elapsed > 0 else 0
        print(
            f"  Batch {batch_idx}/{total_batches} | "
            f"Processed: {batch_start + len(batch)}/{len(valid_samples)} | "
            f"{rate:.1f} samples/s"
        )

    elapsed_total = time.time() - t_start

    # ── Compute metrics ──
    total = counters["total_gt_tags"]
    matched = counters["matched_tags"]

    def pct(n, d):
        return f"{100 * n / d:.1f}%" if d > 0 else "N/A"

    print("\n" + "=" * 60)
    print("MOTION TAG EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples evaluated:  {len(all_results)}")
    print(f"Samples skipped:    {counters['skipped_errors']}")
    print(f"Samples w/o preds:  {counters['no_pred_tags']}")
    print(f"Total GT tags:      {total}")
    print(f"Matched (by obj):   {matched}/{total} ({pct(matched, total)})")
    print(f"Time:               {elapsed_total:.0f}s")
    print()
    print("--- Accuracy (over matched tags) ---")
    print(f"  Direction:  {counters['dir_correct']}/{matched} ({pct(counters['dir_correct'], matched)})")
    print(f"  Speed:      {counters['speed_correct']}/{matched} ({pct(counters['speed_correct'], matched)})")
    print(f"  Scale:      {counters['scale_correct']}/{matched} ({pct(counters['scale_correct'], matched)})")
    print(f"  All three:  {counters['all_correct']}/{matched} ({pct(counters['all_correct'], matched)})")
    print()
    print("--- Accuracy (over all GT tags, unmatched = wrong) ---")
    print(f"  Direction:  {counters['dir_correct']}/{total} ({pct(counters['dir_correct'], total)})")
    print(f"  Speed:      {counters['speed_correct']}/{total} ({pct(counters['speed_correct'], total)})")
    print(f"  Scale:      {counters['scale_correct']}/{total} ({pct(counters['scale_correct'], total)})")
    print(f"  All three:  {counters['all_correct']}/{total} ({pct(counters['all_correct'], total)})")

    # Per-class breakdown
    for attr_name, confusion, valid_set in [
        ("Direction", dir_confusion, VALID_DIRS),
        ("Speed", speed_confusion, VALID_SPEEDS),
        ("Scale", scale_confusion, VALID_SCALES),
    ]:
        print(f"\n--- {attr_name} per-class accuracy ---")
        for cls in sorted(valid_set):
            if cls.lower() not in confusion and cls not in confusion:
                continue
            row = confusion.get(cls, confusion.get(cls.lower(), Counter()))
            total_cls = sum(row.values())
            correct_cls = row.get(cls, 0) + (row.get(cls.lower(), 0) if cls != cls.lower() else 0)
            print(f"  {cls:>12s}: {correct_cls}/{total_cls} ({pct(correct_cls, total_cls)})")

    # ── Save results ──
    output_file = args.output_file
    if output_file is None:
        os.makedirs("evaluation/logs/motion_tags_logs", exist_ok=True)
        output_file = "evaluation/logs/motion_tags_logs/motion_eval_results.json"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_payload = {
        "model_path": args.model_path,
        "dataset_json": args.dataset_json,
        "num_samples": len(all_results),
        "counters": counters,
        "accuracy": {
            "direction_matched": pct(counters["dir_correct"], matched),
            "speed_matched": pct(counters["speed_correct"], matched),
            "scale_matched": pct(counters["scale_correct"], matched),
            "all_matched": pct(counters["all_correct"], matched),
            "direction_total": pct(counters["dir_correct"], total),
            "speed_total": pct(counters["speed_correct"], total),
            "scale_total": pct(counters["scale_correct"], total),
            "all_total": pct(counters["all_correct"], total),
        },
        "per_sample": all_results,
    }
    with open(output_file, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
