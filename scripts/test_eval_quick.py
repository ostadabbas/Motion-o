#!/usr/bin/env python3
"""Quick 2-sample test to verify the eval pipeline works."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evaluation'))
from models.model_vllm import QwenVL_VLLM

MERGED_MODEL = sys.argv[1] if len(sys.argv) > 1 else "outputs/grpo_dense_t07_4737145/checkpoint-800/merged"
VSTAR_ANNO = "/scratch/bai.xiang/eval_benchmarks/V-STaR/V_STaR_test.json"
VSTAR_VIDEOS = "/scratch/bai.xiang/eval_benchmarks/V-STaR/videos"

print(f"Model: {MERGED_MODEL}")
print("Loading model with vLLM...")

model = QwenVL_VLLM(
    MERGED_MODEL,
    rt_shape=True,
    temperature=0.0,
    max_tokens=256,
    video_max_pixels=2097152,
    video_max_frames=16,
)

print("Model loaded! Running 2-sample test...\n")

anno = json.load(open(VSTAR_ANNO))[:2]
for d in anno:
    vid = d["vid"]
    video_path = os.path.join(VSTAR_VIDEOS, f"{vid}.mp4")
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}, skipping")
        continue

    prompt = f"Answer the question about the video: {d['question']}"
    outputs, frames, fps, frame_shape = model([video_path], [prompt], [None])
    answer = outputs[0].outputs[0].text

    print(f"VID: {vid}")
    print(f"Q:   {d['question']}")
    print(f"A:   {answer[:300]}")
    print()

print("Test passed!")
