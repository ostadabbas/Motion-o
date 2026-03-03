#!/usr/bin/env python3
"""
Quick smoke-test for the GRPO vision-token fix.

Loads the merged SFT model, runs generation on a single sample N times with
the same settings as GRPO (temperature, top_p, num_return_sequences), then
feeds prompt+completion back through the model for logps — exactly the path
that crashes.

Run on the cluster with ONE GPU:
    python scripts/test_grpo_fix.py \
        --model outputs/open-o3_motion_sft_4666166/merged \
        --rounds 20

It will report:
  - Whether any vision tokens were generated (with / without suppress_tokens)
  - Whether the logps forward pass succeeds or crashes
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
)

VISION_SPECIAL_IDS = [151652, 151653, 151654, 151655, 151656]
VISION_NAMES = {
    151652: "<|vision_start|>",
    151653: "<|vision_end|>",
    151654: "<|vision_pad|>",
    151655: "<|image_pad|>",
    151656: "<|video_pad|>",
}


def make_dummy_input(processor):
    """Create a minimal single-image prompt to test generation."""
    # 1x1 red pixel — smallest valid image
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe what you see in detail. Think step by step."},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt",
        padding=True,
    )
    return inputs


def check_for_vision_tokens(token_ids, prompt_length):
    """Check if any vision-special tokens appear in the completion portion."""
    completion = token_ids[:, prompt_length:]
    found = {}
    for vid in VISION_SPECIAL_IDS:
        count = (completion == vid).sum().item()
        if count > 0:
            found[VISION_NAMES[vid]] = count
    return found


def run_logps_forward(model, prompt_completion_ids, prompt_inputs, prompt_length):
    """Simulate _get_per_token_logps — the exact path that crashes."""
    # Pop input_ids/attention_mask, keep pixel_values + image_grid_thw
    kwargs = {}
    for k, v in prompt_inputs.items():
        if k in ("input_ids", "attention_mask"):
            continue
        if k == "pixel_values" or k == "image_grid_thw":
            kwargs[k] = v.repeat(prompt_completion_ids.shape[0], 1)
        elif k == "pixel_values_videos" or k == "video_grid_thw":
            kwargs[k] = v.repeat(prompt_completion_ids.shape[0], 1)
        elif k == "second_per_grid_ts":
            continue
        else:
            kwargs[k] = v

    with torch.no_grad():
        logits = model(prompt_completion_ids, **kwargs).logits
    logits = logits[:, :-1, :]
    input_ids = prompt_completion_ids[:, 1:]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids):
        lp = logits_row.log_softmax(dim=-1)
        per_token_logps.append(torch.gather(lp, 1, ids_row.unsqueeze(1)).squeeze(1))
    return torch.stack(per_token_logps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to merged model")
    parser.add_argument("--rounds", type=int, default=20, help="Number of generation rounds to test")
    parser.add_argument("--num_generations", type=int, default=4, help="Sequences per round")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Rounds: {args.rounds}, generations/round: {args.num_generations}")
    print()

    # Load model and processor
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(args.model)
    pad_token_id = processor.tokenizer.pad_token_id

    # Prepare input
    inputs = make_dummy_input(processor)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]

    # ============================================================
    # TEST 1: Generation WITHOUT suppress_tokens (the old behavior)
    # ============================================================
    print("=" * 60)
    print("TEST 1: Generation WITHOUT suppress_tokens")
    print("=" * 60)
    gen_config_unsafe = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=1.0,
        num_return_sequences=args.num_generations,
        pad_token_id=pad_token_id,
    )

    total_vision_leaked_unsafe = 0
    for r in range(args.rounds):
        with torch.no_grad():
            out = model.generate(**inputs, generation_config=gen_config_unsafe)
        found = check_for_vision_tokens(out, prompt_length)
        if found:
            total_vision_leaked_unsafe += 1
            print(f"  Round {r+1}: LEAKED vision tokens: {found}")
    print(f"\nResult: {total_vision_leaked_unsafe}/{args.rounds} rounds had vision tokens in completions")
    print()

    # ============================================================
    # TEST 2: Generation WITH suppress_tokens (the fix)
    # ============================================================
    print("=" * 60)
    print("TEST 2: Generation WITH suppress_tokens (the fix)")
    print("=" * 60)
    gen_config_safe = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=args.num_generations,
        pad_token_id=pad_token_id,
        suppress_tokens=VISION_SPECIAL_IDS,
    )

    total_vision_leaked_safe = 0
    logps_pass = 0
    logps_fail = 0
    for r in range(args.rounds):
        with torch.no_grad():
            out = model.generate(**inputs, generation_config=gen_config_safe)
        found = check_for_vision_tokens(out, prompt_length)
        if found:
            total_vision_leaked_safe += 1
            print(f"  Round {r+1}: LEAKED vision tokens despite suppress: {found}")

        # Now test the logps forward pass (the crash site)
        try:
            logps = run_logps_forward(model, out, inputs, prompt_length)
            logps_pass += 1
        except Exception as e:
            logps_fail += 1
            print(f"  Round {r+1}: LOGPS CRASH: {e}")

        if (r + 1) % 5 == 0:
            print(f"  ... {r+1}/{args.rounds} rounds done ({logps_pass} pass, {logps_fail} fail)")

    print(f"\nResult: {total_vision_leaked_safe}/{args.rounds} rounds had vision tokens in completions")
    print(f"Logps forward: {logps_pass} passed, {logps_fail} failed")

    # ============================================================
    # VERDICT
    # ============================================================
    print()
    print("=" * 60)
    if total_vision_leaked_unsafe > 0 and total_vision_leaked_safe == 0 and logps_fail == 0:
        print("VERDICT: Fix confirmed working!")
        print(f"  - Without fix: {total_vision_leaked_unsafe}/{args.rounds} rounds leaked vision tokens")
        print(f"  - With fix: 0 leaks, 0 crashes across {args.rounds} rounds")
    elif total_vision_leaked_unsafe == 0:
        print("VERDICT: No vision tokens leaked even WITHOUT the fix.")
        print("  The bug may require more rounds or real video data to trigger.")
        print("  The fix is still safe to ship (suppress_tokens is a no-op if tokens were never sampled).")
    elif logps_fail > 0:
        print("VERDICT: Logps still crashed WITH the fix — there may be a second issue.")
        print(f"  Crashes: {logps_fail}/{args.rounds}")
    else:
        print(f"VERDICT: Unexpected — {total_vision_leaked_safe} leaks despite suppress_tokens.")
    print("=" * 60)


if __name__ == "__main__":
    main()
