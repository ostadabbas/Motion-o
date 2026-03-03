#!/usr/bin/env python3
"""
Diagnose a PEFT LoRA merge for Qwen2.5-VL models.

Compares the base model config, adapter config, and merged model config/weights
to find common merge bugs (missing vocab_size, wrong vision token IDs, corrupted
embeddings, etc.).

Usage:
    python scripts/diagnose_merge.py \
        --base  Open-o3-Video/Qwen2.5-VL-open-o3 \
        --adapter outputs/open-o3_motion_sft_4666166 \
        --merged  outputs/open-o3_motion_sft_4666166/merged
"""

import argparse
import json
import os
import sys

def load_json(path):
    with open(path) as f:
        return json.load(f)

def red(s):    return f"\033[91m[FAIL] {s}\033[0m"
def green(s):  return f"\033[92m[ OK ] {s}\033[0m"
def yellow(s): return f"\033[93m[WARN] {s}\033[0m"

def check_config(label, cfg_path, reference_cfg=None):
    print(f"\n{'='*60}")
    print(f"  {label}: {cfg_path}")
    print(f"{'='*60}")

    if not os.path.isfile(cfg_path):
        print(red(f"File not found: {cfg_path}"))
        return None

    cfg = load_json(cfg_path)

    # vocab_size
    vs = cfg.get("vocab_size")
    if vs is None:
        print(red(f"vocab_size is None (PEFT merge bug — must be 152064)"))
    elif vs != 152064:
        print(red(f"vocab_size = {vs}, expected 152064"))
    else:
        print(green(f"vocab_size = {vs}"))

    # vision token IDs
    expected = {
        "vision_start_token_id": 151652,
        "vision_end_token_id":   151653,
        "vision_token_id":       151654,
        "image_token_id":        151655,
        "video_token_id":        151656,
    }
    for key, val in expected.items():
        actual = cfg.get(key)
        if actual is None:
            print(red(f"{key} is MISSING"))
        elif actual != val:
            print(red(f"{key} = {actual}, expected {val}"))
        else:
            print(green(f"{key} = {actual}"))

    # model_type
    mt = cfg.get("model_type")
    if mt != "qwen2_5_vl":
        print(red(f"model_type = {mt!r}, expected 'qwen2_5_vl'"))
    else:
        print(green(f"model_type = {mt!r}"))

    # vision_config
    vc = cfg.get("vision_config")
    if vc is None:
        print(red("vision_config is MISSING entirely"))
    else:
        print(green(f"vision_config present (spatial_patch_size={vc.get('spatial_patch_size')})"))

    # rope_scaling
    rs = cfg.get("rope_scaling")
    if rs is None:
        print(red("rope_scaling is MISSING"))
    else:
        print(green(f"rope_scaling present (mrope_section={rs.get('mrope_section')})"))

    # architectures
    archs = cfg.get("architectures", [])
    if "Qwen2_5_VLForConditionalGeneration" not in archs:
        print(red(f"architectures = {archs}, expected ['Qwen2_5_VLForConditionalGeneration']"))
    else:
        print(green(f"architectures correct"))

    # Cross-check with reference
    if reference_cfg is not None:
        mismatches = []
        for key in reference_cfg:
            if key in ("_name_or_path", "transformers_version", "use_cache"):
                continue
            if key not in cfg:
                mismatches.append(f"  MISSING: {key}")
            elif cfg[key] != reference_cfg[key]:
                mismatches.append(f"  DIFFERS: {key}: merged={cfg[key]!r}  vs  base={reference_cfg[key]!r}")
        if mismatches:
            print(yellow(f"Fields differing from base config:"))
            for m in mismatches:
                print(f"    {m}")
        else:
            print(green("All fields match base config"))

    return cfg


def check_adapter(adapter_dir):
    print(f"\n{'='*60}")
    print(f"  Adapter: {adapter_dir}")
    print(f"{'='*60}")

    ac_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(ac_path):
        print(red(f"adapter_config.json not found at {ac_path}"))
        return

    ac = load_json(ac_path)
    base = ac.get("base_model_name_or_path", "NOT SET")
    print(f"  base_model_name_or_path: {base}")
    if not os.path.isdir(base):
        print(yellow(f"Base model path does not exist on this machine: {base}"))
        print(f"    This is fine if the merge was done on a different machine,")
        print(f"    but if you merge HERE it will fail to find the base weights.")
    else:
        print(green(f"Base model path exists"))

    print(f"  peft_type:       {ac.get('peft_type')}")
    print(f"  r (rank):        {ac.get('r')}")
    print(f"  lora_alpha:      {ac.get('lora_alpha')}")
    print(f"  target_modules:  {ac.get('target_modules')}")
    print(f"  task_type:       {ac.get('task_type')}")


def check_weights(merged_dir):
    print(f"\n{'='*60}")
    print(f"  Weight diagnostics: {merged_dir}")
    print(f"{'='*60}")

    try:
        from safetensors import safe_open
    except ImportError:
        print(yellow("safetensors not installed, skipping weight check"))
        return

    index_path = os.path.join(merged_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        print(yellow("No model.safetensors.index.json, skipping weight check"))
        return

    index = load_json(index_path)
    weight_map = index.get("weight_map", {})

    # Find embed_tokens and lm_head
    embed_file = weight_map.get("model.embed_tokens.weight")
    lm_head_file = weight_map.get("lm_head.weight")

    for name, shard_file in [("embed_tokens", embed_file), ("lm_head", lm_head_file)]:
        if shard_file is None:
            print(red(f"{name} not found in weight_map"))
            continue

        shard_path = os.path.join(merged_dir, shard_file)
        if not os.path.isfile(shard_path):
            print(red(f"Shard file not found: {shard_path}"))
            continue

        key = f"model.embed_tokens.weight" if name == "embed_tokens" else "lm_head.weight"
        with safe_open(shard_path, framework="pt") as f:
            tensor = f.get_tensor(key)

        print(f"\n  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

        if tensor.shape[0] != 152064:
            print(red(f"  {name} has {tensor.shape[0]} rows, expected 152064"))
        else:
            print(green(f"  {name} has correct 152064 rows"))

        # Check for dead rows (all-zero) in the padding region 151665..152063
        # These are the untrained "padding" rows — they should NOT be all-zero
        # in a properly merged model (the base model initializes them).
        padding_rows = tensor[151665:152064]
        n_zero = (padding_rows.abs().sum(dim=1) == 0).sum().item()
        n_total = padding_rows.shape[0]
        if n_zero > 0:
            print(red(f"  {n_zero}/{n_total} padding rows (151665..152063) are all-zero!"))
            print(f"    This means the base model's embeddings were NOT properly carried over.")
        else:
            print(green(f"  All {n_total} padding rows have non-zero values"))

        # Check vision token rows (151652..151656)
        vision_rows = tensor[151652:151657]
        n_zero_v = (vision_rows.abs().sum(dim=1) == 0).sum().item()
        if n_zero_v > 0:
            print(red(f"  {n_zero_v}/5 vision token embeddings are all-zero!"))
        else:
            print(green(f"  All 5 vision token embeddings have non-zero values"))

        # Norms of special-token vs normal-token embeddings
        import torch
        normal_norms = tensor[:151643].float().norm(dim=1)
        special_norms = tensor[151643:151665].float().norm(dim=1)
        padding_norms = tensor[151665:].float().norm(dim=1)
        print(f"  Norm stats (L2 per row):")
        print(f"    Regular tokens [0..151642]:      mean={normal_norms.mean():.4f}, std={normal_norms.std():.4f}")
        print(f"    Special tokens [151643..151664]: mean={special_norms.mean():.4f}, std={special_norms.std():.4f}")
        print(f"    Padding tokens [151665..152063]: mean={padding_norms.mean():.4f}, std={padding_norms.std():.4f}")

        if padding_norms.mean() < 0.01:
            print(red(f"  Padding token embeddings have near-zero norms — merge likely corrupted!"))


def main():
    parser = argparse.ArgumentParser(description="Diagnose PEFT LoRA merge for Qwen2.5-VL")
    parser.add_argument("--base", required=True, help="Path to original base model")
    parser.add_argument("--adapter", required=True, help="Path to SFT adapter output (contains adapter_config.json)")
    parser.add_argument("--merged", required=True, help="Path to merged model")
    args = parser.parse_args()

    # 1. Check base config
    base_cfg_path = os.path.join(args.base, "config.json")
    base_cfg = check_config("Base Model Config", base_cfg_path)

    # 2. Check adapter
    check_adapter(args.adapter)

    # 3. Check merged config (compare against base)
    merged_cfg_path = os.path.join(args.merged, "config.json")
    check_config("Merged Model Config", merged_cfg_path, reference_cfg=base_cfg)

    # 4. Check merged weights
    check_weights(args.merged)

    # 5. Quick comparison of base weights if available
    if os.path.isfile(os.path.join(args.base, "model.safetensors.index.json")):
        print(f"\n{'='*60}")
        print(f"  Base model weight reference")
        print(f"{'='*60}")
        check_weights(args.base)

    print(f"\n{'='*60}")
    print(f"  DONE — review any [FAIL] or [WARN] items above")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
