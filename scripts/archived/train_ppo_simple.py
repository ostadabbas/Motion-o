#!/usr/bin/env python3
"""
CLI tool for simple text-only PPO training on Dora Q&A dataset.

This:
- Loads a Hugging Face dataset (or subset) with 'transcript', 'question', 'answer'
- Runs PPO using `src.ppo_trainer_simple.run_simple_ppo`
- Saves the PPO-tuned model to an output directory

Usage example:
    python scripts/train_ppo_simple.py /path/to/dataset \\
        --output-dir ./outputs/ppo_dora \\
        --model-id Qwen/Qwen2.5-1.5B-Instruct \\
        --max-episodes 256
"""

import os
# Set CUDA_VISIBLE_DEVICES to use only V100 GPUs (0-3), excluding GTX 745 (GPU 4)
# This must be set before importing torch
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Set PyTorch CUDA allocator config to reduce memory fragmentation
# This must be set before importing torch
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_from_disk, Dataset, DatasetDict

from src.ppo_trainer_simple import SimplePPOConfig, run_simple_ppo


def main():
    parser = argparse.ArgumentParser(
        description="Simple text-only PPO training for Dora Q&A"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset directory or file saved via `datasets`",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/ppo",
        help="Directory to save PPO-tuned model",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model ID to PPO-tune",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for HF models",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=512,
        help="Maximum number of PPO episodes (dataset items) to train on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per PPO update",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for PPO",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
        help="Number of PPO epochs per batch",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype for model",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    print(f"Loading dataset from {dataset_path}...")

    if dataset_path.is_dir():
        dataset = load_from_disk(str(dataset_path))
    else:
        # Fall back to Dataset.load_from_disk path-style
        dataset = Dataset.load_from_disk(str(dataset_path))

    # If we have a DatasetDict, use the 'train' split
    if isinstance(dataset, DatasetDict):
        train_dataset = dataset.get("train")
        if train_dataset is None:
            raise ValueError("DatasetDict has no 'train' split.")
    else:
        train_dataset = dataset

    print(f"Dataset loaded with {len(train_dataset)} training examples.")

    cfg = SimplePPOConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        device=args.device,
        dtype=args.dtype,
    )

    print(
        f"Starting PPO training:\n"
        f"- model: {cfg.model_id}\n"
        f"- max_episodes: {args.max_episodes}\n"
        f"- batch_size: {cfg.batch_size}\n"
        f"- output_dir: {cfg.output_dir}"
    )

    run_simple_ppo(
        dataset=train_dataset,
        cfg=cfg,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()


