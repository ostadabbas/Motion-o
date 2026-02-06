#!/usr/bin/env python3
"""
CLI tool for fine-tuning QWEN-VL on Dora Q&A dataset.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.finetune import (
    FinetuneConfig,
    train_stage1_transcript,
    train_stage2_frames
)
from datasets import load_from_disk, Dataset, DatasetDict
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune QWEN-VL on Dora Q&A dataset"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset directory or file"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "both"],
        default="both",
        help="Training stage: 1 (transcript), 2 (frames), or both"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Base model ID"
    )
    parser.add_argument(
        "--stage1-model",
        type=str,
        default=None,
        help="Path to stage 1 model (for stage 2 training)"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        "--no-lora",
        action="store_false",
        dest="use_lora",
        help="Don't use LoRA"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config_dict = {}
    if args.config:
        config_dict = load_config(args.config)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path) if Path(args.dataset_path).is_dir() else Dataset.load_from_disk(args.dataset_path)
    
    # Handle DatasetDict
    split_dataset = dataset.train_test_split(test_size=0.01)
    # Access the splits
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    if train_dataset is None:
        raise ValueError("No training dataset found")
    
    # Create config
    config = FinetuneConfig(
        model_id=args.model_id or config_dict.get("model_id", "Qwen/Qwen2-VL-2B-Instruct"),
        output_dir=args.output_dir,
        cache_dir=args.cache_dir or config_dict.get("cache_dir", None),
        use_lora=args.use_lora if args.use_lora is not None else config_dict.get("use_lora", True),
        lora_r=args.lora_r or config_dict.get("lora_r", 16),
        lora_alpha=args.lora_alpha or config_dict.get("lora_alpha", 32),
        lora_dropout=config_dict.get("lora_dropout", 0.1),
        learning_rate=args.learning_rate or config_dict.get("learning_rate", 2e-4),
        batch_size=args.batch_size or config_dict.get("batch_size", 1),
        gradient_accumulation_steps=args.gradient_accumulation_steps or config_dict.get("gradient_accumulation_steps", 4),
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps or config_dict.get("warmup_steps", 100),
        logging_steps=config_dict.get("logging_steps", 10),
        save_steps=config_dict.get("save_steps", 500),
        eval_steps=config_dict.get("eval_steps", 500),
        max_seq_length=config_dict.get("max_length", 2048),
        max_new_tokens=config_dict.get("max_new_tokens", 64),
        use_4bit=config_dict.get("use_4bit", False),
        device=config_dict.get("device", "cuda"),
        dtype=config_dict.get("dtype", "auto"),
    )
    
    # Train stage 1 (transcript-based)
    if args.stage in ["1", "both"]:
        print("\n" + "="*50)
        print("Stage 1: Training on transcripts (text only)")
        print("="*50)
        stage1_output = f"{args.output_dir}/stage1"
        config.output_dir = stage1_output
        config.stage = "transcript"
        
        model, processor = train_stage1_transcript(
            train_dataset,
            config,
            val_dataset=val_dataset
        )
        print(f"Stage 1 complete. Model saved to {stage1_output}")
        
        # Update stage1_model path for stage 2
        if args.stage == "both":
            args.stage1_model = stage1_output
    
    # Train stage 2 (frame-based)
    if args.stage in ["2", "both"]:
        print("\n" + "="*50)
        print("Stage 2: Training on frames")
        print("="*50)
        stage2_output = f"{args.output_dir}/stage2"
        config.output_dir = stage2_output
        config.stage = "frame"
        
        model, processor = train_stage2_frames(
            train_dataset,
            base_model_path=args.stage1_model,
            config=config,
            val_dataset=val_dataset
        )
        print(f"Stage 2 complete. Model saved to {stage2_output}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

