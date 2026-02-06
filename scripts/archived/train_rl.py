#!/usr/bin/env python3
"""
CLI tool for RL training using PPO.
Note: This is a simplified version. Full PPO implementation would require
more complex setup with TRL's PPOTrainer.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import VLM
from src.rl_trainer import RLAgent
from datasets import load_from_disk, Dataset, DatasetDict
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="RL training for QWEN-VL using PPO"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset directory or file"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Base model ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/rl",
        help="Output directory"
    )
    parser.add_argument(
        "--use-ppo",
        action="store_true",
        help="Use PPO training (otherwise just logging mode)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for PPO"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.41e-5,
        help="Learning rate for PPO"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
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
    try:
        dataset = load_from_disk(args.dataset_path) if Path(args.dataset_path).is_dir() else Dataset.load_from_disk(args.dataset_path)
    except:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=args.dataset_path)["train"]
    
    # Handle DatasetDict
    if isinstance(dataset, DatasetDict):
        train_dataset = dataset.get("train", dataset.get("train", None))
    else:
        train_dataset = dataset
    
    if train_dataset is None:
        raise ValueError("No training dataset found")
    
    print(f"Dataset loaded: {len(train_dataset)} examples")
    
    # Initialize model
    print(f"Loading model {args.model_id}...")
    vlm = VLM(
        model_id=args.model_id or config_dict.get("model_id", "Qwen/Qwen2-VL-2B-Instruct"),
        device=args.device or config_dict.get("device", "cuda"),
        dtype=config_dict.get("dtype", "auto"),
        load_4bit=config_dict.get("use_4bit", True),
        cache_dir=args.cache_dir or config_dict.get("cache_dir", None)
    )
    
    # Initialize RL agent
    agent = RLAgent(vlm, use_ppo=args.use_ppo)
    
    # Training loop
    print(f"\nStarting RL training ({'PPO' if args.use_ppo else 'logging'} mode)...")
    print(f"Iterations: {args.num_iterations}")
    
    for iteration in range(args.num_iterations):
        print(f"\nIteration {iteration + 1}/{args.num_iterations}")
        
        # Sample batch from dataset
        batch_size = min(args.batch_size, len(train_dataset))
        indices = list(range(iteration * batch_size, min((iteration + 1) * batch_size, len(train_dataset))))
        
        if not indices:
            print("Dataset exhausted. Stopping.")
            break
        
        for idx in indices:
            item = train_dataset[idx]
            
            # Extract frames if available
            frames = []
            if "frames" in item:
                from PIL import Image
                import numpy as np
                for frame_data in item["frames"]:
                    if isinstance(frame_data, dict):
                        img_array = frame_data.get("image")
                        if img_array is not None:
                            if isinstance(img_array, np.ndarray):
                                img = Image.fromarray(img_array.astype(np.uint8))
                            else:
                                img = Image.fromarray(np.array(img_array).astype(np.uint8))
                            frames.append((img, frame_data.get("timestamp", 0.0)))
            
            # Generate prediction
            system_prompt = "You are a helpful visual reasoning assistant for kids."
            user_prompt = (
                f"Context: {item.get('transcript', '')}\n"
                f"Question: {item['question']}\n"
                f"Answer:"
            )
            
            if frames:
                prediction = vlm.generate(
                    frames=frames,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=32,
                    temperature=0.3,
                    top_p=0.9
                )
            else:
                # Text-only mode
                prediction = vlm.generate(
                    frames=[],  # Empty frames
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=32,
                    temperature=0.3,
                    top_p=0.9
                )
            
            # Compute reward
            reward = agent.compute_reward(prediction, item.get("answer", ""))
            
            # Observe and learn
            agent.observe_and_learn(
                frames=frames if frames else [],
                context_prompt=item.get("transcript", ""),
                transcript=item.get("transcript", ""),
                question=item["question"],
                prediction=prediction,
                answer=item.get("answer", ""),
                reward=reward
            )
            
            print(f"  Q: {item['question'][:50]}...")
            print(f"  A: {item.get('answer', '')[:50]}...")
            print(f"  P: {prediction[:50]}...")
            print(f"  R: {reward:.2f}")
    
    # Export logs
    print("\nExporting logs...")
    logs = agent.export_logs()
    
    # Save logs
    import json
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logs_path = output_path / "rl_logs.json"
    with open(logs_path, 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"Logs saved to {logs_path}")
    
    print("\nRL training complete!")


if __name__ == "__main__":
    main()

