import os
os.environ["WANDB_MODE"] = "offline" 
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"

# os.environ["MASTER_PORT"] = "29501"
# os.environ["RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# master_port = int(os.getenv('MASTER_PORT', '29500'))
# print(f"Using MASTER_PORT: {master_port}")

from dataclasses import dataclass, field
from typing import Optional
from grpo_trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from data_loader import get_data
from reward_func import (
    ans_acc_reward, 
    ans_tiou_reward, 
    ans_viou_reward, 
    thk_temporal_point_reward, 
    thk_temporal_segment_reward, 
    thk_spatial_reward, 
    format_reward
)
from motion_reward import motion_trajectory_reward

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["ans_acc", "ans_tiou", "ans_viou", "thk_temporal_point", "thk_temporal_segment", "thk_spatial", "motion_trajectory", "format"],
        metadata={"help": "List of reward functions including motion_trajectory for trajectory-level motion reasoning"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "Sampling temperature for generation (1.0 for more exploration, 0.7 for safer)"},
    )


# reward_funcs_registry = {
#     "accuracy": accuracy_reward,
#     "format": format_reward,
#     "temporal": temporal_reward,
#     "spatial": spatial_reward,
# }

reward_funcs_registry = {
    "ans_acc": ans_acc_reward,
    "ans_tiou": ans_tiou_reward,
    "ans_viou": ans_viou_reward,
    "thk_temporal_point": thk_temporal_point_reward,
    "thk_temporal_segment": thk_temporal_segment_reward,
    "thk_spatial": thk_spatial_reward,
    "motion_trajectory": motion_trajectory_reward,  # ⭐ Motion-aware trajectory reward
    "format": format_reward
}


# ans_acc_reward
# ans_tiou_reward
# ans_viou_reward
# thk_temporal_point_reward
# thk_temporal_segment_reward
# thk_spatial_reward
# format_reward


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = get_data(script_args)
    
    # Quick test mode - limit dataset size for faster debugging
    import os
    if os.environ.get("QUICK_TEST", "false").lower() == "true":
        max_samples = int(os.environ.get("MAX_SAMPLES", "10"))
        print(f"\n{'='*50}")
        print(f"🚀 QUICK TEST MODE: Using only {max_samples} samples")
        print(f"{'='*50}\n")
        dataset[script_args.dataset_train_split] = dataset[script_args.dataset_train_split].select(
            range(min(max_samples, len(dataset[script_args.dataset_train_split])))
        )
        training_args.num_train_epochs = 1
        training_args.save_steps = max(1, max_samples // 2)
    
    trainer_cls = Qwen2VLGRPOTrainer  # if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)
    print("training_args:", training_args)
    print("script_args:", script_args)
    print("model_args:", model_args)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        temperature=script_args.temperature,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
