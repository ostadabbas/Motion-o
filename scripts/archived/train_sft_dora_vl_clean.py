import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset, load_from_disk
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl.trainer import SFTTrainer, SFTConfig

from scripts.dora_grpo_dataset import make_dora_grpo_data_module
from src.ppo_trainer_simple import extract_final_answer, string_f1
from typing import TypeVar
from collections.abc import Mapping
from accelerate import PartialState
from trl.data_utils import is_conversational_from_value, is_conversational

TListOrMapping = TypeVar("TListOrMapping", list, Mapping)
def remove_none_values(example: TListOrMapping) -> TListOrMapping:
    if isinstance(example, list):
        return [remove_none_values(value) if isinstance(value, (dict, list)) else value for value in example]
    elif isinstance(example, Mapping):
        return {
            key: remove_none_values(value) if isinstance(value, (dict, list)) else value
            for key, value in example.items()
            if value is not None
        }
    else:
        raise TypeError("Input must be a list or a dictionary.")

def get_dataset_column_names(dataset) -> list[str]:
    return list(next(iter(dataset)).keys()) if dataset.column_names is None else dataset.column_names

class MinimalVLSFTTrainer(SFTTrainer):
    """Minimal subclass - only fixes truncation issue for images."""
    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        packing: bool,
        formatting_func,
        dataset_name: str,
    ):
        # Tabular backends like Arrow/Parquet insert `None` for mismatched keys in nested structures. Clean them from
        # sampled data.
        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = get_dataset_column_names(dataset)
        print(column_names)
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                logger.warning(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            if not is_processed:
                # Convert the dataset to ChatML if needed
                first_example = next(iter(dataset))
                if is_conversational_from_value(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                    column_names = get_dataset_column_names(dataset)
                    dataset = dataset.map(
                        maybe_convert_to_chatml,
                        remove_columns="conversations" if "conversations" in column_names else None,
                        **map_kwargs,
                    )

                # Apply the chat template if needed
                first_example = next(iter(dataset))
                if not is_conversational(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                    def add_eos(example, eos_token):
                        if "text" in example and not example["text"].endswith(eos_token):  # language modeling case
                            example["text"] = example["text"] + eos_token
                        elif "completion" in example and not example["completion"].endswith(eos_token):
                            example["completion"] = example["completion"] + eos_token
                        return example

                    eos_token = processing_class.tokenizer.eos_token if self._is_vlm else processing_class.eos_token
                    dataset = dataset.map(
                        add_eos,
                        fn_kwargs={"eos_token": eos_token},
                        remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                        **map_kwargs,
                    )

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize_fn(example, processing_class, dataset_text_field, assistant_only_loss):
                    system_prompt = "You are a helpful visual reasoning assistant for kids.\n Think step by step and always give a final concise answer in the first sentence."
                    proc_text = f"{system_prompt}\n\nContext: {example['transcript']}\nQuestion: {example['question']}\nAnswer:"
                    example['prompt'] = proc_text
                    example["completion"] = example['answer']
                    if "prompt" in example:  # prompt-completion case
                        output = {}
                        if is_conversational(example):
                            if self._is_vlm:
                                prompt = prepare_multimodal_messages(example["prompt"], images=[])
                                completion = prepare_multimodal_messages(example["completion"], images=[])
                            else:
                                prompt = example["prompt"]
                                completion = example["completion"]
                            prompt_ids = processing_class.apply_chat_template(
                                prompt,
                                tools=example.get("tools"),
                                add_generation_prompt=True,
                                tokenize=True,
                                return_dict=False,
                                **example.get("chat_template_kwargs", {}),
                            )
                            # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                            # even for single examples, while for LLMs it returns lists of ints.
                            prompt_ids = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
                            prompt_completion_processed = processing_class.apply_chat_template(
                                prompt + completion,
                                tools=example.get("tools"),
                                tokenize=True,
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                **example.get("chat_template_kwargs", {}),
                            )
                            # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                            # even for single examples, while for LLMs it returns lists of ints.
                            prompt_completion_processed = {
                                k: v[0] if isinstance(v[0], list) else v
                                for k, v in prompt_completion_processed.items()
                            }
                            prompt_completion_ids = prompt_completion_processed["input_ids"]
                            if "assistant_masks" in prompt_completion_processed:
                                output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
                        else:
                            prompt_ids = processing_class(text=example["prompt"])["input_ids"]
                            prompt_completion_ids = processing_class(text=example["prompt"] + example["completion"])[
                                "input_ids"
                            ]
                            # Fix transformers inconsistency: for VLMs, processing_class returns lists of lists
                            # even for single examples, while for LLMs it returns lists of ints.
                            prompt_ids = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
                            prompt_completion_ids = (
                                prompt_completion_ids[0]
                                if isinstance(prompt_completion_ids[0], list)
                                else prompt_completion_ids
                            )

                        # Check if the tokenized prompt starts with the tokenized prompt+completion
                        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                            logger.warning(
                                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                                "token handling. Verify that the tokenizer is processing text consistently."
                            )

                        # Create completion mask
                        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                        output["input_ids"] = prompt_completion_ids
                        output["completion_mask"] = completion_mask

                    else:  # language modeling case
                        if is_conversational(example):
                            if self._is_vlm:
                                messages = prepare_multimodal_messages(example["messages"], images=[])
                            else:
                                messages = example["messages"]
                            processed = processing_class.apply_chat_template(
                                messages,
                                tools=example.get("tools"),
                                tokenize=True,
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                **example.get("chat_template_kwargs", {}),
                            )
                            # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                            # even for single examples, while for LLMs it returns lists of ints.
                            processed = {k: v[0] if isinstance(v[0], list) else v for k, v in processed.items()}
                            output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
                        else:
                            system_prompt = "You are a helpful visual reasoning assistant for kids.\n Think step by step and always give a final concise answer in the first sentence."
                            proc_text = f"{system_prompt}\n\nContext: {example['transcript']}\nQuestion: {example['question']}\nAnswer:"
                            output = {"input_ids": processing_class(text=proc_text)["input_ids"]}

                    if "assistant_masks" in output and 1 not in output["assistant_masks"]:
                        raise RuntimeError(
                            "You're using `assistant_only_loss=True`, but at least one example has no assistant "
                            "tokens. This usually means the tokenizer's chat template doesn't generate assistant "
                            "masks — it may be missing the `{% generation %}` keyword. Please check the template and "
                            "ensure it's correctly configured to support assistant masking."
                        )
                    print(output.keys())
                    return output

                dataset = dataset.map(
                    tokenize_fn,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "assistant_only_loss": args.assistant_only_loss,
                    },
                    **map_kwargs,
                )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                columns = ["input_ids"]
                if "completion_mask" in get_dataset_column_names(dataset):
                    columns.append("completion_mask")
                if "assistant_masks" in get_dataset_column_names(dataset):
                    columns.append("assistant_masks")

                dataset = dataset.select_columns(columns)

                # Shuffle the dataset before packing. When using wrapped packing, it's important to shuffle before
                # packing as well to avoid correlations between sequences packed together.
                if args.shuffle_dataset:
                    dataset = dataset.shuffle(seed=args.seed)

                # Packing adds new column "seq_lengths" needed for document aware FlashAttention
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
            elif args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)
            # For Liger kernel, ensure only the essential columns
            if args.use_liger_kernel:
                collator_expected_keys = {"input_ids", "seq_lengths", "completion_mask", "assistant_masks"}
                column_names = get_dataset_column_names(dataset)
                dataset = dataset.select_columns(collator_expected_keys.intersection(column_names))

        if args.shuffle_dataset:
            dataset = dataset.shuffle(seed=args.seed)

        return dataset

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "assistant", "images", "question", "answer"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean SFT training for Dora VLM")
    parser.add_argument("dataset_path", type=str, help="Path to dataset")
    parser.add_argument("--output-dir", type=str, default="./outputs/grpo_dora_vl", help="Output directory")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model ID")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # Load dataset
    print(f"[SFT] Loading dataset from {args.dataset_path}...")
    if Path(args.dataset_path).is_dir():
        dataset = load_from_disk(args.dataset_path)
    else:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=args.dataset_path)["train"]
    # Split the dataset into train and test sets
    split_dataset = dataset.train_test_split(test_size=0.001)

    # Access the splits
    # train_dataset = split_dataset["train"]
    dataset = split_dataset["test"]
    print(f"[SFT] Dataset loaded: {len(dataset)} examples")

    # Load processor
    print(f"[SFT] Loading processor from {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=None, trust_remote_code=True)
    processor.image_processor.do_resize = False
    print(f"[SFT] Processor loaded")

    # Load model
    print(f"[SFT] Loading model from {args.model_id}...")
    model_kwargs = {
        "torch_dtype": compute_dtype,
        "device_map": "auto",
        "cache_dir": None,
        "trust_remote_code": True,
    }
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    print(f"[SFT] Model loaded")

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,  # Directory to save the model
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=4,  # Batch size for evaluation
        gradient_accumulation_steps=8,  # Steps to accumulate gradients
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        max_length=None,
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        # Logging and evaluation
        logging_steps=10,  # Steps interval for logging
        eval_steps=10,  # Steps interval for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=20,  # Steps interval for saving
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
    )

    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = MinimalVLSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=processor,
    )

    trainer.train()