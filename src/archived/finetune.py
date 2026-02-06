"""
Fine-tuning implementation for QWEN-VL on Dora Q&A dataset.
Follows Hugging Face cookbook pattern using TRL SFTTrainer.
Supports two-stage training: transcript-based then frame-based.
"""
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
from datasets import Dataset
import numpy as np
from PIL import Image
from transformers import Trainer
import torch.nn.functional as F

def pad_to_max_size(tensor, max_h, max_w):
    h, w = tensor.shape[-2:]
    pad_h = max_h - h
    pad_w = max_w - w
    return F.pad(tensor, (0, pad_w, 0, pad_h))



@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning."""
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    output_dir: str = "./outputs"
    cache_dir: Optional[str] = None
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: Optional[int] = None
    max_seq_length: int = 2048
    max_new_tokens: int = 64
    use_4bit: bool = True
    stage: str = "transcript"  # "transcript" or "frame"
    device: str = "cuda"
    dtype: str = "auto"


def format_messages(item: Dict, stage: str, system_prompt: str) -> List[Dict]:
    """Format dataset item into messages format for chat template."""
    # Build user prompt
    user_prompt = (
        f"Context: {item.get('transcript', '')}\n"
        f"Question: {item['question']}\n"
        f"Answer:"
    )
    answer_text = item.get("answer", "")
    
    # Get images for frame stage
    images = None
    if stage == "frame":
        if "frames" not in item:
            raise ValueError(f"Item missing 'frames' for frame-based training")
        
        images = []
        for frame_data in item["frames"]:
            if isinstance(frame_data, dict):
                img_array = frame_data.get("image")
                if img_array is not None:
                    # Convert nested list to numpy array
                    img_array = np.array(img_array, dtype=np.uint8)
                    if len(img_array.shape) == 3:
                        img = Image.fromarray(img_array)
                        images.append(img)
            elif isinstance(frame_data, Image.Image):
                images.append(frame_data)
        
        if not images:
            raise ValueError(f"Item has no valid images")
    
    # Build messages following cookbook pattern
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    content = []
    if images:
        for img in images:
            content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_prompt})
    
    messages.append({"role": "user", "content": content})
    messages.append({"role": "assistant", "content": answer_text})
    
    return messages


def process_vision_info(messages: List[Dict]) -> tuple:
    """Extract images from messages, following cookbook pattern."""
    image_inputs = []
    text_inputs = []
    
    for message in messages:
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_inputs.append(item["image"])
        elif message.get("role") == "assistant":
            text_inputs.append(message["content"])
    
    return image_inputs if image_inputs else None, text_inputs


def setup_model_for_training(config: FinetuneConfig) -> tuple:
    """Setup model and processor for training, following cookbook pattern."""
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if config.dtype == "float16":
        torch_dtype = torch.float16
    elif config.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    
    quant_config = None
    if config.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        cache_dir=config.cache_dir
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        config.model_id,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=config.cache_dir,
    )
    model.enable_input_require_grads()
    
    # CRITICAL: Prepare model for k-bit training BEFORE applying LoRA
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    if config.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        print("Trainable parameters after LoRA setup:")
        model.print_trainable_parameters()
    
    model.train()
    return model, processor


class VisionDataCollator:
    """Custom data collator for vision-language models."""
    def __init__(self, processor: AutoProcessor):
        self.processor = processor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract messages from features
        messages_list = [f["messages"] for f in features]
        
        # Process each message with images
        batch_inputs = []
        prompt_lengths = []
        for messages in messages_list:
            # Extract images from messages
            images = []
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "image":
                            images.append(item["image"])
                            # print("loaded image", images[-1].size)
            # print(len(images))
            
            # Create prompt messages (without assistant response)
            prompt_messages = [m for m in messages if m.get("role") != "assistant"]
            prompt_messages.append({"role": "assistant", "content": ""})  # Add empty assistant for template
            
            # Apply chat template for prompt
            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # Apply chat template for full conversation
            full_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            
            # Process full conversation first
            inputs = self.processor(
                text=[full_text],
                images=images if images else None,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            
            # Calculate prompt length by finding where assistant response starts
            # Method: tokenize answer separately and find its position in full sequence
            assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
            if assistant_msg and assistant_msg.get("content"):
                answer_text = assistant_msg["content"]
                # Tokenize answer (without images, just text)
                answer_inputs = self.processor.tokenizer(
                    answer_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=False,
                )
                answer_token_ids = answer_inputs["input_ids"].squeeze(0).tolist()
                
                # Find answer tokens in full sequence (search from end to handle duplicates)
                full_token_ids = inputs["input_ids"].squeeze(0).tolist()
                prompt_len = len(full_token_ids)
                
                # Search for answer tokens at the end of the sequence
                if len(answer_token_ids) > 0 and len(full_token_ids) >= len(answer_token_ids):
                    # Check if answer appears at the end
                    if full_token_ids[-len(answer_token_ids):] == answer_token_ids:
                        prompt_len = len(full_token_ids) - len(answer_token_ids)
                    else:
                        # Fallback: search for answer tokens anywhere
                        for i in range(len(full_token_ids) - len(answer_token_ids) + 1):
                            if full_token_ids[i:i+len(answer_token_ids)] == answer_token_ids:
                                prompt_len = i
                                break
            else:
                # Fallback: process prompt separately
                print("seperATE")
                prompt_inputs = self.processor(
                    text=[prompt_text],
                    images=images if images else None,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    do_resize=True,
                )
                prompt_len = prompt_inputs["input_ids"].shape[1]
            
            prompt_lengths.append(prompt_len)
            batch_inputs.append(inputs)
        
        # Collate batch
        # Get max length
        max_len = max(inp["input_ids"].shape[1] for inp in batch_inputs)
        
        # Pad all inputs
        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        for inp in batch_inputs:
            seq_len = inp["input_ids"].shape[1]
            pad_len = max_len - seq_len
            
            # Pad input_ids and attention_mask
            input_ids = inp["input_ids"].squeeze(0)
            attention_mask = inp["attention_mask"].squeeze(0)
            
            if pad_len > 0:
                pad_token_id = self.processor.tokenizer.pad_token_id
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            
            # Handle pixel_values
            if "pixel_values" in inp and inp["pixel_values"] is not None:
                pixel_values_list.append(inp["pixel_values"].squeeze(0))
            
            # Handle image_grid_thw - processor returns tensor [num_images, 3]
            if "image_grid_thw" in inp and inp["image_grid_thw"] is not None:
                # Processor returns tensor of shape [num_images, 3] where each row is [t, h, w]
                grid_thw = inp["image_grid_thw"]
                
                # Ensure it's 2D tensor [num_images, 3]
                if grid_thw.dim() == 1:
                    grid_thw = grid_thw.unsqueeze(0)  # [3] -> [1, 3]
                
                # Store as tensor - will be passed as list to model
                image_grid_thw_list.append(grid_thw)
        
        result = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
        }
        
        if pixel_values_list:
            result["pixel_values"] = torch.stack(pixel_values_list)
        
        if image_grid_thw_list:
            # QWEN-VL's rot_pos_emb expects a tensor [num_images, 3], not a list
            # Since batch_size=1 is standard for vision models with variable image counts,
            # we pass the tensor directly. For batch_size > 1, we'd need proper batching logic.
            batch_size = len(messages_list)
            if batch_size == 1:
                # Single batch item - pass tensor directly (no list wrapper needed)
                result["image_grid_thw"] = image_grid_thw_list[0]
            else:
                # Multiple batch items - pass as list (forward wrapper will handle it)
                # Note: This case requires proper handling in the model forward wrapper
                pass
            result["image_grid_thw"] = image_grid_thw_list
        
        # Create labels: -100 for prompt tokens, actual ids for answer tokens
        labels = result["input_ids"].clone()
        labels.fill_(-100)
        
        for i, prompt_len in enumerate(prompt_lengths):
            seq_len = result["input_ids"][i].shape[0]
            if prompt_len < seq_len:
                labels[i, prompt_len:] = result["input_ids"][i, prompt_len:]
        
        result["labels"] = labels
        
        return result


class MessagesDataset:
    """Dataset wrapper that formats messages on-the-fly for SFTTrainer."""
    def __init__(self, dataset: Dataset, stage: str, system_prompt: str):
        self.dataset = dataset
        self.stage = stage
        self.system_prompt = system_prompt
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        messages = format_messages(item, stage=self.stage, system_prompt=self.system_prompt)
        return {"messages": messages}


def train_stage1_transcript(dataset: Dataset,
                           config: FinetuneConfig,
                           val_dataset: Optional[Dataset] = None) -> tuple:
    """Stage 1: Train on transcripts only (no frames)."""
    model, processor = setup_model_for_training(config)
    
    system_prompt = (
        "You are a helpful visual reasoning assistant for kids. "
        "Think step by step internally. Then output the best final answer."
    )
    
    # Create dataset wrapper that formats messages on-the-fly
    train_dataset = MessagesDataset(dataset, stage="transcript", system_prompt=system_prompt)
    val_dataset_formatted = None
    if val_dataset:
        val_dataset_formatted = MessagesDataset(val_dataset, stage="transcript", system_prompt=system_prompt)
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps" if val_dataset_formatted else "no",
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset_formatted else False,
        report_to="tensorboard" if config.output_dir else None,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    
    # Use custom data collator for vision models
    data_collator = VisionDataCollator(processor)
    
    # No forward wrapper needed for stage 1 (no images)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset_formatted,
        data_collator=data_collator,
    )
    
    trainer.train(resume_from_checkpoint=True,)
    trainer.save_model()
    return model, processor


def train_stage2_frames(dataset: Dataset,
                       base_model_path: Optional[str] = None,
                       config: Optional[FinetuneConfig] = None,
                       val_dataset: Optional[Dataset] = None) -> tuple:
    """Stage 2: Fine-tune on frames (after stage 1 or from scratch)."""
    import torch
    if config is None:
        config = FinetuneConfig()
    
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        cache_dir=config.cache_dir
    )
    
    system_prompt = (
        "You are a helpful visual reasoning assistant for kids. "
        "Think step by step internally. Then output the best final answer."
    )
    
    if base_model_path:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if config.dtype == "float16":
            torch_dtype = torch.float16
        elif config.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        
        quant_config = None
        if config.use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        base_model = AutoModelForVision2Seq.from_pretrained(
            config.model_id,
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=config.cache_dir,
        )
        
        # CRITICAL: Prepare model for k-bit training BEFORE loading PEFT
        if config.use_4bit:
            base_model = prepare_model_for_kbit_training(base_model)
        
        # Load PEFT adapters
        model = PeftModel.from_pretrained(base_model, base_model_path)
        model.enable_input_require_grads()
        # CRITICAL: Set model to training mode
        model.train()

        
        # Ensure adapter parameters are trainable (they should be by default, but double-check)
        for name, param in model.named_parameters():
            if "lora" in name.lower() or "adapter" in name.lower():
                param.requires_grad = True
        
        print("Trainable parameters after loading PEFT model:")
        model.print_trainable_parameters()
    else:
        model, _ = setup_model_for_training(config)
    
    # Create dataset wrapper that formats messages on-the-fly
    train_dataset = MessagesDataset(dataset, stage="frame", system_prompt=system_prompt)
    val_dataset_formatted = None
    if val_dataset:
        val_dataset_formatted = MessagesDataset(val_dataset, stage="frame", system_prompt=system_prompt)
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps" if val_dataset_formatted else "no",
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset_formatted else False,
        report_to="tensorboard" if config.output_dir else None,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        per_device_eval_batch_size=1,
    )
    
    # Use custom data collator for vision models
    data_collator = VisionDataCollator(processor)
    
    # Wrap model forward to handle image_grid_thw correctly
    # QWEN-VL's rot_pos_emb expects a tensor [num_images, 3], not a list
    # When batch_size=1, we should pass the single tensor directly
    original_forward = model.forward
    def batched_forward(*args, **kwargs):
        if "image_grid_thw" in kwargs and isinstance(kwargs["image_grid_thw"], list):
            image_grid_thw_list = kwargs["image_grid_thw"]
            # Get batch size from input_ids
            if "input_ids" in kwargs:
                batch_size = kwargs["input_ids"].shape[0]
            elif args and len(args) > 0 and hasattr(args[0], 'shape'):
                batch_size = args[0].shape[0]
            else:
                batch_size = len(image_grid_thw_list)
            
            # Extract the correct tensor for this batch
            # For batch_size=1, use the first (and only) tensor
            # For batch_size > 1, we'd need to handle it differently, but batch_size should be 1
            if len(image_grid_thw_list) > 0:
                if batch_size == 1:
                    # Single batch item - pass the tensor directly
                    kwargs["image_grid_thw"] = image_grid_thw_list[0]
                else:
                    # Multiple batch items - this shouldn't happen with batch_size=1
                    # But if it does, use the first one (incorrect but won't crash)
                    kwargs["image_grid_thw"] = image_grid_thw_list[0]
            else:
                kwargs.pop("image_grid_thw", None)
        
        return original_forward(*args, **kwargs)
    
    model.forward = batched_forward
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset_formatted,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model()
    return model, processor
