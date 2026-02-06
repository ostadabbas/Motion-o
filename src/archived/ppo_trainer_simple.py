"""
Simple text-only PPO training for Dora Q&A, built from scratch.

We:
- Build prompts from transcript + question
- Let the model generate "thinking + final answer"
- Compute a scalar reward from the final answer vs. gold answer
- Optimize with TRL's PPOTrainer

This intentionally ignores existing RL code and uses only text (no frames) for clarity.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
# Use the NEW Hugging Face TRL API (from trl.experimental.ppo)
from trl.experimental.ppo import PPOConfig, PPOTrainer
from trl.models import AutoModelForCausalLMWithValueHead
import Levenshtein

def normalized_edit_similarity(a, b):
    dist = Levenshtein.distance(a, b)
    max_len = max(len(a), len(b))
    return 1 - dist / max_len

@dataclass
class SimplePPOConfig:
    """Configuration for simple text-only PPO."""

    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "./outputs/ppo"
    cache_dir: Optional[str] = None

    # PPO hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 8            # batch per PPO update
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    max_prompt_length: int = 1024
    max_response_length: int = 128

    # KL control
    init_kl_coef: float = 0.01

    # Device / dtype
    device: str = "cuda"
    dtype: str = "auto"  # "auto", "bfloat16", "float16", "float32"


def build_prompt(item: Dict[str, Any], target_prompt_length: int = 1337) -> str:
    """
    Create the text-only prompt from a Dora dataset item.
    
    Truncates transcript from the BEGINNING to keep recent context close to the question,
    targeting a specific prompt length (default 1337 chars, based on best-performing path questions).
    
    Format: "...recent context near question" (removes old context from beginning, keeps end)
    
    Args:
        item: Dataset item with 'transcript' and 'question' fields
        target_prompt_length: Target total prompt length in characters (default 1337)
    
    Returns:
        Formatted prompt string with truncated transcript if needed
    """
    transcript = item.get("transcript", "")
    question = item.get("question", "")
    
    # Calculate fixed parts of the prompt (without transcript)
    system_msg = "You are a helpful visual reasoning assistant for kids.\n"
    instruction = "Think step by step, then give a final concise answer.\n\n"
    context_prefix = "Context: "
    question_prefix = "\nQuestion: "
    answer_suffix = "\nAnswer (think step by step, then say 'Final answer: <answer>'):\n"
    
    # Calculate fixed length (excluding transcript)
    fixed_length = (
        len(system_msg) + 
        len(instruction) + 
        len(context_prefix) + 
        len(question_prefix) + 
        len(question) + 
        len(answer_suffix)
    )
    
    # Calculate max transcript length to achieve target prompt length
    max_transcript_length = target_prompt_length - fixed_length
    
    # Truncate transcript from the BEGINNING if it's too long
    # This keeps the END (context closest to question) and removes the beginning
    # Format: "...recent context near question" (not "old context...")
    if len(transcript) > max_transcript_length and max_transcript_length > 0:
        # Take the last N characters (removes from beginning, keeps end)
        transcript = transcript[-max_transcript_length:]
        # Add ellipsis at the beginning to indicate truncation
        if max_transcript_length > 10:
            transcript = "..." + transcript[3:]
    
    prompt = (
        f"{system_msg}"
        f"{instruction}"
        f"{context_prefix}{transcript}\n"
        f"{question_prefix}{question}\n"
        f"{answer_suffix}"
    )
    return prompt


def extract_final_answer(text: str) -> str:
    """
    Extract the final answer string from a generated response.

    Heuristics:
    - If 'Final answer:' appears, take everything after it.
    - Otherwise take the last line.
    """
    marker = "Final answer:"
    if marker in text:
        return text.split(marker, 1)[1].strip()

    # Fallback: last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return text.strip()
    return lines[-1]


def string_f1(pred: str, gold: str) -> float:
    return 0.3 * string_f1_simple(pred, gold) + 0.7 * edit_f1(pred, gold)

def string_f1_simple(pred: str, gold: str) -> float:
    """
    Very simple token-level F1 between prediction and gold answer.
    """
    import re

    def normalize(s: str) -> List[str]:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return [t for t in s.split() if t]
    if not isinstance(pred, list):
        pred = normalize(pred)
        gold = normalize(gold)
        if not gold:
            return 0.0
        if not pred:
            return 0.0

    pred_set = set(pred)
    gold_set = set(gold)

    inter = pred_set & gold_set
    if not inter:
        return 0.0

    precision = len(inter) / len(pred_set)
    recall = len(inter) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def edit_f1(pred: str, gold: str) -> float:
    import re

    def normalize(s: str) -> List[str]:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return [t for t in s.split() if t]

    if not isinstance(pred, list):
        pred = normalize(pred)
        gold = normalize(gold)
        if not gold:
            return 0.0
        if not pred:
            return 0.0
    return normalized_edit_similarity(pred, gold)


def compute_reward_from_item(response: str, item: Dict[str, Any]) -> float:
    """Map a model response + dataset item to a scalar reward."""
    gold_answer = item.get("answer", "") or ""
    final_answer = extract_final_answer(response)
    base_score = string_f1(final_answer, gold_answer)
    # Optionally rescale to something like [-1, 1]; keep it [0, 1] for now.
    return float(base_score)


def setup_ppo_components(cfg: SimplePPOConfig, train_dataset: Dataset, total_episodes: int = None):
    """Load tokenizer, reference policy, and PPO policy."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        cache_dir=cfg.cache_dir,
        trust_remote_code=True,
    )
    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        # For many chat models, eos_token is a reasonable pad
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for decoder-only models (required for generation)
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if cfg.dtype == "float16":
        torch_dtype = torch.float16
    elif cfg.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif cfg.dtype == "float32":
        torch_dtype = torch.float32

    # Use 4-bit quantization to drastically reduce memory usage
    quantization_config = None
    if cfg.device == "cuda" and torch.cuda.is_available():
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("[Memory] Using 4-bit quantization to reduce memory")
        except Exception as e:
            print(f"[Memory] Could not enable 4-bit quantization: {e}")
            quantization_config = None

    # Load ref_model with 4-bit quantization (reduces memory by ~75%)
    ref_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=None,
        cache_dir=cfg.cache_dir,
        trust_remote_code=True,
    )
    if quantization_config:
        ref_kwargs["quantization_config"] = quantization_config
    
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **ref_kwargs)
    if cfg.device == "cuda" and torch.cuda.is_available():
        ref_model = ref_model.to("cuda:0")
        print(f"[Memory] ref_model -> cuda:0 (4-bit quantized, ~75% less memory)")

    # Load policy_model with 4-bit quantization
    policy_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=None,
        cache_dir=cfg.cache_dir,
        trust_remote_code=True,
    )
    if quantization_config:
        policy_kwargs["quantization_config"] = quantization_config
    
    policy_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **policy_kwargs)
    if cfg.device == "cuda" and torch.cuda.is_available():
        policy_model = policy_model.to("cuda:0")
        # Enable gradient checkpointing to save memory
        if hasattr(policy_model, "gradient_checkpointing_enable"):
            policy_model.gradient_checkpointing_enable()
        print(f"[GPU] policy_model -> cuda:0 (4-bit quantized, gradient checkpointing enabled)")
    
    # Load value and reward models (sequence classification) - no quantization needed (smaller)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        num_labels=1,
        torch_dtype=torch_dtype,
        device_map=None,
        cache_dir=cfg.cache_dir,
        trust_remote_code=True,
    )
    if cfg.device == "cuda" and torch.cuda.is_available():
        value_model = value_model.to("cuda:0")
        print(f"[GPU] value_model -> cuda:0")
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        num_labels=1,
        torch_dtype=torch_dtype,
        device_map=None,
        cache_dir=cfg.cache_dir,
        trust_remote_code=True,
    )
    if cfg.device == "cuda" and torch.cuda.is_available():
        reward_model = reward_model.to("cuda:0")
        print(f"[GPU] reward_model -> cuda:0")

    # Adjust batch size if dataset is too small
    effective_batch_size = min(cfg.batch_size, len(train_dataset))
    
    ppo_config = PPOConfig(
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=effective_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_ppo_epochs=cfg.ppo_epochs,
        response_length=cfg.max_response_length,
        report_to=None,
        kl_coef=cfg.init_kl_coef,
        output_dir=cfg.output_dir,
        bf16=torch.cuda.is_available(),
        total_episodes=total_episodes if total_episodes is not None else len(train_dataset),
    )

    # Use NEW Hugging Face TRL API (from trl.experimental.ppo)
    # Pass train_dataset as eval_dataset to avoid None error
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # Use train_dataset for eval to avoid None error
        value_model=value_model,
    )

    # Patch PolicyAndValueWrapper to add gradient_checkpointing_disable method
    # This is needed because TRL's unwrap_model_for_generation expects this method
    if hasattr(ppo_trainer.model, 'policy'):
        import types
        
        def gradient_checkpointing_disable(wrapper_self):
            """Disable gradient checkpointing on the underlying policy model."""
            if hasattr(wrapper_self.policy, 'gradient_checkpointing_disable'):
                wrapper_self.policy.gradient_checkpointing_disable()
            wrapper_self.is_gradient_checkpointing = False
        
        def gradient_checkpointing_enable(wrapper_self):
            """Enable gradient checkpointing on the underlying policy model."""
            if hasattr(wrapper_self.policy, 'gradient_checkpointing_enable'):
                wrapper_self.policy.gradient_checkpointing_enable()
            wrapper_self.is_gradient_checkpointing = True
        
        # Add methods to the wrapper instance
        ppo_trainer.model.gradient_checkpointing_disable = types.MethodType(gradient_checkpointing_disable, ppo_trainer.model)
        ppo_trainer.model.gradient_checkpointing_enable = types.MethodType(gradient_checkpointing_enable, ppo_trainer.model)

    # All models stay on accelerator.device (cuda:0) - required for proper device handling
    # 4-bit quantization and gradient checkpointing are enabled to reduce memory usage

    return tokenizer, ppo_trainer


def run_simple_ppo(
    dataset: Dataset,
    cfg: SimplePPOConfig,
    max_episodes: int = 1000,
) -> None:
    """
    Run a basic PPO loop on a Dora dataset (text only).
    
    Uses the NEW Hugging Face TRL API from trl.experimental.ppo.
    Follows the pattern from trl/examples/scripts/ppo/ppo.py
    """
    # Prepare dataset in format expected by PPOTrainer
    # Need 'input_ids' field (following the example)
    from transformers import AutoTokenizer
    temp_tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        cache_dir=cfg.cache_dir,
        trust_remote_code=True,
    )
    if temp_tokenizer.pad_token is None:
        temp_tokenizer.pad_token = temp_tokenizer.eos_token
    temp_tokenizer.padding_side = "left"
    
    # Prepare dataset (limit to max_episodes)
    num_items = min(len(dataset), max_episodes)
    
    def tokenize_item(item):
        """Tokenize a single item, following the PPO example pattern."""
        prompt = build_prompt(item)
        outputs = temp_tokenizer(
            prompt,
            truncation=True,
            max_length=cfg.max_prompt_length,
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}
    
    # Prepare dataset - following the example pattern
    from datasets import Dataset as HFDataset
    prepared_data = [tokenize_item(dataset[i]) for i in range(num_items)]
    train_dataset = HFDataset.from_list(prepared_data)
    
    # DEBUG: Print dataset preparation info
    print(f"\n[DEBUG] Dataset preparation:")
    print(f"  - Number of items: {num_items}")
    for i in range(num_items):
        item = dataset[i]
        prompt = build_prompt(item)
        input_ids_len = len(prepared_data[i]["input_ids"])
        print(f"  - Example {i+1}:")
        print(f"    Question: {item.get('question', '')[:60]}...")
        print(f"    Answer: {item.get('answer', '')}")
        print(f"    Prompt length: {len(prompt)} chars")
        print(f"    Tokenized length: {input_ids_len} tokens")
        if input_ids_len >= cfg.max_prompt_length:
            print(f"    [WARNING] Prompt may be truncated! (max={cfg.max_prompt_length})")
    
    # Store ground truth answers for reward computation
    # CRITICAL FIX: Use input_ids prefix as key since prompts are truncated
    # The question gets cut off, so we can't extract it from the truncated prompt
    # Instead, use the first 200 tokens of input_ids as a unique identifier
    answer_map = {}  # Maps (input_ids_prefix) -> answer
    input_ids_to_question = {}  # For debugging
    for i in range(num_items):
        item = dataset[i]
        input_ids = tuple(prepared_data[i]["input_ids"])
        question = item.get("question", "").strip()
        answer = item.get("answer", "")
        
        # Use first 200 tokens as key (should be unique even after truncation)
        # This works because truncation happens at the END, so the beginning is preserved
        input_ids_prefix = input_ids[:200]
        answer_map[input_ids_prefix] = answer
        input_ids_to_question[input_ids_prefix] = question
        
        print(f"[DEBUG] Added to answer_map: Example {i+1}")
        print(f"    Question: '{question}'")
        print(f"    Answer: '{answer}'")
        print(f"    Input IDs length: {len(input_ids)} tokens")
        print(f"    Prefix length: {len(input_ids_prefix)} tokens")
    
    print(f"[DEBUG] Answer map contains {len(answer_map)} entries (should be {num_items})")
    
    # Verify all questions are unique
    questions = [dataset[i].get("question", "").strip() for i in range(num_items)]
    if len(set(questions)) < len(questions):
        print(f"[WARNING] Some questions are duplicates!")
        from collections import Counter
        question_counts = Counter(questions)
        for q, count in question_counts.items():
            if count > 1:
                print(f"  Question '{q[:50]}...' appears {count} times")
    
    # Patch get_reward BEFORE creating trainer to ensure it's used
    # CRITICAL: Must patch both the utils module AND the PPO trainer's imported reference
    import trl.trainer.utils
    import sys
    
    original_get_reward = trl.trainer.utils.get_reward
    
    # Track reward statistics for debugging (must be defined before patched function)
    reward_stats = {"total_calls": 0, "found_answers": 0, "missing_answers": 0, "rewards_by_example": {}}
    
    def patched_get_reward(model, query_responses, pad_token_id, context_length):
        """Compute rewards from ground truth answers."""
        # DEBUG: Always log that we're being called
        reward_stats["total_calls"] += 1
        if reward_stats["total_calls"] <= 5:
            print(f"\n[DEBUG REWARD] patched_get_reward called! (call #{reward_stats['total_calls']})")
            print(f"  Model type: {type(model)}")
            print(f"  Has _answer_map: {hasattr(model, '_answer_map')}")
            print(f"  Has _tokenizer: {hasattr(model, '_tokenizer')}")
        
        # Check if this is our reward model with answer map
        if hasattr(model, '_answer_map') and hasattr(model, '_tokenizer'):
            batch_size = query_responses.shape[0]
            scores = []
            sequence_lengths_list = []
            
            for i in range(batch_size):
                query_response = query_responses[i]
                # Decode response
                response_tokens = query_response[context_length:]
                # Find actual end (first pad token)
                pad_mask = (response_tokens == pad_token_id)
                if pad_mask.any():
                    actual_length = pad_mask.nonzero()[0][0].item()
                    response_tokens = response_tokens[:actual_length]
                
                response_text = model._tokenizer.decode(response_tokens, skip_special_tokens=True)
                
                # Get query to look up answer
                query_tokens = query_response[:context_length].cpu().tolist()
                
                # CRITICAL FIX: Use input_ids prefix as key since prompts are truncated
                # The question gets cut off during truncation, so we can't extract it
                # Instead, match using the first 200 tokens of the input_ids
                query_prefix = tuple(query_tokens[:200])
                gold_answer = model._answer_map.get(query_prefix, "")
                
                # DEBUG: Show lookup attempt
                if not gold_answer and reward_stats["total_calls"] <= 8:
                    reward_stats["missing_answers"] += 1
                    query_text = model._tokenizer.decode(query_tokens[:50], skip_special_tokens=True)
                    print(f"\n[DEBUG REWARD] Input IDs prefix lookup failed:")
                    print(f"  Query prefix length: {len(query_prefix)} tokens")
                    print(f"  Query preview (first 50 tokens): {query_text[:200]}...")
                    print(f"  Available prefixes in map: {len(model._answer_map)}")
                    # Show what prefixes we have
                    for idx, (prefix, ans) in enumerate(list(model._answer_map.items())[:3]):
                        prefix_text = model._tokenizer.decode(list(prefix)[:50], skip_special_tokens=True)
                        print(f"    Map entry {idx}: prefix_len={len(prefix)}, answer='{ans}', preview='{prefix_text[:100]}...'")
                    # Try to find closest match by comparing first tokens
                    for prefix, ans in model._answer_map.items():
                        if len(prefix) == len(query_prefix):
                            # Check if first 50 tokens match
                            if prefix[:50] == query_prefix[:50]:
                                print(f"  [PREFIX MATCH] Found match with first 50 tokens!")
                                gold_answer = ans
                                break
                else:
                    reward_stats["found_answers"] += 1
                
                # Compute reward
                final_answer = extract_final_answer(response_text)
                reward = compute_reward_from_item(final_answer, {"answer": gold_answer})
                scores.append(reward)
                
                # DEBUG: Log reward details for first few calls
                if reward_stats["total_calls"] <= 4:
                    # Get question for display
                    question = ""
                    if hasattr(model, '_input_ids_to_question'):
                        question = model._input_ids_to_question.get(query_prefix, "")
                    query_text = model._tokenizer.decode(query_tokens[:100], skip_special_tokens=True)
                    print(f"\n[DEBUG REWARD] Call #{reward_stats['total_calls']}, Batch item {i}:")
                    if question:
                        print(f"  Question: '{question}'")
                    print(f"  Query preview: {query_text[:150]}...")
                    print(f"  Gold answer: '{gold_answer}'")
                    print(f"  Response: {response_text[:150]}...")
                    print(f"  Extracted final answer: '{final_answer}'")
                    print(f"  Reward: {reward:.4f}")
                    
                    # Track which example this is
                    example_id = f"example_{reward_stats['total_calls']}"
                    if example_id not in reward_stats["rewards_by_example"]:
                        reward_stats["rewards_by_example"][example_id] = []
                    reward_stats["rewards_by_example"][example_id].append({
                        "gold": gold_answer,
                        "predicted": final_answer,
                        "reward": reward
                    })
                
                # Sequence length is context + response length
                seq_len = context_length + len(response_tokens)
                sequence_lengths_list.append(seq_len)
            
            # Return in format expected by PPO: (reward_logits, final_rewards, sequence_lengths)
            device = query_responses.device
            reward_tensor = torch.tensor(scores, device=device, dtype=torch.float32)
            sequence_lengths = torch.tensor(sequence_lengths_list, device=device, dtype=torch.long)
            
            # Create logits tensor in expected shape: (batch_size, seq_len, 1)
            seq_len = query_responses.shape[1]
            reward_logits = reward_tensor.view(batch_size, 1, 1).expand(batch_size, seq_len, 1)
            
            return reward_logits, reward_tensor, sequence_lengths
        else:
            # Use original for other models (e.g., value model)
            if reward_stats["total_calls"] <= 5:
                print(f"[DEBUG REWARD] Using original_get_reward (not our reward model)")
            return original_get_reward(model, query_responses, pad_token_id, context_length)
    
    # Patch BEFORE creating trainer
    trl.trainer.utils.get_reward = patched_get_reward
    
    # Also patch the PPO trainer's imported reference (it imports get_reward at module level)
    ppo_module = sys.modules.get('trl.experimental.ppo.ppo_trainer')
    if ppo_module and hasattr(ppo_module, 'get_reward'):
        ppo_module.get_reward = patched_get_reward
        print(f"[DEBUG] Patched get_reward in PPO trainer module")
    else:
        print(f"[DEBUG] PPO trainer module: {ppo_module}")
        print(f"[DEBUG] Could not find get_reward in PPO trainer module (will use utils.get_reward)")
    
    print(f"[DEBUG] Patched get_reward function in trl.trainer.utils")
    
    # Setup PPO components (AFTER patching)
    tokenizer, ppo_trainer = setup_ppo_components(cfg, train_dataset, total_episodes=num_items)
    
    # Store answer map in reward model for access during training
    # All models are already on the same device (cuda:0) from setup_ppo_components
    ppo_trainer.reward_model._answer_map = answer_map  # Maps (input_ids_prefix) -> answer
    ppo_trainer.reward_model._input_ids_to_question = input_ids_to_question  # For debugging
    ppo_trainer.reward_model._tokenizer = tokenizer
    
    # Patch create_model_card to skip model card generation (template file missing)
    # This is a non-critical feature - training works fine without it
    original_create_model_card = ppo_trainer.create_model_card
    def patched_create_model_card(self, model_name=None, **kwargs):
        """Skip model card creation - template file is missing in local trl clone."""
        try:
            return original_create_model_card(model_name, **kwargs)
        except FileNotFoundError:
            print("[Warning] Skipping model card generation (template file not found)")
            return None
    
    ppo_trainer.create_model_card = patched_create_model_card.__get__(ppo_trainer, type(ppo_trainer))
    
    # Train
    print(f"\n[PPO] Starting training with {num_items} examples...")
    print(f"[DEBUG] Training configuration:")
    print(f"  - Batch size: {cfg.batch_size}")
    print(f"  - PPO epochs: {cfg.ppo_epochs}")
    print(f"  - Max prompt length: {cfg.max_prompt_length}")
    print(f"  - Max response length: {cfg.max_response_length}")
    
    ppo_trainer.train()
    
    # Print reward statistics after training
    print(f"\n[DEBUG] Reward statistics after training:")
    print(f"  - Total reward calls: {reward_stats['total_calls']}")
    print(f"  - Found answers: {reward_stats['found_answers']}")
    print(f"  - Missing answers: {reward_stats['missing_answers']}")
    if reward_stats['missing_answers'] > 0:
        print(f"  [WARNING] {reward_stats['missing_answers']} examples had missing answer lookups!")
    print(f"  - Rewards by example: {len(reward_stats['rewards_by_example'])} examples tracked")
    
    # Save model
    ppo_trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[PPO] Training complete. Model saved to {cfg.output_dir}")


