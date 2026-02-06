from evaluate_grpo_vl_simple import load_model_and_processor, generate_answer
from src.grpo_dataset import DoraGRPODataset
from src.grpo_reward import extract_final_answer, tokenize_answer
from src.ppo_trainer_simple import string_f1
from tqdm import tqdm
from datasets import load_from_disk
from collections import defaultdict
import json
import random

import re

# checkpoint_path = "./outputs/train_sft/stage2/checkpoint-3969"
# model_base = "Qwen/Qwen2-VL-2B-Instruct"
# dataset main location [currently not working...]
# dataset_path = "/projects/XXXX-1/dora/grpo_dataset_updatedv2/"
# dataset backup [xiangyu's scratch partition]

base_models = [
    ("Qwen/Qwen3-VL-8B-Instruct","train_q3_2"), # 0.5095 0.0187
    # "Qwen/Qwen3-VL-8B-Instruct":"train_q3",
]

# checkpoints = [
#     50, 100, 150, 200, 250, 300,
# ]

checkpoints = [270, 275, 280, 285, 290, 295, 300]

dataset_path = "/scratch/XXXX-6.XXXX-7/grpo_dataset_updatedv2"
# max_examples = 200
max_examples = 534
max_new_tokens = 256
random.seed(0)

def load_model(model_base, ckpt_pth=None):
    if ckpt_pth is not None:
        model, processor = load_model_and_processor(
                model_base,
                checkpoint_path=ckpt_pth,
                use_lora=True,
            )
    else:
        model, processor = load_model_and_processor(
                model_base,
                use_lora=True,
            )
    return model, processor

dataset = load_from_disk(dataset_path)
# Prepare dataset (same format as training)

mcq_path = r"mcq_dataset_updated_spatial_audit.json"

def build_mcq_index(mcq_pth):
    with open(mcq_pth, "r") as f:
        mcq_lib = json.load(f)
    the_dict = {}
    for item in mcq_lib:
        the_dict[item["question"]] = item
    return the_dict

def select_by_sim(choices, pred_answer):
    scores = []
    pred_token = tokenize_answer(pred_answer)
    for entry in choices:
        entry_token = tokenize_answer(entry)
        scores.append(string_f1(entry_token, pred_token))
    return max(range(len(scores)), key=scores.__getitem__), scores

def inject_prompt(message, choices):
    message = message[0]
    assert message['content'][-1]['type'] == 'text'
    modify_str = message['content'][-1]['text']
    context = modify_str.split("\nContext:")[-1].split("\nQuestion:")[0]
    question = modify_str.split("\nContext:")[-1].split("\nQuestion:")[-1].split("\nAnswer")[0]
    new_sys_prompt = 'You are a helpful visual reasoning assistant for kids.\n Think step by step and choose one correct choice from 0, 1, 2 or 3. Only return one single digit as the best answer.\n'
    new_context = "Context:" + context
    new_question = "\nQuestion:" + question
    new_choices = f"\nChoices: " + json.dumps({
        0: choices[0],
        1: choices[1],
        2: choices[2],
        3: choices[3],
    })
    message['content'][-1]['text'] = new_sys_prompt + new_context + new_question + new_choices
    return [message]

def extract_one_digit_answer(text):
    '''
    Extracts the first one-digit number (0–9) from messy answer strings.
    Returns the digit as an int, or None if no digit is found.
    '''
    # Look for a single digit anywhere in the string
    match = re.search(r'\b(\d)\b', text)
    if match:
        return int(match.group(1))
    return None


def eval_comp_f1(dataset, model, processor, mcq_dict):
    results = []
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    # loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # indices = torch.randperm(len(dataset))
    # shuffled_dataset = [dataset[i] for i in indices]

    print(f"\n[3/3] Evaluating on {num_examples} examples...")
    print("=" * 80)
    correct = 0
    for i in tqdm(range(num_examples)):
    # for batch in loader:
        random_idx = random.randint(0, num_examples)
        item = dataset[random_idx]
        # item = {k: v[0] for k, v in batch.items()}
        messages = item["prompt"]
        question = item["question"]
        mcq_item = mcq_dict[question]
        if question not in mcq_dict.keys():
            print("Skipped question ", question)
            continue
        mcq_item = mcq_dict[question]
        mcq_choices = mcq_item["choices"]
        correct_answer = mcq_item["correct_index"]
        
        # Generate prediction
        try:
            generated_text = generate_answer(model, processor, messages, max_new_tokens)
            # Extract final answer (same logic as reward function)
            pred_answer = extract_final_answer(generated_text)
            pred_choice, sim_score = select_by_sim(mcq_choices, pred_answer)
            if pred_choice == correct_answer:
                correct += 1
            if i % 50 == 0:
                print(f"Question: {question}")
                print("Choices:")
                for j in range(4):
                    print(f"{j}. {mcq_choices[j]}")
                print(f"Correct Answer: {correct_answer}")
                print(f"Generated Answer: {pred_answer}")
                print("Similarity Score,", sim_score)
                print("Current Accuracy: ", correct / (i+1))
            
        except Exception as e:
            print(f"ERROR generating answer: {e}")
    
    return correct / num_examples

def eval_direct_instr(dataset, model, processor, mcq_dict):
    results = []
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    discard_num = 0
    # loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # indices = torch.randperm(len(dataset))
    # shuffled_dataset = [dataset[i] for i in indices]

    # Category tracking
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    
    # Track all metadata fields
    metadata_fields = ["reasoning_category", "compositional_complexity", "temporal_span", 
                       "requires_visual", "requires_transcript", "modality"]
    metadata_correct = {field: defaultdict(int) for field in metadata_fields}
    metadata_total = {field: defaultdict(int) for field in metadata_fields}

    print(f"\n[3/3] Evaluating on {num_examples} examples...")
    print("=" * 80)
    correct = 0
    select = list(range(num_examples))
    random.shuffle(select)

    for i in tqdm(range(num_examples)):
    # for batch in loader:
        random_idx = select[i]
        item = dataset[random_idx]
        print(item)
        # item = {k: v[0] for k, v in batch.items()}
        messages = item["prompt"]
        question = item["question"]
        if question not in mcq_dict.keys():
            print("Skipped question ", question)
            continue
        mcq_item = mcq_dict[question]
        mcq_choices = mcq_item["choices"]
        correct_answer = mcq_item["correct_index"]
        category = mcq_item.get("reasoning_category", "unknown")
        messages = inject_prompt(messages, mcq_choices)
        
        # Generate prediction
        try:
            generated_text = generate_answer(model, processor, messages, max_new_tokens)
            # Extract final answer (same logic as reward function)
            # pred_answer = extract_final_answer(generated_text)
            pred_choice = generated_text.split("<|im_end|>")[0]
            if len(pred_choice) > 1:
                pred_choice = extract_one_digit_answer(pred_choice)
                assert pred_choice is not None
            else:
                pred_choice = int(pred_choice)

            category_total[category] += 1
            
            # Track all metadata fields
            for field in metadata_fields:
                value = str(mcq_item.get(field, "unknown"))
                metadata_total[field][value] += 1
                if pred_choice == correct_answer:
                    metadata_correct[field][value] += 1


            if pred_choice == correct_answer:
                correct += 1
                category_correct[category] += 1
            if i % 50 == 0:
                print(f"Question: {question}")
                print("Choices:")
                for j in range(4):
                    print(f"{j}. {mcq_choices[j]}")
                print(f"Correct Answer: {correct_answer}")
                print(f"Generated Answer: {pred_choice}")
                print("Current Accuracy: ", correct / (i+1))
            
        except Exception as e:
            print(f"ERROR generating answer: {e}")
            print("Generated Text: ", generated_text)
            discard_num += 1

    # Print per-category results
    print("\n" + "=" * 80)
    print("RESULTS BY CATEGORY:")
    print("=" * 80)
    for cat in sorted(category_total.keys()):
        cat_acc = category_correct[cat] / category_total[cat] if category_total[cat] > 0 else 0
        print(f"{cat}: {category_correct[cat]}/{category_total[cat]} = {cat_acc:.4f}")
    
    # Print results for all metadata fields
    for field in metadata_fields:
        print("\n" + "=" * 80)
        print(f"RESULTS BY {field.upper()}:")
        print("=" * 80)
        for val in sorted(metadata_total[field].keys()):
            total = metadata_total[field][val]
            corr = metadata_correct[field][val]
            acc = corr / total if total > 0 else 0
            print(f"{val}: {corr}/{total} = {acc:.4f}")
    print("=" * 80)
    print("failed percentage: ", discard_num / num_examples)
    print("adjusted for error: ", correct / num_examples)
    return correct / (num_examples-discard_num)

mcq_dict = build_mcq_index(mcq_path)
for base_model, v in base_models:
    for ckpt in checkpoints:
        checkpoint_path = f"./outputs/{v}/checkpoint-{ckpt}"
        model, processor = load_model(base_model, ckpt_pth=checkpoint_path)
        eval_dataset = DoraGRPODataset(
            dataset=dataset,
            processor=processor,
            max_prompt_length=512,
            max_completion_length=256,
            use_frames=True,
            max_frames=4,
            system_prompt="You are a helpful visual reasoning assistant for kids. \nThink step by step and always give a final concise answer in the first sentence.",
        )
        print(eval_direct_instr(eval_dataset, model, processor, mcq_dict))
        print(checkpoint_path)
