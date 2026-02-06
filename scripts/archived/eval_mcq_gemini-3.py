import sys
sys.path.insert(0, '/projects/XXXX-2-storage/Workspace/dora/dora')
from src.grpo_dataset import DoraGRPODataset
from src.grpo_reward import extract_final_answer, tokenize_answer
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoProcessor
from collections import defaultdict
import json
import random
from google import genai  # NEW import
import re
import time
from google.genai import types
import argparse
import os

# Configure Gemini API with NEW package
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

dataset_path = "/scratch/XXXX-6.XXXX-7/grpo_dataset_updatedv2"
max_new_tokens = 256
random.seed(0)

# Use Qwen processor for consistent preprocessing
qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
dataset = load_from_disk(dataset_path)

def build_mcq_index(mcq_pth):
    with open(mcq_pth, "r") as f:
        mcq_lib = json.load(f)
    the_dict = {}
    for item in mcq_lib:
        the_dict[item["question"]] = item
    return the_dict

def inject_prompt(message, choices, use_thinking=False):
    """Same as Qwen eval - modifies prompt to MCQ format."""
    message = message[0]
    assert message['content'][-1]['type'] == 'text'
    modify_str = message['content'][-1]['text']
    context = modify_str.split("\nContext:")[-1].split("\nQuestion:")[0]
    question = modify_str.split("\nContext:")[-1].split("\nQuestion:")[-1].split("\nAnswer")[0]
    
    if use_thinking:
        new_sys_prompt = 'Directly choose one correct choice from 0, 1, 2 or 3. Only return one single digit as the best answer.\n'
    else:
        new_sys_prompt = 'Directly choose one correct choice from 0, 1, 2 or 3. Only return one single digit as the best answer.\n'
    
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
    '''Extract first one-digit number (0–9) from answer strings.'''
    match = re.search(r'\b(\d)\b', text)
    if match:
        return int(match.group(1))
    return None

def generate_gemini_answer(client, messages, use_thinking=False, max_retries=10):
    """Generate answer from Gemini API using NEW google.genai package."""
    for attempt in range(max_retries):
        try:
            content = []
            
            # Extract images and text from messages
            message = messages[0]
            for item in message['content']:
                if item['type'] == 'image':
                    content.append(item['image'])  # PIL Image
                elif item['type'] == 'text':
                    content.append(item['text'])
            
            # Build config based on thinking flag
            if use_thinking:
                config = types.GenerateContentConfig(
                    max_output_tokens=max_new_tokens,
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_level="low")
                )
            else:
                config = types.GenerateContentConfig(
                    max_output_tokens=max_new_tokens,
                    temperature=0.0
                )
            
            # Generate with NEW API
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=content,
                config=config
            )
            
            return response.text
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                wait_time = min(2 ** attempt * 30, 600)
                print(f"\n⏳ Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n❌ Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise e
    return None

def eval_gemini(dataset, client, mcq_dict, max_examples=10, use_thinking=False):
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    discard_num = 0

    # Category tracking
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    
    # Track all metadata fields
    metadata_fields = ["reasoning_category", "compositional_complexity", "temporal_span", 
                       "requires_visual", "requires_transcript", "modality"]
    metadata_correct = {field: defaultdict(int) for field in metadata_fields}
    metadata_total = {field: defaultdict(int) for field in metadata_fields}

    thinking_mode = "WITH THINKING" if use_thinking else "WITHOUT THINKING"
    print(f"\n[Gemini Eval - {thinking_mode}] Evaluating on {num_examples} examples...")
    print("=" * 80)
    correct = 0
    select = list(range(num_examples))
    random.shuffle(select)

    for i in tqdm(range(num_examples)):
        random_idx = select[i]
        item = dataset[random_idx]
        messages = item["prompt"]
        question = item["question"]
        
        if question not in mcq_dict.keys():
            print("Skipped question ", question)
            continue
            
        mcq_item = mcq_dict[question]
        mcq_choices = mcq_item["choices"]
        correct_answer = mcq_item["correct_index"]
        category = mcq_item.get("reasoning_category", "unknown")
        messages = inject_prompt(messages, mcq_choices, use_thinking=use_thinking)
        
        # Generate prediction
        try:
            print(f"\n[{i+1}/{num_examples}] Processing: {question[:50]}...")
            generated_text = generate_gemini_answer(client, messages, use_thinking=use_thinking)
            
            if generated_text is None:
                print(f"  ❌ No response received")
                discard_num += 1
                continue
            
            print(f"  ✓ Response: {generated_text[:100]}...")
                
            # Extract answer
            pred_choice = generated_text.strip()
            if len(pred_choice) > 1:
                pred_choice = extract_one_digit_answer(pred_choice)
                if pred_choice is None:
                    print(f"Could not extract digit from: {generated_text}")
                    discard_num += 1
                    continue
            else:
                try:
                    pred_choice = int(pred_choice)
                except:
                    pred_choice = extract_one_digit_answer(pred_choice)
                    if pred_choice is None:
                        discard_num += 1
                        continue

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
                print(f"\nQuestion: {question}")
                print("Choices:")
                for j in range(4):
                    print(f"{j}. {mcq_choices[j]}")
                print(f"Correct Answer: {correct_answer}")
                print(f"Generated Answer: {pred_choice}")
                print(f"Current Accuracy: {correct / (i+1):.4f}")
            
            time.sleep(5)  # Increased delay
            
        except Exception as e:
            print(f"❌ ERROR generating answer: {e}")
            discard_num += 1

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS BY CATEGORY:")
    print("=" * 80)
    for cat in sorted(category_total.keys()):
        cat_acc = category_correct[cat] / category_total[cat] if category_total[cat] > 0 else 0
        print(f"{cat}: {category_correct[cat]}/{category_total[cat]} = {cat_acc:.4f}")
    
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
    
    print(f"\nfailed percentage: {discard_num / num_examples:.4f}")
    final_acc = correct / (num_examples - discard_num) if (num_examples - discard_num) > 0 else 0
    return final_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Gemini on MCQ dataset')
    parser.add_argument('--thinking', action='store_true', help='Enable thinking mode')
    parser.add_argument('--max_examples', type=int, default=534, help='Maximum number of examples to evaluate')
    args = parser.parse_args()
    
    mcq_path = r"mcq_dataset_updated_spatial_audit.json"
    
    # Create dataset with appropriate system prompt
    if args.thinking:
        system_prompt = "You are a helpful visual reasoning assistant for kids."
    else:
        system_prompt = "You are a helpful visual reasoning assistant for kids."
    
    eval_dataset = DoraGRPODataset(
        dataset=dataset,
        processor=qwen_processor,
        max_prompt_length=512,
        max_completion_length=256,
        use_frames=True,
        max_frames=4,
        system_prompt=system_prompt,
    )
    
    mcq_dict = build_mcq_index(mcq_path)
    
    print(f"Running evaluation with thinking={'ENABLED' if args.thinking else 'DISABLED'}")
    final_acc = eval_gemini(eval_dataset, client, mcq_dict, 
                           max_examples=args.max_examples, 
                           use_thinking=args.thinking)
    print(f"\nFinal Accuracy: {final_acc}")