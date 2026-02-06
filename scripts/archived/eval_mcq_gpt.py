import sys
sys.path.insert(0, '/projects/XXXX-2-storage/Workspace/dora/dora')
from transformers import AutoModel, AutoTokenizer, AutoProcessor

from src.grpo_dataset import DoraGRPODataset
from tqdm import tqdm
from datasets import load_from_disk
from collections import defaultdict
import json
from openai import OpenAI
import random
import re
import time
import base64
import io
import argparse
import os

# Configure OpenAI API
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

dataset_path = "/scratch/XXXX-6.XXXX-7/grpo_dataset_updatedv2"
max_new_tokens = 256
random.seed(0)

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
        new_sys_prompt = 'Think and choose one correct choice from 0, 1, 2 or 3. Only return one single digit.\n'
    else:
        new_sys_prompt = 'Directly choose one correct choice from 0, 1, 2 or 3. Only return one single digit.\n'
    
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

def pil_to_base64(img):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def generate_gpt4v_answer(client, messages, use_thinking=False, max_retries=5):
    """Generate answer from GPT-4V API using same message format as Qwen."""
    for attempt in range(max_retries):
        try:
            # Build OpenAI message format
            openai_content = []
            
            # Extract images and text from messages (same format as Qwen)
            message = messages[0]
            for item in message['content']:
                if item['type'] == 'image':
                    # Convert PIL Image to base64
                    img_base64 = pil_to_base64(item['image'])
                    openai_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}",
                            "detail": "low"
                        }
                    })
                elif item['type'] == 'text':
                    openai_content.append({
                        "type": "text",
                        "text": item['text']
                    })
            
            # Build request with optional reasoning
            request_params = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": openai_content}
                ],
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            }
            
            response = client.chat.completions.create(**request_params)
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                wait_time = (attempt + 1) * 30
                print(f"\n⏳ Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n❌ Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise e
    return None

def eval_gpt4v(dataset, client, mcq_dict, max_examples=10, use_thinking=False):
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
    print(f"\n[GPT-4V Eval - {thinking_mode}] Evaluating on {num_examples} examples...")
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
            generated_text = generate_gpt4v_answer(client, messages, use_thinking=use_thinking)
            
            if generated_text is None:
                print(f"  ❌ No response received")
                discard_num += 1
                continue
            
            print(f"  ✓ Response: {generated_text[:100]}...")
                
            # Extract answer (same logic as Qwen eval)
            pred_choice = generated_text.strip()
            if len(pred_choice) > 1:
                pred_choice = extract_one_digit_answer(pred_choice)
                if pred_choice is None:
                    print(f"  ⚠️ Could not extract digit from: {generated_text}")
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
            
            # Small delay to avoid rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ ERROR generating answer: {e}")
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
    
    print(f"\nfailed percentage: {discard_num / num_examples:.4f}")
    final_acc = correct / (num_examples - discard_num) if (num_examples - discard_num) > 0 else 0
    return final_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate GPT-4V on MCQ dataset')
    parser.add_argument('--thinking', action='store_true', help='Enable thinking mode')
    parser.add_argument('--max_examples', type=int, default=534, help='Maximum number of examples to evaluate')
    args = parser.parse_args()
    
    mcq_path = r"mcq_dataset_updated_spatial_audit.json"
    
    # Use Qwen processor for consistent preprocessing
    processor = AutoProcessor.from_pretrained("OpenGVLab/InternVideo2_5_Chat_8B", trust_remote_code=True)
    
    dataset = load_from_disk(dataset_path)
    
    # Create dataset with appropriate system prompt
    if args.thinking:
        system_prompt = "You are a helpful visual reasoning assistant for kids."
    else:
        system_prompt = "You are a helpful visual reasoning assistant for kids."
    
    eval_dataset = DoraGRPODataset(
        dataset=dataset,
        processor=processor,
        max_prompt_length=512,
        max_completion_length=256,
        use_frames=True,
        max_frames=4,
        system_prompt=system_prompt,
    )
    
    mcq_dict = build_mcq_index(mcq_path)
    
    print(f"Running evaluation with thinking={'ENABLED' if args.thinking else 'DISABLED'}")
    final_acc = eval_gpt4v(eval_dataset, client, mcq_dict,
                          max_examples=args.max_examples,
                          use_thinking=args.thinking)
    print(f"\nFinal Accuracy: {final_acc}")