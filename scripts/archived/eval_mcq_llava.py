import sys
sys.path.insert(0, '/projects/XXXX-2-storage/Workspace/dora/dora')

# pip install git+XXXX
from src.grpo_dataset import DoraGRPODataset
from tqdm import tqdm
from datasets import load_from_disk
from collections import defaultdict
from transformers import AutoProcessor
from PIL import Image
import numpy as np
import copy
import torch
import json
import random
import re
import warnings

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

warnings.filterwarnings("ignore")

dataset_path = "/scratch/XXXX-6.XXXX-7/grpo_dataset_updatedv2"
max_examples = 534
max_new_tokens = 256
random.seed(0)

# LLaVA-Video model setup
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, 
    torch_dtype="bfloat16", 
    device_map=device_map,
    attn_implementation="eager"  # or "sdpa" for scaled dot product attention
)
model.eval()

# Use Qwen processor for consistent preprocessing
qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

dataset = load_from_disk(dataset_path)

eval_dataset = DoraGRPODataset(
    dataset=dataset,
    processor=qwen_processor,
    max_prompt_length=512,
    max_completion_length=256,
    use_frames=True,
    max_frames=4,
    system_prompt="Think step by step and always give a final concise answer in the first sentence.",
    # system_prompt="Directly choose the correct answer. Respond with ONLY a single digit (0, 1, 2, or 3). No explanation.\n",
)

mcq_path = r"mcq_dataset_updated_spatial_audit.json"

def build_mcq_index(mcq_pth):
    with open(mcq_pth, "r") as f:
        mcq_lib = json.load(f)
    the_dict = {}
    for item in mcq_lib:
        the_dict[item["question"]] = item
    return the_dict

def inject_prompt(message, choices):
    """Same as Qwen eval - modifies prompt to MCQ format."""
    message = message[0]
    assert message['content'][-1]['type'] == 'text'
    modify_str = message['content'][-1]['text']
    context = modify_str.split("\nContext:")[-1].split("\nQuestion:")[0]
    question = modify_str.split("\nContext:")[-1].split("\nQuestion:")[-1].split("\nAnswer")[0]
    new_sys_prompt = 'Directly choose one correct choice from 0, 1, 2 or 3. Only return one single digit as the best answer.\n'
    # new_sys_prompt = "Directly choose the correct answer. Respond with ONLY a single digit (0, 1, 2, or 3). No explanation.\n"
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
    '''Extract first one-digit number (0–9) from answer strings.
       Returns None if multiple digits are found (model echoed choices).'''
    # Find all standalone digits
    matches = re.findall(r'\b(\d)\b', text)
    print(f"  Matches found: {matches}")
    
    # If more than one digit found, model probably echoed choices - mark as failed
    if len(matches) > 1:
        return None
    
    # If exactly one digit found, return it
    if len(matches) == 1:
        return int(matches[0])
    
    return None

def generate_llava_video_answer(model, tokenizer, image_processor, messages):
    """Generate answer from LLaVA-Video using same message format as Qwen."""
    try:
        # Extract images and text from messages
        message = messages[0]
        images = []
        text_prompt = ""
        
        for item in message['content']:
            if item['type'] == 'image':
                images.append(item['image'])  # PIL Image
            elif item['type'] == 'text':
                text_prompt = item['text']
        
        # Process images for LLaVA-Video
        if images:
            # Convert PIL images to numpy array (treating as video frames)
            frames = []
            for img in images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_array = np.array(img)
                frames.append(img_array)
            
            frames = np.stack(frames)  # Shape: (num_frames, H, W, 3)
            
            # Process with image_processor
            # Changed .half() to .bfloat16() to match model dtype
            video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
            video = [video]
            modalities = ["video"]
            
            # Build question with image token
            question = DEFAULT_IMAGE_TOKEN + f"\n{text_prompt}"
        else:
            video = None
            modalities = ["text"]
            question = text_prompt
        
        # Use conversation template
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)
        
        # Generate
        with torch.no_grad():
            if video is not None:
                output = model.generate(
                    input_ids,
                    images=video,
                    modalities=modalities,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=max_new_tokens,
                )
            else:
                output = model.generate(
                    input_ids,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=max_new_tokens,
                )
        
        text_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        return text_output
        
    except Exception as e:
        print(f"\n❌ Error in generate_llava_video_answer: {e}")
        raise e

def eval_llava_video(dataset, model, tokenizer, image_processor, mcq_dict):
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

    print(f"\n[LLaVA-Video Eval] Evaluating on {num_examples} examples...")
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
        messages = inject_prompt(messages, mcq_choices)
        
        # Generate prediction
        try:
            print(f"\n[{i+1}/{num_examples}] Processing: {question[:50]}...")
            generated_text = generate_llava_video_answer(model, tokenizer, image_processor, messages)
            
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

mcq_dict = build_mcq_index(mcq_path)
print("Final Accuracy:", eval_llava_video(eval_dataset, model, tokenizer, image_processor, mcq_dict))