import sys
sys.path.insert(0, '/projects/XXXX-2-storage/Workspace/dora/dora')

from src.grpo_dataset import DoraGRPODataset
from tqdm import tqdm
from datasets import load_from_disk
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import json
import random
import re

dataset_path = "/scratch/XXXX-6.XXXX-7/grpo_dataset_updatedv2"
max_examples = 534
max_new_tokens = 256
random.seed(0)

# InternVideo2.5 model setup
model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda().to(torch.bfloat16)

# Use Qwen processor for consistent dataset preprocessing
# qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
qwen_processor = AutoProcessor.from_pretrained("OpenGVLab/InternVideo2_5_Chat_8B", trust_remote_code=True)

dataset = load_from_disk(dataset_path)

eval_dataset = DoraGRPODataset(
    dataset=dataset,
    processor=qwen_processor,
    max_prompt_length=512,
    max_completion_length=256,
    use_frames=True,
    max_frames=4,
    system_prompt="You are a helpful visual reasoning assistant for kids.",
)

mcq_path = r"mcq_dataset_updated_spatial_audit.json"

# InternVideo2.5 image processing (from official demo)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_internvideo(image, input_size=448, max_num=1):
    """Process a single PIL image for InternVideo2.5 using dynamic preprocessing."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values  # Shape: [num_patches, 3, 448, 448]

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

def generate_internvideo_answer(model, tokenizer, messages):
    """Generate answer from InternVideo2.5 using same message format as Qwen."""
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
        
        if images:
            # InternVideo2.5 requires minimum 4 frames (local_num_frames = 4)
            # Duplicate images if we have fewer than 4
            min_frames = 4
            while len(images) < min_frames:
                images = images + images  # Duplicate
            images = images[:min_frames]  # Take exactly min_frames (or more if you had more)
            
            pixel_values_list = []
            num_patches_list = []
            transform = build_transform(input_size=448)
            
            for img in images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Process like the video demo does for each frame
                tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=1)
                pixel_values = [transform(tile) for tile in tiles]
                pixel_values = torch.stack(pixel_values)  # [num_tiles, 3, 448, 448]
                
                num_patches_list.append(pixel_values.shape[0])
                pixel_values_list.append(pixel_values)
            
            # Concatenate all frames' patches
            pixel_values = torch.cat(pixel_values_list, dim=0)
            pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
            
            print(f"  DEBUG: num images: {len(images)}")
            print(f"  DEBUG: Final pixel_values shape: {pixel_values.shape}")
            print(f"  DEBUG: num_patches_list: {num_patches_list}")
            
            # Build video prefix
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
            question = video_prefix + text_prompt
        else:
            pixel_values = None
            num_patches_list = None
            question = text_prompt
        
        generation_config = dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            top_p=0.1,
            num_beams=1
        )
        
        with torch.no_grad():
            if pixel_values is not None:
                output, _ = model.chat(
                    tokenizer, 
                    pixel_values, 
                    question, 
                    generation_config, 
                    num_patches_list=num_patches_list, 
                    history=None, 
                    return_history=True
                )
            else:
                output, _ = model.chat(
                    tokenizer, 
                    None, 
                    question, 
                    generation_config, 
                    history=None, 
                    return_history=True
                )
        
        return output
        
    except Exception as e:
        print(f"\n❌ Error in generate_internvideo_answer: {e}")
        import traceback
        traceback.print_exc()
        raise e
def eval_internvideo(dataset, model, tokenizer, mcq_dict):
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    discard_num = 0

    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    
    metadata_fields = ["reasoning_category", "compositional_complexity", "temporal_span", 
                       "requires_visual", "requires_transcript", "modality"]
    metadata_correct = {field: defaultdict(int) for field in metadata_fields}
    metadata_total = {field: defaultdict(int) for field in metadata_fields}

    print(f"\n[InternVideo2.5 Eval] Evaluating on {num_examples} examples...")
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
        
        try:
            print(f"\n[{i+1}/{num_examples}] Processing: {question[:50]}...")
            generated_text = generate_internvideo_answer(model, tokenizer, messages)
            
            if generated_text is None:
                print(f"  ❌ No response received")
                discard_num += 1
                continue
            
            print(f"  ✓ Response: {generated_text[:100]}...")
                
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

mcq_dict = build_mcq_index(mcq_path)
print("Final Accuracy:", eval_internvideo(eval_dataset, model, tokenizer, mcq_dict))