"""
Test SFT model motion tag generation - mimicking training code exactly.
"""

import os
import sys
sys.path.insert(0, '.')

import torch
import json
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from datasets import Dataset, DatasetDict

# Import training functions
from training.train_sft import prepare_dataset
from training.vision_process import process_vision_info
from configs.data_root import DATA_ROOT

# Key frame roots
STR_KF_ROOT = os.path.join(DATA_ROOT, "videos/stgr/temporal_grounding/kfs")
STR_PLM_KF_ROOT = os.path.join(DATA_ROOT, "videos/stgr/plm/kfs")

def prepare_input_like_training(example, processor):
    """
    Prepare input exactly like training collate_fn does.
    """
    # Prepare messages
    messages = example["messages"][:2]  # System + User (no assistant for generation)
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process vision info
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    if example["task"] == "temporal-spatial free-form QA":
        width, height = video_inputs[0].size(3), video_inputs[0].size(2)
        image_size = (width, height)
        
        # Load key frames
        key_frame_root = STR_PLM_KF_ROOT if example['source'] == "STR_plm_rdcap" else STR_KF_ROOT
        key_frames = []
        
        for key_frame in example["key_frames"]:
            kf_path = os.path.join(key_frame_root, key_frame["path"])
            kf = Image.open(kf_path)
            kf = kf.convert('RGB')
            resized_kf = kf.resize(image_size)
            resized_kf = np.array(resized_kf)
            resized_kf = np.transpose(resized_kf, (2, 0, 1))
            resized_kf = torch.from_numpy(resized_kf)
            key_frames.append((key_frame["time"], resized_kf))
        
        # Build frame prompt
        frame_prompt = ""
        refined_image_inputs = []
        kf_idx = 0
        ori_idx = 0
        frame_idx = 1
        
        while ori_idx < len(video_inputs[0]):
            time_now = int(ori_idx / video_kwargs['fps'][0])
            if kf_idx < len(key_frames) and time_now >= key_frames[kf_idx][0]:
                refined_image_inputs.append(key_frames[kf_idx][1])
                time_now = key_frames[kf_idx][0]
                frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                kf_idx += 1
            else:
                refined_image_inputs.append(video_inputs[0][ori_idx])
                time_now = round(ori_idx / video_kwargs['fps'][0], 1)
                frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                ori_idx += 1
            frame_idx += 1
        
        refined_image_inputs = torch.stack(refined_image_inputs)
        text = text.replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)
        
        # Process with processor
        inputs = processor(
            text=[text],
            images=[refined_image_inputs],  # Video as image sequence
            videos=None,
            return_tensors="pt",
            padding=True,
            do_resize=False
        )
    
    elif example["task"] in ["temporal QA", "General video QA MCQ", "General video QA Free-form"]:
        frame_prompt = ""
        ori_idx = 0
        while ori_idx < len(video_inputs[0]):
            time_now = round(ori_idx / video_kwargs['fps'][0], 1)
            frame_prompt += f"Frame {ori_idx + 1} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
            ori_idx += 1
        frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
        text = text.replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)
        
        inputs = processor(
            text=[text],
            images=video_inputs,
            videos=None,
            return_tensors="pt",
            padding=True,
            do_resize=False
        )
    else:
        raise ValueError(f"Unsupported task: {example['task']}")
    
    return inputs


def test_sft_motion_generation():
    print("="*70)
    print("SFT Model Motion Tag Generation Test (Training-Style)")
    print("="*70)
    
    # Clear GPU memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model
    model_path = "outputs/sft_h200_4403849/checkpoint-5000"  # ← UPDATED to latest model
    print(f"\n1. Loading SFT model: {model_path}")
    
    # GPU 0 = GTX 745 (4GB), GPUs 1-4 = V100 (32GB)
    # Force to use GPU 1 (V100)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # Maps to physical GPU 1 (first visible)
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("✅ Model loaded on V100 GPU")
    
    # Clear cache after loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load sample
    print("\n2. Loading sample from SFT dataset...")
    dataset_path = os.path.join(DATA_ROOT, "json_data/STGR-SFT-filtered-motion.json")  # ← v3 dataset
    
    with open(dataset_path) as f:
        data = json.load(f)
    
    # Use sample #5283 - has STRONGEST motion (speed: 2.621 units/s)
    import re
    sample_idx = 5283
    sample = data[sample_idx]
    
    print(f"   Using sample #{sample_idx} (STRONGEST motion: speed 2.621 units/s)")
    print(f"   Motion: up-left motion (speed: 2.621 units/s, smooth)")
    print("✅ Found sample with motion tags")
    print(f"   Task: {sample['task']}")
    print(f"   Question: {sample['question'][:80]}...")
    
    # Extract expected motion tags
    motion_tags = re.findall(r'<motion>([^<]+)</motion>', sample['reasoning_process'])
    print(f"\n   Expected motion tags ({len(motion_tags)}):")
    for i, tag in enumerate(motion_tags, 1):
        is_stationary = 'stationary' in tag.lower()
        marker = "⚠️ (stationary)" if is_stationary else "✅ (moving)"
        print(f"     {i}. <motion>{tag}</motion> {marker}")
    
    # Prepare sample
    print("\n3. Preparing input (mimicking training collate_fn)...")
    prepared = prepare_dataset(sample)
    
    try:
        inputs = prepare_input_like_training(prepared, processor)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print("✅ Input prepared")
    except Exception as e:
        print(f"❌ Error preparing input: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Generate
    print("\n4. Generating response...")
    print("   (This may take 30-60 seconds...)")
    
    # Clear cache before generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,  # Reduced from 512 to save memory
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        
        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)
        
        print("✅ Generation complete")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Display result
    print("\n" + "="*70)
    print("GENERATED RESPONSE:")
    print("="*70)
    print(response)
    print("="*70)
    
    # Check for tags
    print("\n5. Analyzing response...")
    
    has_think = '<think>' in response
    has_answer = '<answer>' in response
    has_obj = '<obj>' in response
    has_box = '<box>' in response
    has_time = '<t>' in response and '</t>s' in response
    has_motion = '<motion>' in response
    
    print(f"   <think> tag: {'✅' if has_think else '❌'}")
    print(f"   <answer> tag: {'✅' if has_answer else '❌'}")
    print(f"   <obj> tag: {'✅' if has_obj else '❌'}")
    print(f"   <box> tag: {'✅' if has_box else '❌'}")
    print(f"   <t>...</t>s tag: {'✅' if has_time else '❌'}")
    print(f"   <motion> tag: {'✅' if has_motion else '❌'}")
    
    if has_motion:
        gen_motion_tags = re.findall(r'<motion>([^<]+)</motion>', response)
        print(f"\n   Generated motion tags ({len(gen_motion_tags)}):")
        for i, tag in enumerate(gen_motion_tags, 1):
            print(f"     {i}. <motion>{tag}</motion>")
    
    # Final verdict
    print("\n" + "="*70)
    if has_motion:
        print("🎉 SUCCESS: SFT model outputs <motion> tags!")
        print("   ✅ Motion-aware reasoning is working")
        print("   ✅ GRPO motion reward should work")
        return True
    else:
        print("⚠️  ISSUE: SFT model does NOT output <motion> tags")
        print("   ❌ Model learned spatial-temporal grounding (<obj><box><t>)")
        print("   ❌ But NOT motion descriptions (<motion>)")
        print("\n   Possible reasons:")
        print("   1. Motion tags are too rare in training data")
        print("   2. Model needs stronger supervision for motion")
        print("   3. May emerge during GRPO training")
        return False


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - skipping test")
        sys.exit(0)
    
    success = test_sft_motion_generation()
    sys.exit(0 if success else 1)
