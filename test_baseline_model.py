"""
Baseline Model Response Test
Tests if the model describes motion in the original video without augmentation.
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def test_baseline():
    print("="*80)
    print("BASELINE MODEL RESPONSE TEST")
    print("="*80)
    
    video_path = "test_videos/Ball_Animation_Video_Generation.mp4"
    
    # Load model
    print("\nLoading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model.eval()
    
    # Prepare prompt
    prompts = [
        "Describe what is happening in this video.",
        "What objects are in this video and what are they doing?",
        "Describe the motion you see in this video.",
    ]
    
    results = []
    
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(model.device)
        
        # Generate response
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        
        response = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"\nRESPONSE:")
        print(f"{response}")
        
        # Check for motion keywords
        motion_keywords = ['move', 'moving', 'motion', 'left', 'right', 'across', 
                          'ball', 'rolls', 'travels', 'goes', 'from', 'to']
        found_keywords = [kw for kw in motion_keywords if kw in response.lower()]
        
        print(f"\nMotion keywords found: {found_keywords}")
        mentions_motion = len(found_keywords) > 0
        print(f"Mentions motion: {mentions_motion}")
        
        results.append({
            'prompt': prompt,
            'response': response,
            'mentions_motion': mentions_motion,
            'keywords_found': found_keywords,
        })
        
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['prompt'][:50]}...")
        print(f"   Motion awareness: {result['mentions_motion']}")
        print(f"   Keywords: {', '.join(result['keywords_found']) if result['keywords_found'] else 'None'}")
    
    overall_motion = sum(r['mentions_motion'] for r in results) / len(results)
    print(f"\nOverall motion awareness: {overall_motion:.1%}")
    
    # Save results
    import json
    with open("results/baseline_response.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/baseline_response.json")


if __name__ == "__main__":
    test_baseline()
