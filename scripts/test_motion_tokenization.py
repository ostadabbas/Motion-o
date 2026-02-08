#!/usr/bin/env python3
"""
Test that <motion> tags work correctly with Qwen2.5-VL tokenizer.

This script verifies that the new Motion Chain of Thought (MCoT) format
doesn't break tokenization or cause unexpected behavior.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from transformers import AutoProcessor, AutoTokenizer
except ImportError as e:
    print(f"Error: Missing required packages. {e}")
    print("Please install: pip install transformers")
    sys.exit(1)


def test_motion_tag_tokenization():
    """Test tokenization of motion tags."""
    
    print("="*80)
    print("Motion Chain of Thought (MCoT) Tokenization Test")
    print("="*80)
    
    # Load tokenizer
    print("\n1. Loading Qwen2.5-VL tokenizer...")
    try:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        tokenizer = processor.tokenizer
        print("   ✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load tokenizer: {e}")
        return False
    
    # Test cases
    test_cases = [
        {
            "name": "Basic motion tag",
            "text": "<motion>leftward motion (speed: 0.1 units/s, smooth)</motion>"
        },
        {
            "name": "Complete temporal-spatial-motion chain",
            "text": "<obj>man</obj><box>[0.5,0.5,0.7,0.7]</box>at<t>10.0</t>s<motion>leftward motion (speed: 0.1 units/s, smooth)</motion>"
        },
        {
            "name": "Multiple objects with motion",
            "text": "<obj>car</obj><box>[0.2,0.3,0.4,0.5]</box>at<t>5.0</t>s<obj>car</obj><box>[0.3,0.3,0.5,0.5]</box>at<t>6.0</t>s<motion>rightward motion (speed: 0.05 units/s, smooth)</motion>"
        },
        {
            "name": "Stationary object",
            "text": "<obj>building</obj><box>[0.1,0.1,0.3,0.3]</box>at<t>1.0</t>s<motion>stationary (no significant motion)</motion>"
        },
        {
            "name": "Full think block with motion",
            "text": "<think>The man is visible at <obj>man</obj><box>[0.46,0.43,0.77,0.94]</box>at<t>47.5</t>s and later at <obj>man</obj><box>[0.41,0.38,0.77,0.99]</box>at<t>54.2</t>s<motion>leftward motion (speed: 0.004 units/s, smooth)</motion>. This indicates he is moving left.</think>"
        },
        {
            "name": "Diagonal motion",
            "text": "<motion>up-right motion (speed: 0.08 units/s, jerky)</motion>"
        },
        {
            "name": "Erratic motion",
            "text": "<motion>downward motion (speed: 0.15 units/s, erratic)</motion>"
        }
    ]
    
    print("\n2. Testing tokenization of motion tags...\n")
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        name = test_case["name"]
        text = test_case["text"]
        
        print(f"   Test {i}: {name}")
        print(f"   Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        
        try:
            # Tokenize
            tokens = tokenizer(text, return_tensors="pt")
            input_ids = tokens['input_ids'][0]
            
            # Decode back
            decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
            
            # Check token count
            token_count = len(input_ids)
            
            print(f"   ✓ Tokenized successfully: {token_count} tokens")
            
            # Verify key tags are preserved in decoded text
            if "<motion>" in text and "<motion>" not in decoded:
                print(f"   ⚠ Warning: <motion> tag not preserved in decoded text")
                all_passed = False
            
            if "</motion>" in text and "</motion>" not in decoded:
                print(f"   ⚠ Warning: </motion> tag not preserved in decoded text")
                all_passed = False
            
        except Exception as e:
            print(f"   ✗ Tokenization failed: {e}")
            all_passed = False
        
        print()
    
    # Test with actual conversation format
    print("3. Testing full conversation format...\n")
    
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a video understanding assistant. When describing object motion, use <motion> tags."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the person doing in the video?"}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<think>The person appears at <obj>person</obj><box>[0.3,0.4,0.6,0.9]</box>at<t>2.0</t>s and moves to <obj>person</obj><box>[0.4,0.4,0.7,0.9]</box>at<t>5.0</t>s<motion>rightward motion (speed: 0.033 units/s, smooth)</motion>. They are walking right.</think><answer>The person is walking to the right.</answer>"}]
        }
    ]
    
    try:
        # This tests the full chat template formatting
        formatted = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        print(f"   ✓ Chat template applied successfully")
        print(f"   Formatted length: {len(formatted)} characters")
        
        # Tokenize the formatted conversation
        tokens = tokenizer(formatted, return_tensors="pt")
        print(f"   ✓ Full conversation tokenized: {len(tokens['input_ids'][0])} tokens")
        
    except Exception as e:
        print(f"   ✗ Conversation formatting failed: {e}")
        all_passed = False
    
    print()
    
    # Summary
    print("="*80)
    if all_passed:
        print("✓ All tests passed! Motion tags are compatible with Qwen2.5-VL tokenizer.")
    else:
        print("⚠ Some tests showed warnings or failed. Review output above.")
    print("="*80)
    
    return all_passed


def main():
    """Main entry point."""
    success = test_motion_tag_tokenization()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
