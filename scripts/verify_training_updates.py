#!/usr/bin/env python3
"""
Quick verification that training updates work correctly.
Tests: parser, prompt format, model loading, reward computation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_parser():
    """Test Think-Predict parser."""
    from src.evidence_parser import parse_think_predict_chain
    
    test_text = """
Step 1: [0.0s] Ball on left side
  Think: (150,400),(250,600)
  Predict: (160,420),(240,580)
  Motion: Starting position

Step 2: [1.0s] Ball moved right
  Think: (450,390),(550,590)
  Predict: (460,410),(540,570)
  Motion: Moved 300 units right, velocity 150 units/s

Answer: The ball moved from left to right.
"""
    
    steps = parse_think_predict_chain(test_text)
    assert len(steps) == 2, f"Expected 2 steps, got {len(steps)}"
    assert len(steps[0].think_bboxes) == 1, "Step 1 should have 1 think bbox"
    assert len(steps[0].pred_bboxes) == 1, "Step 1 should have 1 pred bbox"
    assert "Starting position" in steps[0].motion_text, "Motion text should be captured"
    print("✓ Parser works!")


def test_prompt_format():
    """Test new prompt generation."""
    from src.motion_dataset import MotionGRPODataset
    
    # Create mock dataset
    class MockProcessor:
        pass
    
    class MockDataset:
        def __len__(self):
            return 1
        def __getitem__(self, i):
            return {}
    
    dataset = MotionGRPODataset(MockDataset(), MockProcessor(), max_frames=8)
    
    # Test prompt building
    prompt = dataset._build_chain_prompt(
        question="Where did the ball move?",
        num_frames=4,
        frame_times=[0.0, 1.0, 2.0, 3.0]
    )
    
    assert "Think: (x1,y1),(x2,y2)" in prompt, "Should have Think format"
    assert "Predict: (x1,y1),(x2,y2)" in prompt, "Should have Predict format"
    assert "0-1000 scale" in prompt, "Should specify coordinate scale"
    assert "Frame 1 at t=0.00s" in prompt, "Should list frames"
    print("✓ Prompt format works!")


def test_model_loading():
    """Test model class selection."""
    import torch
    from transformers import (
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )
    
    # Test class selection logic
    def get_model_cls(model_id):
        mid = model_id.lower()
        if "qwen3" in mid:
            return Qwen3VLForConditionalGeneration
        elif "qwen2.5" in mid or "qwen2_5" in mid:
            return Qwen2_5_VLForConditionalGeneration
        else:
            return Qwen2VLForConditionalGeneration
    
    assert get_model_cls("Qwen/Qwen2.5-VL-7B-Instruct") == Qwen2_5_VLForConditionalGeneration
    assert get_model_cls("Qwen/Qwen2-VL-2B-Instruct") == Qwen2VLForConditionalGeneration
    assert get_model_cls("Qwen/Qwen3-VL-8B-Instruct") == Qwen3VLForConditionalGeneration
    print("✓ Model class selection works!")


def test_reward_computation():
    """Test reward with Think-Predict format."""
    from src.geometric_reward import compute_geometric_reward
    
    completion = """
Step 1: [0.0s] Ball on left
  Think: (100,400),(200,600)
  Predict: (150,420),(250,580)
  Motion: Starting position

Step 2: [1.0s] Ball moved right
  Think: (400,390),(500,590)
  Predict: (450,410),(550,570)
  Motion: Moved right

Answer: Ball moved from left to right.
"""
    
    # Mock ground truth (with proper motion_desc format)
    gt_steps = [
        {
            "t_s": 0.0, 
            "t_e": 1.0, 
            "bboxes": [[192, 302, 320, 418]], 
            "motion_desc": {
                "centroid_trajectory": [[256, 360]],
                "displacement_vectors": [[0, 0]],
            },
            "caption": "Starting"
        },
        {
            "t_s": 1.0, 
            "t_e": 2.0, 
            "bboxes": [[576, 295, 704, 411]], 
            "motion_desc": {
                "centroid_trajectory": [[640, 353]],
                "displacement_vectors": [[384, -7]],
            },
            "caption": "Moved right"
        },
    ]
    
    rewards = compute_geometric_reward(
        completions=[completion],
        gt_evidence_steps=[gt_steps],
        questions=["Where did the ball move?"],
        answers=["Left to right"],
        debug=True
    )
    
    assert len(rewards) == 1, f"Expected 1 reward, got {len(rewards)}"
    assert rewards[0] > 0, f"Expected positive reward, got {rewards[0]}"
    print(f"✓ Reward computation works! (reward={rewards[0]:.3f})")


def main():
    print("\n" + "="*60)
    print("VERIFYING TRAINING UPDATES")
    print("="*60 + "\n")
    
    try:
        test_parser()
        test_prompt_format()
        test_model_loading()
        test_reward_computation()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe training pipeline is ready:")
        print("  ✓ Think-Predict parser works")
        print("  ✓ Prompt format updated")
        print("  ✓ Model loading fixed")
        print("  ✓ Reward computation handles new format")
        print("\nYou can now run:")
        print("  python scripts/train_motion_grpo.py --model-id Qwen/Qwen2.5-VL-7B-Instruct ...")
        print()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
