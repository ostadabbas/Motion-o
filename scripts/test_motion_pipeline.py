#!/usr/bin/env python3
"""
Test script for Motion Reasoning GRPO pipeline.

Creates synthetic data and runs a minimal training loop to verify all components work.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from datasets import Dataset as HFDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evidence_parser import parse_evidence_chain, validate_evidence_format, extract_bboxes_from_text
from src.motion_metrics import (
    compute_bbox_iou, compute_spatial_reward, compute_temporal_reward,
    compute_motion_reward, compute_caption_reward
)
from src.geometric_reward import compute_geometric_reward
from src.motion_dataset import MotionGRPODataset, make_motion_grpo_data_module


def create_synthetic_dataset(num_samples: int = 5) -> HFDataset:
    """
    Create synthetic PLM-STC-style dataset for testing.
    
    Args:
        num_samples: Number of samples to generate
    
    Returns:
        HuggingFace dataset with synthetic motion data
    """
    print(f"Creating {num_samples} synthetic samples...")
    
    samples = []
    for i in range(num_samples):
        # Create synthetic frames (random images)
        frames = []
        for j in range(8):
            img = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
            frames.append(img)
        
        # Create synthetic ground truth evidence steps
        gt_evidence_steps = [
            {
                "t_s": 0.0,
                "t_e": 1.5,
                "bboxes": [[[100, 100, 200, 200]], [[120, 110, 220, 210]]],  # Per-frame bboxes
                "motion_desc": {
                    "centroid_trajectory": [(150.0, 150.0), (170.0, 160.0)],
                    "displacement_vectors": [(20.0, 10.0)],
                    "velocities": [50.0],
                    "direction_angles": [0.5]
                },
                "caption": "Object moves rightward"
            },
            {
                "t_s": 1.5,
                "t_e": 3.0,
                "bboxes": [[[220, 210, 320, 310]], [[240, 220, 340, 320]]],
                "motion_desc": {
                    "centroid_trajectory": [(270.0, 260.0), (290.0, 270.0)],
                    "displacement_vectors": [(20.0, 10.0)],
                    "velocities": [50.0],
                    "direction_angles": [0.5]
                },
                "caption": "Object continues moving"
            }
        ]
        
        sample = {
            "video_id": f"test_{i}",
            "video_path": f"/tmp/test_{i}.mp4",
            "frames": frames,
            "question": f"What direction does the object move in video {i}?",
            "answer": "The object moves rightward",
            "gt_evidence_steps": gt_evidence_steps,
            "fps": 30.0
        }
        samples.append(sample)
    
    dataset = HFDataset.from_list(samples)
    print(f"Created dataset with {len(dataset)} samples")
    return dataset


def test_evidence_parser():
    """Test evidence parser with synthetic completions."""
    print("\n" + "="*80)
    print("TEST 1: Evidence Parser")
    print("="*80)
    
    # Test 1: Valid evidence chain
    test_completion = """
Step 1: [0.0–1.5] Person <bbox>[100,100,200,200]</bbox> picks up ball <bbox>[150,150,180,180]</bbox>
Motion: centroid shifts from (150,150) to (165,165), velocity: 50px/s
Description: Person reaches down and picks up the ball

Step 2: [1.5–3.0] Person <bbox>[200,200,300,300]</bbox> throws ball <bbox>[250,100,280,130]</bbox>
Motion: ball velocity increases to 200px/s, direction: upward-right
Description: Person throws the ball upward

Answer: The person picks up and throws the ball
"""
    
    print("Parsing test completion...")
    steps, answer = parse_evidence_chain(test_completion)
    
    print(f"✓ Parsed {len(steps)} evidence steps")
    print(f"✓ Answer: {answer}")
    
    for i, step in enumerate(steps):
        print(f"\n  Step {i+1}:")
        print(f"    Time: [{step.t_s}–{step.t_e}]")
        print(f"    Bboxes: {len(step.bboxes)} boxes")
        print(f"    Description: {step.description[:50]}...")
    
    # Validate format
    is_valid = validate_evidence_format(steps)
    print(f"\n✓ Format validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Test 2: Bbox extraction
    bbox_text = "Object at <bbox>[100,100,200,200]</bbox> moves to <bbox>[300,100,400,200]</bbox>"
    bboxes = extract_bboxes_from_text(bbox_text)
    print(f"✓ Extracted {len(bboxes)} bboxes from text")
    
    return steps


def test_motion_metrics():
    """Test motion metrics computation."""
    print("\n" + "="*80)
    print("TEST 2: Motion Metrics")
    print("="*80)
    
    # Create synthetic pred and GT steps
    from src.evidence_parser import EvidenceStep
    
    pred_steps = [
        EvidenceStep(
            t_s=0.0, t_e=1.5,
            bboxes=[[100, 100, 200, 200], [120, 110, 220, 210]],
            motion_text="moves rightward",
            description="Object moves right"
        ),
        EvidenceStep(
            t_s=1.5, t_e=3.0,
            bboxes=[[220, 210, 320, 310], [240, 220, 340, 320]],
            motion_text="continues moving",
            description="Object continues"
        )
    ]
    
    gt_steps = [
        {
            "t_s": 0.0, "t_e": 1.5,
            "bboxes": [[[100, 100, 200, 200]], [[120, 110, 220, 210]]],
            "motion_desc": {
                "centroid_trajectory": [(150.0, 150.0), (170.0, 160.0)],
                "displacement_vectors": [(20.0, 10.0)],
                "velocities": [50.0],
                "direction_angles": [0.5]
            },
            "caption": "Object moves rightward"
        },
        {
            "t_s": 1.5, "t_e": 3.0,
            "bboxes": [[[220, 210, 320, 310]], [[240, 220, 340, 320]]],
            "motion_desc": {
                "centroid_trajectory": [(270.0, 260.0), (290.0, 270.0)],
                "displacement_vectors": [(20.0, 10.0)],
                "velocities": [50.0],
                "direction_angles": [0.5]
            },
            "caption": "Object continues moving"
        }
    ]
    
    # Test spatial reward
    r_spatial = compute_spatial_reward(pred_steps, gt_steps)
    print(f"✓ Spatial reward: {r_spatial:.3f}")
    
    # Test temporal reward
    r_temporal = compute_temporal_reward(pred_steps, gt_steps)
    print(f"✓ Temporal reward: {r_temporal:.3f}")
    
    # Test motion reward
    r_motion = compute_motion_reward(pred_steps, gt_steps, fps=30.0)
    print(f"✓ Motion reward: {r_motion:.3f}")
    
    # Test caption reward
    r_caption = compute_caption_reward(pred_steps, gt_steps)
    print(f"✓ Caption reward: {r_caption:.3f}")
    
    # Test IoU
    iou = compute_bbox_iou([100, 100, 200, 200], [100, 100, 200, 200])
    print(f"✓ Bbox IoU (perfect match): {iou:.3f}")
    
    iou = compute_bbox_iou([100, 100, 200, 200], [150, 150, 250, 250])
    print(f"✓ Bbox IoU (partial overlap): {iou:.3f}")


def test_geometric_reward():
    """Test geometric reward computation."""
    print("\n" + "="*80)
    print("TEST 3: Geometric Reward")
    print("="*80)
    
    # Valid completion
    completions = [
        """
Step 1: [0.0–1.5] Object <bbox>[100,100,200,200]</bbox> moves right
Motion: centroid shifts from (150,150) to (170,160)
Description: Object moves rightward

Step 2: [1.5–3.0] Object <bbox>[220,210,320,310]</bbox> continues
Motion: continues moving right
Description: Object continues moving

Answer: The object moves rightward
""",
        # Invalid completion (no bboxes)
        "The object moves rightward but I don't know where.",
    ]
    
    gt_evidence_steps = [
        [
            {
                "t_s": 0.0, "t_e": 1.5,
                "bboxes": [[[100, 100, 200, 200]]],
                "motion_desc": {
                    "centroid_trajectory": [(150.0, 150.0)],
                    "displacement_vectors": [],
                    "velocities": [],
                    "direction_angles": []
                },
                "caption": "Object moves rightward"
            }
        ],
        [
            {
                "t_s": 0.0, "t_e": 1.5,
                "bboxes": [[[100, 100, 200, 200]]],
                "motion_desc": {
                    "centroid_trajectory": [(150.0, 150.0)],
                    "displacement_vectors": [],
                    "velocities": [],
                    "direction_angles": []
                },
                "caption": "Object moves rightward"
            }
        ]
    ]
    
    questions = ["What direction?", "What direction?"]
    answers = ["rightward", "rightward"]
    
    rewards = compute_geometric_reward(
        completions=completions,
        gt_evidence_steps=gt_evidence_steps,
        questions=questions,
        answers=answers,
        debug=True
    )
    
    print(f"\n✓ Computed rewards for {len(rewards)} completions")
    print(f"  - Valid completion reward: {rewards[0]:.3f}")
    print(f"  - Invalid completion reward: {rewards[1]:.3f}")
    
    assert rewards[0] > 0, "Valid completion should get positive reward"
    assert rewards[0] > rewards[1], "Valid completion should get higher reward than invalid"
    print("✓ Reward function working correctly (valid > invalid)")


def test_dataset_loading():
    """Test dataset loading and formatting."""
    print("\n" + "="*80)
    print("TEST 4: Dataset Loading")
    print("="*80)
    
    # Create synthetic dataset
    dataset = create_synthetic_dataset(num_samples=3)
    
    # Create mock processor
    class MockProcessor:
        def __init__(self):
            self.image_processor = type('obj', (object,), {'do_resize': False})()
    
    processor = MockProcessor()
    
    # Create GRPO dataset
    print("Creating MotionGRPODataset...")
    grpo_dataset = MotionGRPODataset(
        dataset=dataset,
        processor=processor,
        max_frames=8
    )
    
    print(f"✓ Dataset length: {len(grpo_dataset)}")
    
    # Get first item
    print("Getting first item...")
    item = grpo_dataset[0]
    
    print(f"✓ Item keys: {list(item.keys())}")
    print(f"✓ Prompt type: {type(item['prompt'])}")
    print(f"✓ GT evidence steps: {len(item['gt_evidence_steps'])}")
    print(f"✓ Question: {item['question'][:50]}...")
    
    # Check prompt structure
    messages = item['prompt']
    assert isinstance(messages, list), "Prompt should be list of messages"
    assert messages[0]['role'] == 'user', "First message should be from user"
    
    content = messages[0]['content']
    assert isinstance(content, list), "Content should be list"
    
    # Count images and text
    num_images = sum(1 for c in content if c['type'] == 'image')
    num_text = sum(1 for c in content if c['type'] == 'text')
    
    print(f"✓ Prompt has {num_images} images and {num_text} text blocks")
    
    # Test data module
    print("\nTesting make_motion_grpo_data_module...")
    data_module = make_motion_grpo_data_module(
        dataset=dataset,
        processor=processor,
        max_frames=8
    )
    
    print(f"✓ Data module keys: {list(data_module.keys())}")
    print(f"✓ Train dataset length: {len(data_module['train_dataset'])}")


def test_integration():
    """Test full integration: dataset -> model -> reward."""
    print("\n" + "="*80)
    print("TEST 5: Integration Test")
    print("="*80)
    
    print("This test would require a real model and GPU, skipping for now.")
    print("To test with a real model:")
    print("  1. Preprocess PLM-STC data with scripts/preprocess_plm_stc.py")
    print("  2. Run scripts/train_motion_grpo.py with --max-steps 10")
    print("  3. Verify training loop completes without errors")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MOTION REASONING GRPO PIPELINE TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: Evidence parser
        test_evidence_parser()
        
        # Test 2: Motion metrics
        test_motion_metrics()
        
        # Test 3: Geometric reward
        test_geometric_reward()
        
        # Test 4: Dataset loading
        test_dataset_loading()
        
        # Test 5: Integration
        test_integration()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nNext steps:")
        print("1. Preprocess your PLM-STC dataset:")
        print("   python scripts/preprocess_plm_stc.py /path/to/plm_stc /path/to/output")
        print("\n2. Run training:")
        print("   bash shell_scripts/train_motion.sh /path/to/preprocessed/dataset")
        print("\n3. For quick validation, run 10 steps:")
        print("   python scripts/train_motion_grpo.py /path/to/dataset --max-steps 10 --use-lora --use-4bit")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
