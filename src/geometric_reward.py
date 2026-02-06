"""
Geometric Reward Function for Motion Reasoning GRPO Training.

Computes multi-dimensional reward combining spatial, temporal, motion, and caption metrics.
"""

from typing import List, Dict, Optional
import sys
import re
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evidence_parser import parse_evidence_chain, validate_evidence_format, parse_think_predict_chain, ThinkPredictStep
from motion_metrics import (
    compute_spatial_reward,
    compute_temporal_reward,
    compute_motion_reward,
    compute_caption_reward
)


def compute_geometric_reward(
    completions: Optional[List[str]] = None,
    gt_evidence_steps: Optional[List[List[Dict]]] = None,
    questions: Optional[List[str]] = None,
    answers: Optional[List[str]] = None,
    img_widths: Optional[List[int]] = None,  # Video dimensions per sample
    img_heights: Optional[List[int]] = None,  # Video dimensions per sample
    fps: float = 30.0,
    lambda_s: float = 0.25,  # Spatial weight
    lambda_t: float = 0.15,  # Temporal weight
    lambda_m: float = 0.35,  # Motion weight
    lambda_c: float = 0.20,  # Caption weight
    lambda_f: float = 0.05,  # Format weight (for binary gate)
    debug: bool = False,
    **kwargs
) -> List[float]:
    """
    Compute multi-dimensional geometric reward for each completion.
    
    Pipeline:
    1. Parse completion text → evidence steps + answer
    2. R_format: Check parseability and validity (binary gate)
    3. If R_format=0, return 0.0 (invalid output)
    4. Otherwise compute component rewards:
       - R_spatial: Bbox IoU
       - R_temporal: Interval IoU
       - R_motion: Trajectory match (direction + speed + smoothness)
       - R_caption: Text similarity (token F1 + Levenshtein)
    5. Return weighted sum: R = λ_s·R_s + λ_t·R_t + λ_m·R_m + λ_c·R_c
    
    Args:
        completions: List of model completion strings
        gt_evidence_steps: List of ground truth evidence step lists
        questions: List of questions (for logging)
        answers: List of ground truth answers
        fps: Frames per second for motion computation
        lambda_s: Weight for spatial reward
        lambda_t: Weight for temporal reward
        lambda_m: Weight for motion reward
        lambda_c: Weight for caption reward
        lambda_f: Weight for format reward (used for logging, gate is binary)
        debug: Enable debug logging
        **kwargs: Additional arguments (may contain data from trainer)
    
    Returns:
        List of reward floats, one per completion
    """
    # Handle TRL's different calling conventions
    if completions is None:
        completions = kwargs.get('completions', [])
    
    if gt_evidence_steps is None:
        gt_evidence_steps = kwargs.get('gt_evidence_steps', [])
    
    if questions is None:
        questions = kwargs.get('question', kwargs.get('questions', []))
    
    if answers is None:
        answers = kwargs.get('answer', kwargs.get('answers', []))
    
    # Validate inputs
    if not completions:
        if debug:
            print("[GEOMETRIC_REWARD] WARNING: No completions provided!")
        return [0.0]
    
    if not gt_evidence_steps:
        if debug:
            print("[GEOMETRIC_REWARD] WARNING: No ground truth evidence provided!")
        return [0.0] * len(completions)
    
    # Ensure gt_evidence_steps has same length as completions
    if len(gt_evidence_steps) < len(completions):
        # Repeat or pad
        gt_evidence_steps = gt_evidence_steps * ((len(completions) // len(gt_evidence_steps)) + 1)
        gt_evidence_steps = gt_evidence_steps[:len(completions)]
    
    rewards = []
    
    for i, completion in enumerate(completions):
        # Extract text from completion (handles both string and message dict formats)
        if isinstance(completion, list) and completion and isinstance(completion[0], dict):
            # Message dict format: extract content
            completion_text = ""
            for msg in completion:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    completion_text = content if isinstance(content, str) else str(content)
                    break
        elif isinstance(completion, str):
            completion_text = completion
        else:
            completion_text = str(completion) if completion else ""
        
        # Get ground truth for this sample
        gt_steps = gt_evidence_steps[i] if i < len(gt_evidence_steps) else []
        gt_answer = answers[i] if answers and i < len(answers) else ""
        question = questions[i] if questions and i < len(questions) else ""
        
        if debug and i == 0:
            print(f"\n[GEOMETRIC_REWARD] Processing completion {i}:")
            print(f"  Question: {question[:100] if question else 'N/A'}")
            print(f"  Completion length: {len(completion_text)} chars")
            print(f"  GT steps: {len(gt_steps)}")
        
        # Parse completion - try Think-Predict format first, fallback to old format
        pred_steps = []
        pred_answer = ""
        use_think_predict = False
        
        try:
            # Get video dimensions for this sample (from dataset via kwargs)
            img_width = kwargs.get('img_width', [1280])[i] if 'img_width' in kwargs else (img_widths[i] if img_widths and i < len(img_widths) else 1280)
            img_height = kwargs.get('img_height', [720])[i] if 'img_height' in kwargs else (img_heights[i] if img_heights and i < len(img_heights) else 720)
            
            if debug and i == 0:
                print(f"  Video dimensions: {img_width}×{img_height}")
                if 'img_width' in kwargs:
                    print(f"  Got dimensions from kwargs: img_width={kwargs.get('img_width')}, img_height={kwargs.get('img_height')}")
            
            # Try new Think-Predict format with actual video dimensions
            think_pred_steps = parse_think_predict_chain(completion_text, img_width=img_width, img_height=img_height)
            if think_pred_steps and len(think_pred_steps) > 0:
                # Convert ThinkPredictStep to EvidenceStep format using Predict bboxes
                from evidence_parser import EvidenceStep
                pred_steps = []
                for tp_step in think_pred_steps:
                    # Use Predict bboxes as main bboxes (refined estimates)
                    bboxes = tp_step.pred_bboxes if tp_step.pred_bboxes else tp_step.think_bboxes
                    if bboxes:
                        pred_steps.append(EvidenceStep(
                            t_s=tp_step.time if tp_step.time is not None else 0.0,
                            t_e=(tp_step.time + 1.0) if tp_step.time is not None else 1.0,
                            bboxes=bboxes,
                            motion_text=tp_step.motion_text,
                            description=tp_step.description,
                        ))
                use_think_predict = True
                # Extract answer
                answer_match = re.search(r'Answer:\s*([^\n]+)', completion_text, re.IGNORECASE)
                pred_answer = answer_match.group(1).strip() if answer_match else ""
        except Exception as e:
            if debug and i == 0:
                print(f"  [INFO] Think-Predict parse failed, trying old format: {e}")
        
        # Fallback to old format if needed
        if not pred_steps:
            try:
                pred_steps, pred_answer = parse_evidence_chain(completion_text)
            except Exception as e:
                if debug:
                    print(f"  [ERROR] Failed to parse completion (both formats): {e}")
                rewards.append(0.0)
                continue
        
        # R_format: Binary gate for validity
        is_valid = len(pred_steps) > 0
        
        if not is_valid:
            if debug and i == 0:
                print(f"  R_format: INVALID (parsed {len(pred_steps)} steps)")
            rewards.append(0.0)
            continue
        
        if debug and i == 0:
            print(f"  R_format: VALID ({len(pred_steps)} steps)")
        
        # Compute component rewards
        try:
            r_spatial = compute_spatial_reward(pred_steps, gt_steps)
            r_temporal = compute_temporal_reward(pred_steps, gt_steps)
            r_motion = compute_motion_reward(pred_steps, gt_steps, fps)
            r_caption = compute_caption_reward(pred_steps, gt_steps)
            
            if debug and i == 0:
                print(f"  R_spatial: {r_spatial:.3f}")
                print(f"  R_temporal: {r_temporal:.3f}")
                print(f"  R_motion: {r_motion:.3f}")
                print(f"  R_caption: {r_caption:.3f}")
                
                # Debug: show sample bbox comparison
                if pred_steps and gt_steps:
                    p_bbox = pred_steps[0].bboxes[0] if pred_steps[0].bboxes else None
                    g_bbox = gt_steps[0].get('bboxes', [[]])[0][0] if gt_steps[0].get('bboxes', [[]]) and gt_steps[0]['bboxes'][0] else None
                    if p_bbox and g_bbox:
                        print(f"  Sample bbox comparison:")
                        print(f"    Pred: {p_bbox}")
                        print(f"    GT:   {g_bbox}")
            
            # Weighted combination
            total_reward = (
                lambda_s * r_spatial +
                lambda_t * r_temporal +
                lambda_m * r_motion +
                lambda_c * r_caption
            )
            
            # Ensure in [0, 1]
            total_reward = max(0.0, min(1.0, total_reward))
            
            if debug and i == 0:
                print(f"  Total reward: {total_reward:.3f}")
            
            rewards.append(float(total_reward))
        
        except Exception as e:
            if debug:
                print(f"  [ERROR] Failed to compute rewards: {e}")
            rewards.append(0.0)
    
    # Log summary
    if debug and len(rewards) > 0:
        avg_reward = sum(rewards) / len(rewards)
        nonzero_rewards = [r for r in rewards if r > 0]
        print(f"\n[GEOMETRIC_REWARD] Batch summary:")
        print(f"  Total completions: {len(rewards)}")
        print(f"  Average reward: {avg_reward:.3f}")
        print(f"  Non-zero rewards: {len(nonzero_rewards)} / {len(rewards)}")
        if nonzero_rewards:
            print(f"  Avg non-zero: {sum(nonzero_rewards) / len(nonzero_rewards):.3f}")
    
    return rewards


# Convenience function for extracting final answer (for compatibility)
def extract_final_answer(text: str) -> str:
    """
    Extract final answer from model output.
    
    Args:
        text: Model output text
    
    Returns:
        Extracted answer string
    """
    from evidence_parser import extract_final_answer as parser_extract
    return parser_extract(text)


# For backward compatibility with old reward function signature
def compute_dora_reward(*args, **kwargs) -> List[float]:
    """
    Backward compatibility wrapper - redirects to compute_geometric_reward.
    """
    return compute_geometric_reward(*args, **kwargs)
