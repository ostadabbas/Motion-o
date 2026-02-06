"""
Motion Metrics for Geometric Reward Computation.

Implements spatial, temporal, motion, and caption similarity metrics
for comparing predicted and ground truth motion evidence chains.
"""

import numpy as np
import math
import re
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment


# ============================================================================
# Spatial Metrics
# ============================================================================

def compute_bbox_iou(pred_bbox: List[int], gt_bbox: List[int]) -> float:
    """
    Compute IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        pred_bbox: Predicted bbox [x1, y1, x2, y2]
        gt_bbox: Ground truth bbox [x1, y1, x2, y2]
    
    Returns:
        IoU score between 0.0 and 1.0
    """
    if len(pred_bbox) != 4 or len(gt_bbox) != 4:
        return 0.0
    
    px1, py1, px2, py2 = pred_bbox
    gx1, gy1, gx2, gy2 = gt_bbox
    
    # Compute intersection
    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # Compute union
    pred_area = (px2 - px1) * (py2 - py1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    union_area = pred_area + gt_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return float(iou)


def match_bboxes_hungarian(pred_bboxes: List[List[int]], 
                          gt_bboxes: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Hungarian matching to establish correspondence between predicted 
    and ground truth bboxes based on IoU scores.
    
    Args:
        pred_bboxes: List of predicted bboxes [[x1,y1,x2,y2], ...]
        gt_bboxes: List of ground truth bboxes [[x1,y1,x2,y2], ...]
    
    Returns:
        List of (pred_idx, gt_idx) matches
    """
    if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
        return []
    
    # Compute cost matrix (negative IoU for minimization)
    cost_matrix = np.zeros((len(pred_bboxes), len(gt_bboxes)))
    for i, pred_bbox in enumerate(pred_bboxes):
        for j, gt_bbox in enumerate(gt_bboxes):
            iou = compute_bbox_iou(pred_bbox, gt_bbox)
            cost_matrix[i, j] = -iou  # Negative for minimization
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter out matches with IoU = 0
    matches = []
    for i, j in zip(row_ind, col_ind):
        iou = -cost_matrix[i, j]
        if iou > 0:
            matches.append((int(i), int(j)))
    
    return matches


def compute_spatial_reward(pred_steps: List, gt_steps: List) -> float:
    """
    R_spatial: Average IoU across all matched bboxes in all steps.
    
    Args:
        pred_steps: List of predicted EvidenceStep objects
        gt_steps: List of ground truth evidence step dicts
    
    Returns:
        Average spatial IoU score between 0.0 and 1.0
    """
    if not pred_steps or not gt_steps:
        return 0.0
    
    total_iou = 0.0
    total_matches = 0
    
    # Match steps first (by temporal IoU or simply pair by order)
    num_steps = min(len(pred_steps), len(gt_steps))
    
    for i in range(num_steps):
        pred_step = pred_steps[i]
        gt_step = gt_steps[i]
        
        pred_bboxes = pred_step.bboxes if hasattr(pred_step, 'bboxes') else []
        gt_bboxes = gt_step.get('bboxes', [])
        
        # Handle case where gt_bboxes is per-frame: [[bbox1], [bbox2], ...]
        # Flatten to list of bboxes
        if gt_bboxes and isinstance(gt_bboxes[0], list) and len(gt_bboxes[0]) > 0:
            if isinstance(gt_bboxes[0][0], list):
                # Per-frame format: flatten
                gt_bboxes_flat = []
                for frame_boxes in gt_bboxes:
                    gt_bboxes_flat.extend(frame_boxes)
                gt_bboxes = gt_bboxes_flat
        
        if not pred_bboxes or not gt_bboxes:
            continue
        
        # Match bboxes
        matches = match_bboxes_hungarian(pred_bboxes, gt_bboxes)
        
        for pred_idx, gt_idx in matches:
            iou = compute_bbox_iou(pred_bboxes[pred_idx], gt_bboxes[gt_idx])
            total_iou += iou
            total_matches += 1
    
    if total_matches == 0:
        return 0.0
    
    return total_iou / total_matches


# ============================================================================
# Temporal Metrics
# ============================================================================

def compute_temporal_iou(pred_interval: Tuple[float, float], 
                        gt_interval: Tuple[float, float]) -> float:
    """
    Temporal IoU between two [t_s, t_e] intervals.
    
    Args:
        pred_interval: Predicted (start, end) time
        gt_interval: Ground truth (start, end) time
    
    Returns:
        Temporal IoU between 0.0 and 1.0
    """
    pred_start, pred_end = pred_interval
    gt_start, gt_end = gt_interval
    
    # Compute intersection
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    
    if inter_end <= inter_start:
        return 0.0
    
    inter_duration = inter_end - inter_start
    
    # Compute union
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union_duration = union_end - union_start
    
    if union_duration == 0:
        return 0.0
    
    temporal_iou = inter_duration / union_duration
    return float(temporal_iou)


def compute_temporal_reward(pred_steps: List, gt_steps: List) -> float:
    """
    R_temporal: Average temporal IoU across evidence steps.
    
    Args:
        pred_steps: List of predicted EvidenceStep objects
        gt_steps: List of ground truth evidence step dicts
    
    Returns:
        Average temporal IoU between 0.0 and 1.0
    """
    if not pred_steps or not gt_steps:
        return 0.0
    
    total_iou = 0.0
    num_steps = min(len(pred_steps), len(gt_steps))
    
    for i in range(num_steps):
        pred_step = pred_steps[i]
        gt_step = gt_steps[i]
        
        pred_interval = (pred_step.t_s, pred_step.t_e)
        gt_interval = (gt_step.get('t_s', 0.0), gt_step.get('t_e', 1.0))
        
        iou = compute_temporal_iou(pred_interval, gt_interval)
        total_iou += iou
    
    if num_steps == 0:
        return 0.0
    
    return total_iou / num_steps


# ============================================================================
# Motion Metrics
# ============================================================================

def compute_centroid_trajectory(bboxes: List[List[int]]) -> List[Tuple[float, float]]:
    """
    Compute centroid (cx, cy) for each bbox in sequence.
    
    Args:
        bboxes: List of [x1, y1, x2, y2] bounding boxes
    
    Returns:
        List of (cx, cy) centroids
    """
    centroids = []
    for bbox in bboxes:
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centroids.append((float(cx), float(cy)))
        else:
            centroids.append((0.0, 0.0))
    return centroids


def compute_displacement_vectors(trajectory: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Frame-to-frame displacement vectors from centroid trajectory.
    
    Args:
        trajectory: List of (cx, cy) centroids
    
    Returns:
        List of (dx, dy) displacement vectors
    """
    if len(trajectory) < 2:
        return []
    
    displacements = []
    for i in range(len(trajectory) - 1):
        cx1, cy1 = trajectory[i]
        cx2, cy2 = trajectory[i + 1]
        dx = cx2 - cx1
        dy = cy2 - cy1
        displacements.append((float(dx), float(dy)))
    
    return displacements


def direction_cosine_similarity(pred_vectors: List[Tuple[float, float]], 
                                gt_vectors: List[Tuple[float, float]]) -> float:
    """
    Cosine similarity between displacement vector sequences.
    Measures if predicted motion direction matches ground truth.
    
    Args:
        pred_vectors: Predicted (dx, dy) displacement vectors
        gt_vectors: Ground truth (dx, dy) displacement vectors
    
    Returns:
        Average cosine similarity between 0.0 and 1.0
    """
    if not pred_vectors or not gt_vectors:
        return 0.0
    
    # Align vectors (match by index)
    num_vectors = min(len(pred_vectors), len(gt_vectors))
    if num_vectors == 0:
        return 0.0
    
    total_cos_sim = 0.0
    valid_count = 0
    
    for i in range(num_vectors):
        pred_dx, pred_dy = pred_vectors[i]
        gt_dx, gt_dy = gt_vectors[i]
        
        # Compute magnitudes
        pred_mag = math.sqrt(pred_dx**2 + pred_dy**2)
        gt_mag = math.sqrt(gt_dx**2 + gt_dy**2)
        
        if pred_mag == 0 or gt_mag == 0:
            continue
        
        # Compute cosine similarity
        dot_product = pred_dx * gt_dx + pred_dy * gt_dy
        cos_sim = dot_product / (pred_mag * gt_mag)
        
        # Normalize to [0, 1] from [-1, 1]
        cos_sim = (cos_sim + 1.0) / 2.0
        
        total_cos_sim += cos_sim
        valid_count += 1
    
    if valid_count == 0:
        return 0.0
    
    return total_cos_sim / valid_count


def speed_fidelity_score(pred_vectors: List[Tuple[float, float]], 
                        gt_vectors: List[Tuple[float, float]]) -> float:
    """
    Symmetric min-ratio of displacement magnitudes.
    min(|pred|, |gt|) / max(|pred|, |gt|) averaged across frames.
    
    Args:
        pred_vectors: Predicted (dx, dy) displacement vectors
        gt_vectors: Ground truth (dx, dy) displacement vectors
    
    Returns:
        Average speed fidelity between 0.0 and 1.0
    """
    if not pred_vectors or not gt_vectors:
        return 0.0
    
    num_vectors = min(len(pred_vectors), len(gt_vectors))
    if num_vectors == 0:
        return 0.0
    
    total_fidelity = 0.0
    
    for i in range(num_vectors):
        pred_dx, pred_dy = pred_vectors[i]
        gt_dx, gt_dy = gt_vectors[i]
        
        pred_mag = math.sqrt(pred_dx**2 + pred_dy**2)
        gt_mag = math.sqrt(gt_dx**2 + gt_dy**2)
        
        if pred_mag == 0 and gt_mag == 0:
            # Both static
            total_fidelity += 1.0
        elif pred_mag == 0 or gt_mag == 0:
            # One is static, other is moving
            total_fidelity += 0.0
        else:
            # Both moving
            min_mag = min(pred_mag, gt_mag)
            max_mag = max(pred_mag, gt_mag)
            fidelity = min_mag / max_mag
            total_fidelity += fidelity
    
    return total_fidelity / num_vectors


def trajectory_smoothness_penalty(trajectory: List[Tuple[float, float]], 
                                  fps: float = 30.0,
                                  max_speed_px_per_sec: float = 500.0) -> float:
    """
    Penalize physically implausible jumps in consecutive centroids.
    
    Args:
        trajectory: List of (cx, cy) centroids
        fps: Frames per second
        max_speed_px_per_sec: Maximum plausible speed in pixels per second
    
    Returns:
        Penalty in [0, 1] where 0 = smooth, 1 = very jumpy
    """
    if len(trajectory) < 2:
        return 0.0
    
    displacements = compute_displacement_vectors(trajectory)
    
    max_displacement_per_frame = max_speed_px_per_sec / fps
    
    total_penalty = 0.0
    for dx, dy in displacements:
        displacement = math.sqrt(dx**2 + dy**2)
        
        if displacement > max_displacement_per_frame:
            # Penalize based on how much it exceeds
            excess = displacement - max_displacement_per_frame
            penalty = min(excess / max_displacement_per_frame, 1.0)
            total_penalty += penalty
    
    if len(displacements) == 0:
        return 0.0
    
    avg_penalty = total_penalty / len(displacements)
    return min(avg_penalty, 1.0)


def compute_motion_reward(pred_steps: List, 
                         gt_steps: List, 
                         fps: float = 30.0,
                         w_direction: float = 0.4,
                         w_speed: float = 0.4,
                         w_smoothness: float = 0.2) -> float:
    """
    R_motion: Weighted combination of direction, speed, and smoothness.
    
    Args:
        pred_steps: List of predicted EvidenceStep objects
        gt_steps: List of ground truth evidence step dicts
        fps: Frames per second
        w_direction: Weight for direction accuracy
        w_speed: Weight for speed fidelity
        w_smoothness: Weight for smoothness
    
    Returns:
        Motion reward between 0.0 and 1.0
    """
    if not pred_steps or not gt_steps:
        return 0.0
    
    num_steps = min(len(pred_steps), len(gt_steps))
    if num_steps == 0:
        return 0.0
    
    total_direction = 0.0
    total_speed = 0.0
    total_smoothness = 0.0
    valid_steps = 0
    
    for i in range(num_steps):
        pred_step = pred_steps[i]
        gt_step = gt_steps[i]
        
        # Get bboxes
        pred_bboxes = pred_step.bboxes if hasattr(pred_step, 'bboxes') else []
        gt_motion_desc = gt_step.get('motion_desc', {})
        gt_trajectory = gt_motion_desc.get('centroid_trajectory', [])
        gt_displacements = gt_motion_desc.get('displacement_vectors', [])
        
        if not pred_bboxes or not gt_displacements:
            continue
        
        # Compute predicted trajectory and displacements
        pred_trajectory = compute_centroid_trajectory(pred_bboxes)
        pred_displacements = compute_displacement_vectors(pred_trajectory)
        
        if not pred_displacements:
            continue
        
        # Direction accuracy
        direction_sim = direction_cosine_similarity(pred_displacements, gt_displacements)
        total_direction += direction_sim
        
        # Speed fidelity
        speed_fid = speed_fidelity_score(pred_displacements, gt_displacements)
        total_speed += speed_fid
        
        # Smoothness
        smoothness_penalty = trajectory_smoothness_penalty(pred_trajectory, fps)
        smoothness_score = 1.0 - smoothness_penalty
        total_smoothness += smoothness_score
        
        valid_steps += 1
    
    if valid_steps == 0:
        return 0.0
    
    avg_direction = total_direction / valid_steps
    avg_speed = total_speed / valid_steps
    avg_smoothness = total_smoothness / valid_steps
    
    motion_reward = (w_direction * avg_direction + 
                    w_speed * avg_speed + 
                    w_smoothness * avg_smoothness)
    
    return motion_reward


# ============================================================================
# Caption Metrics
# ============================================================================

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words for F1 computation.
    
    Args:
        text: Input text
    
    Returns:
        List of lowercase tokens
    """
    if not text:
        return []
    
    # Normalize: lowercase, remove punctuation, split on whitespace
    text = text.lower()
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    
    return tokens


def token_f1_score(pred_text: str, gt_text: str) -> float:
    """
    Token-level F1 between predicted and ground truth text.
    
    Args:
        pred_text: Predicted text
        gt_text: Ground truth text
    
    Returns:
        F1 score between 0.0 and 1.0
    """
    pred_tokens = tokenize_text(pred_text)
    gt_tokens = tokenize_text(gt_text)
    
    if not gt_tokens:
        return 0.0
    
    if not pred_tokens:
        return 0.0
    
    # Count overlaps
    gt_set = set(gt_tokens)
    pred_set = set(pred_tokens)
    
    # True positives: tokens in both
    tp = len(gt_set & pred_set)
    
    if tp == 0:
        return 0.0
    
    # Precision and recall
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def normalized_levenshtein(pred_text: str, gt_text: str) -> float:
    """
    Normalized edit distance (1 - distance/max_len).
    
    Args:
        pred_text: Predicted text
        gt_text: Ground truth text
    
    Returns:
        Normalized similarity between 0.0 and 1.0
    """
    if not pred_text and not gt_text:
        return 1.0
    
    if not pred_text or not gt_text:
        return 0.0
    
    # Compute Levenshtein distance
    m, n = len(pred_text), len(gt_text)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_text[i - 1] == gt_text[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    distance = dp[m][n]
    max_len = max(m, n)
    
    similarity = 1.0 - (distance / max_len)
    return max(0.0, similarity)


def compute_caption_reward(pred_steps: List, 
                          gt_steps: List,
                          w_f1: float = 0.6,
                          w_levenshtein: float = 0.4) -> float:
    """
    R_caption: Weighted average of token F1 and normalized Levenshtein
    for all caption texts in evidence steps.
    
    Args:
        pred_steps: List of predicted EvidenceStep objects
        gt_steps: List of ground truth evidence step dicts
        w_f1: Weight for token F1
        w_levenshtein: Weight for Levenshtein similarity
    
    Returns:
        Caption reward between 0.0 and 1.0
    """
    if not pred_steps or not gt_steps:
        return 0.0
    
    num_steps = min(len(pred_steps), len(gt_steps))
    if num_steps == 0:
        return 0.0
    
    total_f1 = 0.0
    total_lev = 0.0
    
    for i in range(num_steps):
        pred_step = pred_steps[i]
        gt_step = gt_steps[i]
        
        pred_caption = pred_step.description if hasattr(pred_step, 'description') else ""
        gt_caption = gt_step.get('caption', '')
        
        # Compute metrics
        f1 = token_f1_score(pred_caption, gt_caption)
        lev = normalized_levenshtein(pred_caption, gt_caption)
        
        total_f1 += f1
        total_lev += lev
    
    avg_f1 = total_f1 / num_steps
    avg_lev = total_lev / num_steps
    
    caption_reward = w_f1 * avg_f1 + w_levenshtein * avg_lev
    return caption_reward
