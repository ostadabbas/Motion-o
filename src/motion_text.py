"""
Motion Text Generator for Motion Chain of Thought (MCoT).

Converts motion metrics (direction, speed, smoothness) into natural language
descriptions for the <motion> tag in reasoning chains.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from src.motion_metrics import (
    compute_centroid_trajectory,
    compute_displacement_vectors,
    trajectory_smoothness_penalty
)


def compute_dominant_direction(displacements: List[Tuple[float, float]]) -> str:
    """
    Compute the dominant direction of motion from displacement vectors.
    
    Args:
        displacements: List of (dx, dy) displacement vectors
        
    Returns:
        Direction string: "leftward", "rightward", "upward", "downward", 
                         "diagonal", or "stationary"
    """
    if not displacements:
        return "stationary"
    
    # Average displacement
    avg_dx = np.mean([dx for dx, dy in displacements])
    avg_dy = np.mean([dy for dx, dy in displacements])
    
    # Check if movement is negligible (threshold: 0.01 normalized units)
    magnitude = math.sqrt(avg_dx**2 + avg_dy**2)
    if magnitude < 0.01:
        return "stationary"
    
    # Determine primary direction based on larger component
    abs_dx = abs(avg_dx)
    abs_dy = abs(avg_dy)
    
    # If both components are significant, call it diagonal
    if abs_dx > 0.02 and abs_dy > 0.02:
        h_dir = "left" if avg_dx < 0 else "right"
        v_dir = "up" if avg_dy < 0 else "down"
        return f"{v_dir}-{h_dir}"
    
    # Otherwise, determine dominant direction
    if abs_dx > abs_dy:
        return "leftward" if avg_dx < 0 else "rightward"
    else:
        return "upward" if avg_dy < 0 else "downward"


def compute_average_speed(displacements: List[Tuple[float, float]], 
                          time_deltas: List[float]) -> float:
    """
    Compute average speed from displacements and time intervals.
    
    Args:
        displacements: List of (dx, dy) displacement vectors
        time_deltas: List of time intervals between observations
        
    Returns:
        Average speed in normalized units per second
    """
    if not displacements or not time_deltas or len(displacements) != len(time_deltas):
        return 0.0
    
    speeds = []
    for (dx, dy), dt in zip(displacements, time_deltas):
        if dt > 0:
            magnitude = math.sqrt(dx**2 + dy**2)
            speed = magnitude / dt
            speeds.append(speed)
    
    return np.mean(speeds) if speeds else 0.0


def compute_trajectory_quality(trajectory: List[Tuple[float, float]], 
                               fps: float = 30.0) -> str:
    """
    Assess trajectory smoothness and return quality descriptor.
    
    Args:
        trajectory: List of (cx, cy) centroids
        fps: Frames per second (default 30)
        
    Returns:
        Quality string: "smooth", "jerky", or "erratic"
    """
    if len(trajectory) < 2:
        return "smooth"
    
    smoothness_penalty = trajectory_smoothness_penalty(trajectory, fps)
    
    # Map penalty to quality descriptor
    if smoothness_penalty < 0.1:
        return "smooth"
    elif smoothness_penalty < 0.3:
        return "jerky"
    else:
        return "erratic"


def generate_motion_text(bboxes: List[List[float]], 
                         timestamps: List[float],
                         fps: float = 30.0) -> str:
    """
    Generate natural language motion description from bbox trajectory.
    
    Args:
        bboxes: List of [x1, y1, x2, y2] bounding boxes (normalized [0,1])
        timestamps: List of timestamps in seconds
        fps: Video frames per second
        
    Returns:
        Motion description string for <motion> tag
    """
    # Handle edge cases
    if not bboxes or len(bboxes) == 0:
        return "no tracking data"
    
    if len(bboxes) == 1:
        return "stationary (single frame)"
    
    # Compute trajectory
    trajectory = compute_centroid_trajectory(bboxes)
    
    if len(trajectory) < 2:
        return "stationary (insufficient data)"
    
    # Compute displacement vectors
    displacements = compute_displacement_vectors(trajectory)
    
    if not displacements:
        return "stationary (no movement)"
    
    # Check if all displacements are near zero
    total_displacement = sum(math.sqrt(dx**2 + dy**2) for dx, dy in displacements)
    if total_displacement < 0.01:
        return "stationary (no significant motion)"
    
    # Compute time deltas
    time_deltas = []
    for i in range(len(timestamps) - 1):
        dt = timestamps[i + 1] - timestamps[i]
        if dt > 0:
            time_deltas.append(dt)
    
    # Ensure we have valid time deltas
    if not time_deltas:
        return "stationary (invalid timestamps)"
    
    # Compute motion attributes
    direction = compute_dominant_direction(displacements)
    avg_speed = compute_average_speed(displacements, time_deltas)
    quality = compute_trajectory_quality(trajectory, fps)
    
    # Handle stationary case
    if direction == "stationary":
        return "stationary (no significant motion)"
    
    # Format speed with 3 decimal places
    speed_str = f"{avg_speed:.3f}"
    
    # Generate hybrid natural language description
    motion_text = f"{direction} motion (speed: {speed_str} units/s, {quality})"
    
    return motion_text


def generate_motion_text_from_metrics(direction_vector: Tuple[float, float],
                                      speed: float,
                                      smoothness_score: float) -> str:
    """
    Generate motion text from pre-computed metrics.
    
    Args:
        direction_vector: (dx, dy) average displacement vector
        speed: Average speed in units per second
        smoothness_score: Smoothness score in [0, 1] where 1 is smooth
        
    Returns:
        Motion description string
    """
    dx, dy = direction_vector
    magnitude = math.sqrt(dx**2 + dy**2)
    
    if magnitude < 0.01:
        return "stationary (no significant motion)"
    
    # Determine direction
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    
    if abs_dx > 0.02 and abs_dy > 0.02:
        h_dir = "left" if dx < 0 else "right"
        v_dir = "up" if dy < 0 else "down"
        direction = f"{v_dir}-{h_dir}"
    elif abs_dx > abs_dy:
        direction = "leftward" if dx < 0 else "rightward"
    else:
        direction = "upward" if dy < 0 else "downward"
    
    # Determine quality from smoothness score
    if smoothness_score > 0.9:
        quality = "smooth"
    elif smoothness_score > 0.7:
        quality = "jerky"
    else:
        quality = "erratic"
    
    speed_str = f"{speed:.3f}"
    motion_text = f"{direction} motion (speed: {speed_str} units/s, {quality})"
    
    return motion_text


def batch_generate_motion_texts(trajectories: Dict[str, List[Tuple[List[float], float]]],
                                fps: float = 30.0) -> Dict[str, str]:
    """
    Generate motion texts for multiple object trajectories.
    
    Args:
        trajectories: Dict mapping object_name to list of (bbox, timestamp) tuples
        fps: Video frames per second
        
    Returns:
        Dict mapping object_name to motion description string
    """
    motion_texts = {}
    
    for object_name, trajectory_data in trajectories.items():
        if not trajectory_data:
            motion_texts[object_name] = "no tracking data"
            continue
        
        # Separate bboxes and timestamps
        bboxes = [bbox for bbox, _ in trajectory_data]
        timestamps = [ts for _, ts in trajectory_data]
        
        # Generate motion text
        motion_text = generate_motion_text(bboxes, timestamps, fps)
        motion_texts[object_name] = motion_text
    
    return motion_texts
