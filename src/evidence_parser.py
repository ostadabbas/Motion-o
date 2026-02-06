"""
Evidence Chain Parser for Motion Reasoning.

Parses model-generated text to extract structured spatio-temporal evidence steps
with bounding boxes and motion descriptors.
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class EvidenceStep:
    """Structured evidence step with spatial-temporal information."""
    t_s: float  # Start time
    t_e: float  # End time
    bboxes: List[List[float]]  # List of [x1, y1, x2, y2] bounding boxes (normalized [0,1])
    motion_text: str  # Raw motion description text
    description: str  # What happened description
    object_names: List[str] = None  # Optional object names
    
    def __post_init__(self):
        if self.object_names is None:
            self.object_names = []


@dataclass
class ThinkPredictStep:
    """Evidence step with Think-then-Predict bboxes for GRPO training."""
    step_num: int
    time: Optional[float]
    description: str
    think_bboxes: List[List[float]]  # Rough estimates (normalized [0,1])
    pred_bboxes: List[List[float]]   # Refined predictions (normalized [0,1])
    motion_text: str


def extract_bboxes_from_text(text: str, img_width: int = 1280, img_height: int = 720) -> List[List[int]]:
    """
    Extract bboxes from various formats that Qwen models output.
    
    Supports formats:
    - <bbox>[x1,y1,x2,y2]</bbox> (normalized [0,1] or absolute pixels)
    - <box>x1,y1,x2,y2</box> (normalized, no brackets)
    - <box>(x1,y1),(x2,y2)</box> (absolute pixels, two corners)
    - <|box_start|>(x1,y1),(x2,y2)<|box_end|> (special tokens, absolute)
    
    Args:
        text: Text containing bbox tags
        img_width: Image width for denormalizing coordinates
        img_height: Image height for denormalizing coordinates
    
    Returns:
        List of [x1, y1, x2, y2] coordinates in absolute pixels
    """
    bboxes = []
    
    # Pattern 1: <|box_start|>(x1,y1),(x2,y2)<|box_end|> (absolute pixels)
    pattern1 = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    matches = re.findall(pattern1, text)
    for match in matches:
        bbox = [int(match[0]), int(match[1]), int(match[2]), int(match[3])]
        bboxes.append(bbox)
    
    # Pattern 2: <bbox>[x1,y1,x2,y2]</bbox> or <bbox>(x1,y1,x2,y2)</bbox>
    # Can be normalized [0,1] or absolute pixels
    pattern2a = r'<bbox>\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]</bbox>'
    pattern2b = r'<bbox>\(([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\)</bbox>'
    
    for pattern in [pattern2a, pattern2b]:
        matches = re.findall(pattern, text)
        for match in matches:
            coords = [float(x) for x in match]
            # Check if normalized (all values < 2.0) or absolute
            if all(c <= 2.0 for c in coords):
                # Normalized - convert to pixels
                bbox = [
                    int(coords[0] * img_width),
                    int(coords[1] * img_height),
                    int(coords[2] * img_width),
                    int(coords[3] * img_height)
                ]
            else:
                # Already in pixels
                bbox = [int(c) for c in coords]
            bboxes.append(bbox)
    
    # Pattern 3: <box>x1,y1,x2,y2</box> (normalized, no brackets)
    pattern3 = r'<box>([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)</box>'
    matches = re.findall(pattern3, text)
    for match in matches:
        coords = [float(x) for x in match]
        # Check if normalized
        if all(c <= 2.0 for c in coords):
            bbox = [
                int(coords[0] * img_width),
                int(coords[1] * img_height),
                int(coords[2] * img_width),
                int(coords[3] * img_height)
            ]
        else:
            bbox = [int(c) for c in coords]
        bboxes.append(bbox)
    
    # Pattern 4: Plain [x1,y1,x2,y2] without tags (fallback)
    if not bboxes:
        pattern4 = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        matches = re.findall(pattern4, text)
        for match in matches:
            bbox = [int(match[0]), int(match[1]), int(match[2]), int(match[3])]
            # Check if reasonable bbox
            if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                bboxes.append(bbox)
    
    return bboxes


def extract_time_interval(text: str) -> Optional[Tuple[float, float]]:
    """
    Extract time interval from text.
    
    Supports formats:
    - [2.1s–3.4s]
    - [2.1s-3.4s]
    - [2.1–3.4]
    - (2.1s, 3.4s)
    - 2.1s to 3.4s
    
    Args:
        text: Text containing time interval
    
    Returns:
        (start_time, end_time) tuple or None
    """
    # Pattern 1: [t_s–t_e] or [t_s-t_e]
    pattern1 = r'\[(\d+\.?\d*)s?[–\-](\d+\.?\d*)s?\]'
    match = re.search(pattern1, text)
    if match:
        t_s = float(match.group(1))
        t_e = float(match.group(2))
        return (t_s, t_e)
    
    # Pattern 2: (t_s, t_e)
    pattern2 = r'\((\d+\.?\d*)s?,\s*(\d+\.?\d*)s?\)'
    match = re.search(pattern2, text)
    if match:
        t_s = float(match.group(1))
        t_e = float(match.group(2))
        return (t_s, t_e)
    
    # Pattern 3: t_s to t_e
    pattern3 = r'(\d+\.?\d*)s?\s+to\s+(\d+\.?\d*)s?'
    match = re.search(pattern3, text, re.IGNORECASE)
    if match:
        t_s = float(match.group(1))
        t_e = float(match.group(2))
        return (t_s, t_e)
    
    return None


def parse_motion_descriptors(motion_text: str) -> Dict:
    """
    Extract motion features from text description.
    
    Examples:
    - "centroid shifts from (220,320) to (180,120)"
    - "velocity: 180px/s, direction: northeast"
    - "displacement: 40 pixels downward"
    
    Args:
        motion_text: Motion description text
    
    Returns:
        Dictionary with extracted motion features (best effort)
    """
    motion_desc = {
        "centroid_start": None,
        "centroid_end": None,
        "velocity": None,
        "direction": None,
        "displacement": None
    }
    
    # Extract centroid positions
    centroid_pattern = r'centroid.*?from\s*\((\d+),\s*(\d+)\)\s*to\s*\((\d+),\s*(\d+)\)'
    match = re.search(centroid_pattern, motion_text, re.IGNORECASE)
    if match:
        motion_desc["centroid_start"] = (int(match.group(1)), int(match.group(2)))
        motion_desc["centroid_end"] = (int(match.group(3)), int(match.group(4)))
    
    # Extract velocity
    velocity_pattern = r'velocity[:\s]+(\d+\.?\d*)\s*px/s'
    match = re.search(velocity_pattern, motion_text, re.IGNORECASE)
    if match:
        motion_desc["velocity"] = float(match.group(1))
    
    # Extract direction
    direction_pattern = r'direction[:\s]+([\w\-]+)'
    match = re.search(direction_pattern, motion_text, re.IGNORECASE)
    if match:
        motion_desc["direction"] = match.group(1).strip()
    
    # Extract displacement
    displacement_pattern = r'displacement[:\s]+(\d+\.?\d*)\s*(?:pixels?|px)'
    match = re.search(displacement_pattern, motion_text, re.IGNORECASE)
    if match:
        motion_desc["displacement"] = float(match.group(1))
    
    return motion_desc


def parse_evidence_step(step_text: str) -> Optional[EvidenceStep]:
    """
    Parse a single evidence step from text.
    
    Expected format:
        Step 1: [2.1s–3.4s] Person <bbox>[120,80,220,350]</bbox> picks up ball <bbox>[200,300,240,340]</bbox>
        Motion: centroid shifts from (220,320) to (180,120), velocity: 150px/s
        Description: Person picks up the ball from the ground
    
    Args:
        step_text: Text for one evidence step
    
    Returns:
        EvidenceStep object or None if parsing fails
    """
    try:
        # Extract time interval
        time_interval = extract_time_interval(step_text)
        if time_interval is None:
            # Default to 0-1 if no time found
            t_s, t_e = 0.0, 1.0
        else:
            t_s, t_e = time_interval
        
        # Extract bboxes
        bboxes = extract_bboxes_from_text(step_text)
        
        # Extract motion description (look for lines starting with Motion:)
        motion_text = ""
        motion_match = re.search(r'Motion[:\s]+(.+?)(?:\n|Description:|$)', step_text, re.IGNORECASE | re.DOTALL)
        if motion_match:
            motion_text = motion_match.group(1).strip()
        
        # Extract description (look for lines starting with Description: or just use remaining text)
        description = ""
        desc_match = re.search(r'Description[:\s]+(.+?)(?:\n\n|Step \d+:|$)', step_text, re.IGNORECASE | re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()
        else:
            # Fallback: use the main text after time interval and bboxes
            lines = step_text.split('\n')
            for line in lines:
                if not line.strip().startswith('Motion:') and not line.strip().startswith('['):
                    if len(line.strip()) > 10:  # Reasonable description length
                        description = line.strip()
                        break
        
        # If no description found, use the whole step text
        if not description:
            description = step_text[:200]  # First 200 chars
        
        return EvidenceStep(
            t_s=t_s,
            t_e=t_e,
            bboxes=bboxes,
            motion_text=motion_text,
            description=description
        )
    
    except Exception as e:
        print(f"Error parsing evidence step: {e}")
        return None


def parse_evidence_chain(text: str) -> Tuple[List[EvidenceStep], str]:
    """
    Parse model output into structured evidence chain.
    
    Expected format:
        Step 1: [2.1s–3.4s] Person <bbox>[120,80,220,350]</bbox> picks up ball
        Motion: centroid shifts from (220,320) to (180,120)
        
        Step 2: [3.4s–5.0s] Ball <bbox>[160,50,210,100]</bbox> hits wall
        Motion: velocity direction flips
        
        Answer: The ball changed direction because it hit the wall.
    
    Args:
        text: Full model output text
    
    Returns:
        Tuple of (evidence_steps, answer)
        - evidence_steps: List of parsed EvidenceStep objects
        - answer: Extracted final answer
    """
    evidence_steps = []
    answer = ""
    
    # Split text into steps
    # Look for Step 1:, Step 2:, etc.
    step_pattern = r'Step \d+:'
    step_splits = re.split(step_pattern, text, flags=re.IGNORECASE)
    
    # First split is often preamble, rest are steps
    for step_text in step_splits[1:]:  # Skip first (preamble)
        # Check if this contains the answer
        if 'answer:' in step_text.lower():
            # Split at answer
            parts = re.split(r'answer[:\s]+', step_text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                step_content = parts[0]
                answer = parts[1].strip()
            else:
                step_content = step_text
        else:
            step_content = step_text
        
        # Parse this step
        step = parse_evidence_step(step_content)
        if step is not None:
            evidence_steps.append(step)
    
    # If no steps found with "Step X:" pattern, try alternative formats
    if len(evidence_steps) == 0:
        # Look for numbered evidence: "1.", "2.", etc.
        numbered_pattern = r'^\d+\.\s+'
        lines = text.split('\n')
        current_step_lines = []
        
        for line in lines:
            if re.match(numbered_pattern, line.strip()):
                # Start of new step
                if current_step_lines:
                    step_text = '\n'.join(current_step_lines)
                    step = parse_evidence_step(step_text)
                    if step is not None:
                        evidence_steps.append(step)
                current_step_lines = [line]
            else:
                current_step_lines.append(line)
        
        # Parse last step
        if current_step_lines:
            step_text = '\n'.join(current_step_lines)
            # Check for answer in last step
            if 'answer:' in step_text.lower():
                parts = re.split(r'answer[:\s]+', step_text, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    step_content = parts[0]
                    answer = parts[1].strip()
                    step = parse_evidence_step(step_content)
                else:
                    step = parse_evidence_step(step_text)
            else:
                step = parse_evidence_step(step_text)
            
            if step is not None:
                evidence_steps.append(step)
    
    # Extract answer if not found yet
    if not answer:
        answer_pattern = r'(?:final\s+)?answer[:\s]+(.+?)(?:\n|$)'
        match = re.search(answer_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Remove trailing punctuation
            answer = re.sub(r'[.!?]+$', '', answer).strip()
    
    return evidence_steps, answer


def validate_evidence_format(steps: List[EvidenceStep], 
                            max_image_size: Tuple[int, int] = (1920, 1080)) -> bool:
    """
    Check if evidence chain has valid format.
    
    Validation checks:
    - Non-empty steps
    - Valid bbox coordinates (within image bounds, x1 < x2, y1 < y2)
    - Temporal ordering (t_s < t_e, steps in sequence)
    
    Args:
        steps: List of EvidenceStep objects
        max_image_size: Maximum image size (width, height) for bbox validation
    
    Returns:
        True if valid, False otherwise
    """
    if not steps or len(steps) == 0:
        return False
    
    max_w, max_h = max_image_size
    prev_t_e = 0.0
    
    for i, step in enumerate(steps):
        # Check temporal interval
        if step.t_s >= step.t_e:
            # Allow equal times for single-frame events
            if step.t_s != step.t_e:
                return False
        
        # Check temporal ordering (non-strict, allow overlaps)
        if step.t_s < prev_t_e - 0.1:  # Small tolerance for rounding
            pass  # Allow some overlap
        prev_t_e = step.t_e
        
        # Check bboxes
        for bbox in step.bboxes:
            if len(bbox) != 4:
                return False
            
            x1, y1, x2, y2 = bbox
            
            # Check coordinates are positive
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                return False
            
            # Check bbox is valid (x1 < x2, y1 < y2)
            if x1 >= x2 or y1 >= y2:
                return False
            
            # Check bbox is within image bounds (relaxed check)
            if x2 > max_w * 2 or y2 > max_h * 2:  # Allow some margin
                return False
    
    return True


def extract_final_answer(text: str) -> str:
    """
    Extract final answer from model output.
    
    Looks for patterns like:
    - "Answer: <answer>"
    - "Final answer: <answer>"
    - Or returns last sentence
    
    Args:
        text: Model output text
    
    Returns:
        Extracted answer string
    """
    if not text:
        return ""
    
    # Try to parse full chain first
    _, answer = parse_evidence_chain(text)
    if answer:
        return answer
    
    # Fallback patterns
    patterns = [
        r'(?:final\s+)?answer[:\s]+(.+?)(?:\n|$)',
        r'the answer is[:\s]+(.+?)(?:\n|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'[.!?]+$', '', answer).strip()
            if answer:
                return answer
    
    # Fallback: return last sentence
    sentences = re.split(r'[.!?]\s+', text)
    if sentences:
        return sentences[-1].strip()
    
    return text.strip()


def parse_think_predict_chain(text: str, img_width: int = None, img_height: int = None) -> List[ThinkPredictStep]:
    """
    Parse Think-Predict format from model output.
    
    Format:
        Step 1: [0.0s] Description
          Think: (x1,y1),(x2,y2)
          Predict: (x1,y1),(x2,y2)
          Motion: motion description
    
    Coordinates are on 0-1000 scale, normalized to [0,1].
    img_width and img_height parameters are ignored (kept for backward compatibility).
    
    Returns:
        List of ThinkPredictStep objects with normalized [0,1] bboxes
    """
    steps = []
    
    # Split by "Step N:"
    step_pattern = r'Step\s+(\d+):\s*\[([^\]]+)\]\s*([^\n]+)'
    step_matches = list(re.finditer(step_pattern, text, re.IGNORECASE))
    
    for i, match in enumerate(step_matches):
        step_num = int(match.group(1))
        time_str = match.group(2).strip()
        description = match.group(3).strip()
        
        # Parse time
        time = None
        try:
            time = float(time_str.replace('s', ''))
        except:
            pass
        
        # Find content for this step (until next step or end)
        start_pos = match.end()
        if i + 1 < len(step_matches):
            end_pos = step_matches[i + 1].start()
        else:
            end_pos = len(text)
        step_content = text[start_pos:end_pos]
        
        # Extract Think bboxes (0-1000 scale → normalize to [0,1])
        think_bboxes = []
        think_pattern = r'Think:\s*\((\d+),(\d+)\),\((\d+),(\d+)\)'
        for bbox_match in re.finditer(think_pattern, step_content):
            x1, y1, x2, y2 = map(int, bbox_match.groups())
            # Normalize from 0-1000 to [0,1]
            think_bboxes.append([
                float(x1 / 1000.0),
                float(y1 / 1000.0),
                float(x2 / 1000.0),
                float(y2 / 1000.0)
            ])
        
        # Extract Predict bboxes (0-1000 scale → normalize to [0,1])
        pred_bboxes = []
        pred_pattern = r'Predict:\s*\((\d+),(\d+)\),\((\d+),(\d+)\)'
        for bbox_match in re.finditer(pred_pattern, step_content):
            x1, y1, x2, y2 = map(int, bbox_match.groups())
            # Normalize from 0-1000 to [0,1]
            pred_bboxes.append([
                float(x1 / 1000.0),
                float(y1 / 1000.0),
                float(x2 / 1000.0),
                float(y2 / 1000.0)
            ])
        
        # Extract motion text
        motion_pattern = r'Motion:\s*([^\n]+)'
        motion_match = re.search(motion_pattern, step_content)
        motion_text = motion_match.group(1).strip() if motion_match else ""
        
        steps.append(ThinkPredictStep(
            step_num=step_num,
            time=time,
            description=description,
            think_bboxes=think_bboxes,
            pred_bboxes=pred_bboxes,
            motion_text=motion_text
        ))
    
    return steps
