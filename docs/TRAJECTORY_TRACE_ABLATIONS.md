# Trajectory Trace Representations - Ablation Study Guide

## Overview

This document describes different ways to represent object trajectories/motion paths in video reasoning, designed for ablation studies comparing explicit trajectory representations.

**Current MotionR1 Approach:** Scattered temporal mentions + motion summary tag

**Purpose:** Test whether explicit trajectory traces improve motion understanding compared to implicit scattered mentions.

---

## Background

### Current Approach (Baseline)

**Format:**
```xml
<obj>car</obj><box>[0.1,0.2,0.3,0.4]</box>at<t>1.0</t>s
The car moves across the scene...
<obj>car</obj><box>[0.2,0.3,0.4,0.5]</box>at<t>2.0</t>s
continuing its path...
<obj>car</obj><box>[0.3,0.4,0.5,0.6]</box>at<t>3.0</t>s
<motion>rightward motion (speed: 0.141 units/s, accel: +0.000 units/s²)</motion>
```

**Properties:**
- ✅ Natural language flow
- ✅ Compatible with Open-o3-Video
- ✅ Token efficient
- ❌ Trajectory is implicit (scattered across text)
- ❌ Model must "connect the dots" mentally

**Hypothesis:** Adding explicit trajectory traces may help the model better understand motion continuity.

---

## Ablation Study Options

### **Ablation 1: Compact Trajectory Tag**

**Format:**
```xml
<obj>car</obj><trace t="1.0-3.0s" boxes="[[0.1,0.2,0.3,0.4],[0.2,0.3,0.4,0.5],[0.3,0.4,0.5,0.6]]"/>
<motion>rightward motion (speed: 0.141 units/s, accel: +0.000 units/s²)</motion>
```

**Key Features:**
- All trajectory data in one structured tag
- Explicit time range
- Array of bounding boxes

**Implementation Complexity:** Medium

**Expected Impact:**
- May help model learn trajectory as single entity
- Good for programmatic processing
- Less natural for language models

**Use Case:** Best for highly structured reasoning tasks

---

### **Ablation 2: Explicit Path Annotation**

**Format:**
```xml
<obj>car</obj> follows this path:
  <keyframe t="1.0s"><box>[0.1,0.2,0.3,0.4]</box></keyframe>
  <keyframe t="2.0s"><box>[0.2,0.3,0.4,0.5]</box></keyframe>
  <keyframe t="3.0s"><box>[0.3,0.4,0.5,0.6]</box></keyframe>
<motion>rightward motion (speed: 0.141 units/s, accel: +0.000 units/s²)</motion>
```

**Key Features:**
- Clear visual structure
- Easy to see full trajectory
- Natural language connector ("follows this path")

**Implementation Complexity:** High (requires reformatting all data)

**Expected Impact:**
- Strong trajectory awareness
- May improve on complex multi-waypoint motions
- Higher token cost

**Use Case:** Best for complex curved trajectories

---

### **Ablation 3: Centroid Path**

**Format:**
```xml
<obj>car</obj><box>[0.1,0.2,0.3,0.4]</box>at<t>1.0</t>s
<obj>car</obj><box>[0.2,0.3,0.4,0.5]</box>at<t>2.0</t>s
<obj>car</obj><box>[0.3,0.4,0.5,0.6]</box>at<t>3.0</t>s
<path>centroid: (0.2,0.3)→(0.25,0.35)→(0.4,0.5)</path>
<motion>rightward motion (speed: 0.141 units/s, accel: +0.000 units/s²)</motion>
```

**Key Features:**
- Lightweight (centroids only)
- Shows the "line" of motion
- Arrow notation (→) makes direction clear

**Implementation Complexity:** Low

**Expected Impact:**
- May help with direction reasoning
- Good for straight-line motions
- Loses bounding box size info

**Use Case:** Best for direction-focused tasks

---

### **Ablation 4: Motion Vector Summary**

**Format:**
```xml
<obj>car</obj><box>[0.1,0.2,0.3,0.4]</box>at<t>1.0</t>s
<obj>car</obj><box>[0.2,0.3,0.4,0.5]</box>at<t>2.0</t>s
<obj>car</obj><box>[0.3,0.4,0.5,0.6]</box>at<t>3.0</t>s
<trajectory frames="3" displacement="(+0.2,+0.2)" duration="2.0s"/>
<motion>rightward motion (speed: 0.141 units/s, accel: +0.000 units/s²)</motion>
```

**Key Features:**
- Compact summary statistics
- Shows net displacement vector
- Includes duration explicitly

**Implementation Complexity:** Low

**Expected Impact:**
- May help with speed/displacement reasoning
- Good for overall motion understanding
- Misses intermediate waypoints

**Use Case:** Best for displacement-focused reasoning

---

### **Ablation 5: Visual ASCII Path**

**Format:**
```xml
<obj>car</obj> motion path (1.0s → 3.0s):
  *---------*---------*  (leftward→rightward)
  ↑         ↑         ↑
  1.0s      2.0s      3.0s
<motion>rightward motion (speed: 0.141 units/s, accel: +0.000 units/s²)</motion>
```

**Key Features:**
- Highly interpretable visual representation
- Shows spatial relationships clearly
- Natural for language models

**Implementation Complexity:** Medium

**Expected Impact:**
- Strong interpretability
- May help with temporal ordering
- High token cost

**Use Case:** Best for interpretability studies

---

### **Ablation 6: Hybrid Approach (Recommended)**

**Format:**
```xml
<obj>car</obj><box>[0.1,0.2,0.3,0.4]</box>at<t>1.0</t>s
The car moves across the scene...
<obj>car</obj><box>[0.2,0.3,0.4,0.5]</box>at<t>2.0</t>s
continuing its path...
<obj>car</obj><box>[0.3,0.4,0.5,0.6]</box>at<t>3.0</t>s
<trajectory>path covers 3 keyframes from 1.0s to 3.0s, net displacement: rightward 0.2 units</trajectory>
<motion>rightward motion (speed: 0.141 units/s, accel: +0.000 units/s²)</motion>
```

**Key Features:**
- Preserves Open-o3-Video format
- Adds natural language trajectory summary
- Helps connect scattered mentions

**Implementation Complexity:** Low

**Expected Impact:**
- Best of both worlds
- May improve without major token overhead
- Easy to compare with baseline

**Use Case:** **Best first ablation to try**

---

### **Ablation 7: Motion Tag with Path Metadata**

**Format:**
```xml
<obj>car</obj><box>[0.1,0.2,0.3,0.4]</box>at<t>1.0</t>s
<obj>car</obj><box>[0.2,0.3,0.4,0.5]</box>at<t>2.0</t>s
<obj>car</obj><box>[0.3,0.4,0.5,0.6]</box>at<t>3.0</t>s
<motion path="3 keyframes, 2.0s duration">rightward motion (speed: 0.141 units/s, accel: +0.000 units/s²)</motion>
```

**Key Features:**
- Minimal change (attributes only)
- Adds trajectory context to motion tag
- Very token efficient

**Implementation Complexity:** Very Low

**Expected Impact:**
- Slight improvement from explicit frame count
- Helps model understand trajectory confidence
- Minimal overhead

**Use Case:** **Easiest ablation to implement**

---

## Implementation Guide

### For Ablation 7 (Easiest Start)

**File to modify:** `scripts/augment_motion_data_simple.py`

```python
def generate_motion_text_with_metadata(bboxes: List[List[float]], 
                                        timestamps: List[float]) -> str:
    """Generate motion description with trajectory metadata."""
    if not bboxes or len(bboxes) == 0:
        return "no tracking data"
    
    if len(bboxes) == 1:
        return "stationary (single frame)"
    
    # Compute direction, speed, acceleration
    direction, avg_speed, acceleration = compute_direction_speed_acceleration(bboxes, timestamps)
    
    if direction == "stationary":
        return "stationary (no significant motion)"
    
    # Add trajectory metadata
    num_keyframes = len(bboxes)
    duration = timestamps[-1] - timestamps[0]
    
    speed_str = f"{avg_speed:.3f}"
    accel_str = f"{acceleration:+.3f}"
    
    # Include path metadata in motion tag
    motion_text = f"{direction} motion (speed: {speed_str} units/s, accel: {accel_str} units/s², path: {num_keyframes} keyframes, {duration:.1f}s duration)"
    
    return motion_text
```

**Expected output:**
```xml
<motion>rightward motion (speed: 0.141 units/s, accel: +0.000 units/s², path: 3 keyframes, 2.0s duration)</motion>
```

---

### For Ablation 6 (Hybrid Approach)

**File to modify:** `scripts/augment_motion_data_simple.py`

```python
def insert_trajectory_summary(reasoning_process: str, 
                               object_name: str, 
                               bboxes: List[List[float]], 
                               timestamps: List[float]) -> str:
    """Insert trajectory summary before motion tag."""
    pos = find_last_object_mention(reasoning_process, object_name)
    if pos is None:
        return reasoning_process
    
    # Compute net displacement
    centroid_start = compute_centroid(bboxes[0])
    centroid_end = compute_centroid(bboxes[-1])
    displacement_x = centroid_end[0] - centroid_start[0]
    displacement_y = centroid_end[1] - centroid_start[1]
    
    # Determine displacement direction
    if abs(displacement_x) > abs(displacement_y):
        disp_dir = "rightward" if displacement_x > 0 else "leftward"
        disp_mag = abs(displacement_x)
    else:
        disp_dir = "downward" if displacement_y > 0 else "upward"
        disp_mag = abs(displacement_y)
    
    trajectory_summary = (
        f"\n<trajectory>path covers {len(bboxes)} keyframes from {timestamps[0]:.1f}s to {timestamps[-1]:.1f}s, "
        f"net displacement: {disp_dir} {disp_mag:.3f} units</trajectory>"
    )
    
    return reasoning_process[:pos] + trajectory_summary + reasoning_process[pos:]


def augment_sample(sample: Dict) -> Dict:
    """Augment a single sample with motion tags and trajectory summaries."""
    # ... existing code ...
    
    for object_name, trajectory_data in tracked_objects.items():
        bboxes = [bbox for bbox, _ in trajectory_data]
        timestamps = [ts for _, ts in trajectory_data]
        
        if len(bboxes) >= 2:
            # Add trajectory summary
            augmented_reasoning = insert_trajectory_summary(
                augmented_reasoning, object_name, bboxes, timestamps
            )
            # Add motion tag
            motion_text = generate_motion_text(bboxes, timestamps)
            augmented_reasoning = insert_motion_tag(
                augmented_reasoning, object_name, motion_text
            )
    
    # ... rest of code ...
```

---

### For Ablation 3 (Centroid Path)

**File to modify:** `scripts/augment_motion_data_simple.py`

```python
def generate_centroid_path(bboxes: List[List[float]]) -> str:
    """Generate centroid path string with arrows."""
    centroids = [compute_centroid(bbox) for bbox in bboxes]
    path_str = "→".join([f"({cx:.2f},{cy:.2f})" for cx, cy in centroids])
    return f"<path>centroid: {path_str}</path>"


def augment_sample(sample: Dict) -> Dict:
    """Augment with centroid paths."""
    # ... existing code ...
    
    for object_name, trajectory_data in tracked_objects.items():
        bboxes = [bbox for bbox, _ in trajectory_data]
        timestamps = [ts for _, ts in trajectory_data]
        
        if len(bboxes) >= 2:
            # Add centroid path
            centroid_path = generate_centroid_path(bboxes)
            augmented_reasoning = insert_motion_tag(
                augmented_reasoning, object_name, centroid_path
            )
            # Add motion tag
            motion_text = generate_motion_text(bboxes, timestamps)
            augmented_reasoning = insert_motion_tag(
                augmented_reasoning, object_name, motion_text
            )
    
    # ... rest of code ...
```

---

## Experimental Setup for Ablation Study

### Recommended Ablation Order:

1. **Baseline (Current):** No trajectory traces, just scattered mentions + motion tag
2. **Ablation 7:** Motion tag with path metadata (easiest)
3. **Ablation 6:** Hybrid with trajectory summary (recommended)
4. **Ablation 3:** Centroid path (if you need direction focus)
5. **Ablation 4:** Motion vector summary (if displacement matters)

### Metrics to Track:

**Primary:**
- V-STAR performance (motion reasoning)
- Motion trajectory reward score
- Speed/direction/acceleration accuracy

**Secondary:**
- Training loss convergence
- Inference speed (tokens/sec)
- Token overhead (avg tokens per sample)

**Qualitative:**
- Generated motion descriptions
- Trajectory understanding in edge cases
- Handling of complex curved paths

### Dataset Preparation:

For each ablation:
```bash
# Create new dataset version
python scripts/augment_motion_data_simple.py \
    --input /mnt/data/stgr/json_data/STGR-SFT-subset.json \
    --output /mnt/data/stgr/json_data/STGR-SFT-subset-motion-ablation-X.json

# Train SFT
sbatch scripts/sbatch_sft_full.sh  # (modify to use ablation dataset)

# Evaluate
python eval/test_motion_understanding.py
```

### Expected Results:

**Hypothesis 1:** Explicit trajectory traces improve motion understanding
- **Test:** Compare V-STAR scores across ablations
- **Expected:** Ablation 6 (hybrid) shows best performance

**Hypothesis 2:** Token overhead hurts with complex traces
- **Test:** Compare training time and convergence
- **Expected:** Ablation 7 (metadata) has minimal overhead

**Hypothesis 3:** Natural language traces work better than structured tags
- **Test:** Compare Ablation 2 (structured) vs Ablation 6 (NL)
- **Expected:** Natural language (Ablation 6) performs better

---

## Publishing Considerations

### Ablation Study Table Format:

```
| Method | V-STAR mAM | Motion Acc | Tokens/Sample | Training Time |
|--------|-----------|------------|---------------|---------------|
| Baseline (scattered) | X.X% | X.X% | XXX | Xh |
| + Path metadata | X.X% | X.X% | XXX | Xh |
| + Trajectory summary | X.X% | X.X% | XXX | Xh |
| + Centroid path | X.X% | X.X% | XXX | Xh |
```

### Key Claims to Test:

1. "Explicit trajectory traces improve motion reasoning by X%"
2. "Natural language traces outperform structured representations"
3. "Minimal metadata (Ablation 7) achieves Y% of gains with Z% overhead"

---

## Advanced: Multi-Object Trajectory Traces

For complex scenes with multiple moving objects:

```xml
Scene shows two objects moving:

<obj>car</obj> trajectory:
  <keyframe t="1.0s"><box>[0.1,0.2,0.3,0.4]</box></keyframe>
  <keyframe t="2.0s"><box>[0.2,0.3,0.4,0.5]</box></keyframe>
  <motion>rightward motion (speed: 0.141 units/s, accel: +0.000 units/s²)</motion>

<obj>person</obj> trajectory:
  <keyframe t="1.0s"><box>[0.6,0.5,0.8,0.9]</box></keyframe>
  <keyframe t="2.0s"><box>[0.5,0.5,0.7,0.9]</box></keyframe>
  <motion>leftward motion (speed: 0.100 units/s, accel: +0.000 units/s²)</motion>

<interaction>car and person moving in opposite directions</interaction>
```

**Use case:** Studying motion relationships and interactions

---

## Future Directions

### Potential Extensions:

1. **Relative trajectories:** "car approaches person at 0.5 units/s"
2. **Path smoothness visualization:** Use visual markers for jerky vs smooth
3. **Predicted vs observed:** Compare expected path with actual path
4. **Occlusion-aware traces:** Show where tracking was lost/regained
5. **3D trajectory projection:** If depth estimation available

### Emerging Research Questions:

- Do trajectory traces help with motion prediction tasks?
- Can explicit paths improve action recognition?
- Do traces help with long-term temporal reasoning?
- How do traces affect compositional generalization?

---

## Summary

**Current Approach (Baseline):**
- Scattered temporal mentions + motion summary
- Token efficient, natural language
- Implicit trajectory representation

**Recommended First Ablation (Easiest Win):**
- **Ablation 7:** Add metadata to motion tag
- Minimal change, low overhead
- Quick to implement and test

**Recommended Best Ablation (Optimal):**
- **Ablation 6:** Hybrid with trajectory summary
- Explicit but natural language
- Good balance of performance vs overhead

**Next Steps:**
1. Implement Ablation 7 as quick test
2. If promising, try Ablation 6
3. Run full ablation study comparing top 3-4 variants
4. Publish results with ablation table

---

## Files to Create for Ablations

```
scripts/
├── augment_motion_ablation7.py   # Path metadata variant
├── augment_motion_ablation6.py   # Hybrid trajectory summary
├── augment_motion_ablation3.py   # Centroid path
└── compare_ablations.py          # Evaluation script

configs/
└── ablation_configs.yaml         # Dataset paths for each ablation

docs/
└── TRAJECTORY_TRACE_ABLATIONS.md # This file
```

**Status:** Ready for implementation! 🚀
