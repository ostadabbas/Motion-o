# Implementation Plan: Integrating Motion Reward into Open-o3 Video

## Overview

This document provides **step-by-step instructions** to integrate our motion-aware trajectory reward into Open-o3 Video's infrastructure.

---

## Phase 1: Setup & Baseline (Day 1)

### 1.1 Setup Data Paths

```bash
cd /home/bi.ga/Workspace/vlmm-mcot/Open-o3-Video

# Edit data root
vim src/r1-v/configs/data_root.py
```

**Update:**
```python
DATA_ROOT = "/path/to/your/stgr/dataset"  # Set your actual path
```

### 1.2 Download STGR Dataset

Follow their instructions in README to download:
- STGR-SFT-30k (JSON + videos)
- STGR-RL-36k (JSON + videos)

Expected structure:
```
${DATA_ROOT}/
├── json_data/
│   ├── STGR-SFT.json
│   └── STGR-RL.json
└── videos/
    ├── gqa/
    ├── stgr/
    │   ├── plm/
    │   └── temporal_grounding/
    ├── timerft/
    ├── treevgr/
    ├── tvg_r1/
    ├── videoespresso/
    └── videor1/
```

### 1.3 Run Baseline Training

```bash
cd src/r1-v

# Edit model paths in run_sft_video.sh
vim ../scripts/run_sft_video.sh
# Set MODEL_PATH="/path/to/Qwen2.5-VL-7B-Instruct"
# Set OUT_DIR="/path/to/checkpoints/sft"

# Run SFT
bash ../scripts/run_sft_video.sh

# Edit model paths in run_grpo_video.sh
vim ../scripts/run_grpo_video.sh
# Set MODEL_PATH="/path/to/checkpoints/sft"  # Use SFT checkpoint
# Set OUT_DIR="/path/to/checkpoints/rl_baseline"

# Run baseline RL (without motion reward)
bash ../scripts/run_grpo_video.sh
```

### 1.4 Evaluate Baseline

```bash
cd ../../eval

# Edit eval scripts to point to your checkpoint
bash scripts/eval_all.sh
```

**Record baseline results:**
- V-STAR mAM: _____
- V-STAR mLGM: _____
- VideoMME: _____

---

## Phase 2: Integrate Motion Reward (Day 2-3)

### 2.1 Copy Motion Metrics Module

```bash
cd /home/bi.ga/Workspace/vlmm-mcot/Open-o3-Video

# Copy motion metrics
cp ../src/motion_metrics.py src/r1-v/src/open_r1/
cp ../src/geometric_reward.py src/r1-v/src/open_r1/
cp ../src/evidence_parser.py src/r1-v/src/open_r1/
```

### 2.2 Add Motion Reward Function

Create: `src/r1-v/src/open_r1/motion_reward_integration.py`

```python
"""
Motion-aware trajectory reward for Open-o3 Video integration.
"""

import re
import json
import numpy as np
from motion_metrics import (
    compute_bbox_centroid,
    compute_displacement_vector,
    compute_direction_similarity,
    compute_speed_fidelity,
    compute_trajectory_smoothness
)


def parse_temporal_spatial_claims(think_content: str):
    """
    Parse think content to extract temporal-spatial claims.
    Format: <obj>name</obj><box>[x1,y1,x2,y2]</box>at<t>time</t>s
    """
    pattern = r"<obj>(.*?)</obj>((?:<box>\[.*?\]</box>)+)at<t>(.*?)</t>s"
    parsed_claims = []
    count = 0

    for match in re.finditer(pattern, think_content, re.DOTALL):
        try:
            object_name = match.group(1).strip()
            all_boxes_str = match.group(2)
            timestamp_str = match.group(3).strip()
            timestamp = float(timestamp_str)
            
            individual_box_strs = re.findall(r'\[.*?\]', all_boxes_str)
            bboxes = [json.loads(b_str) for b_str in individual_box_strs]
            
            parsed_claims.append({
                "id": count,
                "object_name": object_name,
                "timestamp": timestamp,
                "bboxes": bboxes
            })
            count += 1

        except (json.JSONDecodeError, ValueError, IndexError) as e:
            continue
            
    return parsed_claims


def convert_coord_format(bbox, image_size):
    """Convert normalized bbox [0,1] to pixel coordinates."""
    nx_min, ny_min, nx_max, ny_max = bbox
    width, height = image_size
    x_min = nx_min * width
    y_min = ny_min * height
    x_max = nx_max * width
    y_max = ny_max * height
    return [x_min, y_min, x_max, y_max]


def motion_trajectory_reward(completions, **kwargs):
    """
    Motion-aware trajectory reward.
    
    Evaluates:
    1. Direction similarity: Cosine similarity of displacement vectors
    2. Speed fidelity: Velocity magnitude matching
    3. Trajectory smoothness: Acceleration penalty for physically implausible motion
    
    Returns:
        List of rewards in [0, 1]
    """
    motion_rewards = []
    idx = 0
    
    for completion in completions:
        think_match = re.search(r"<think>(.*?)</think>", completion[0]["content"], re.DOTALL)
        
        # Skip if no think section or wrong task type
        if not think_match:
            motion_rewards.append(0.0)
            idx += 1
            continue
        
        task = kwargs.get('task', [''])[0]
        
        # Only compute motion reward for temporal-spatial tasks
        if task not in ["temporal-spatial free-form QA"]:
            motion_rewards.append(0.0)
            idx += 1
            continue
        
        think_content = think_match.group(1)
        parsed_claims = parse_temporal_spatial_claims(think_content)
        
        # Need at least 2 temporal points for motion
        if not parsed_claims or len(parsed_claims) < 2:
            motion_rewards.append(0.0)
            idx += 1
            continue
        
        try:
            # Get GT trajectory
            gt_items = kwargs["key_items"][idx]
            gt_frames = kwargs["key_frames"][idx]
            image_size = kwargs["image_size"][idx]
            fps = 30.0  # Default FPS
            
            # Sort claims by timestamp
            parsed_claims = sorted(parsed_claims, key=lambda x: x['timestamp'])
            
            # Extract predicted bbox trajectory
            pred_bboxes = []
            pred_times = []
            for claim in parsed_claims:
                if claim['bboxes'] and len(claim['bboxes']) > 0:
                    bbox = claim['bboxes'][0]  # Take first bbox
                    # Convert to pixel coordinates if normalized
                    if all(0 <= c <= 1 for c in bbox):
                        bbox = convert_coord_format(bbox, image_size)
                    pred_bboxes.append(bbox)
                    pred_times.append(claim['timestamp'])
            
            if len(pred_bboxes) < 2:
                motion_rewards.append(0.0)
                idx += 1
                continue
            
            # Extract GT bbox trajectory
            gt_bboxes = []
            gt_times = []
            sorted_frames = sorted(gt_frames, key=lambda x: x['time'])
            
            for frame in sorted_frames:
                frame_idx = str(frame["idx"])
                if frame_idx in gt_items and gt_items[frame_idx]:
                    obj_key = list(gt_items[frame_idx].keys())[0]
                    gt_bbox = gt_items[frame_idx][obj_key][0]
                    gt_bbox = convert_coord_format(gt_bbox, image_size)
                    gt_bboxes.append(gt_bbox)
                    gt_times.append(frame['time'])
            
            if len(gt_bboxes) < 2:
                motion_rewards.append(0.0)
                idx += 1
                continue
            
            # Compute motion metrics
            
            # 1. Direction similarity (displacement vectors)
            pred_centroids = [compute_bbox_centroid(bbox) for bbox in pred_bboxes]
            gt_centroids = [compute_bbox_centroid(bbox) for bbox in gt_bboxes]
            
            pred_displacements = [
                compute_displacement_vector(pred_centroids[i], pred_centroids[i+1])
                for i in range(len(pred_centroids) - 1)
            ]
            gt_displacements = [
                compute_displacement_vector(gt_centroids[i], gt_centroids[i+1])
                for i in range(len(gt_centroids) - 1)
            ]
            
            if not pred_displacements or not gt_displacements:
                motion_rewards.append(0.0)
                idx += 1
                continue
            
            # Match predicted and GT displacements (use min length)
            min_len = min(len(pred_displacements), len(gt_displacements))
            direction_score = compute_direction_similarity(
                pred_displacements[:min_len],
                gt_displacements[:min_len]
            )
            
            # 2. Speed fidelity (velocity magnitude)
            pred_speeds = []
            for i in range(len(pred_times) - 1):
                dt = pred_times[i+1] - pred_times[i]
                if dt > 0:
                    dx, dy = pred_displacements[i]
                    speed = np.sqrt(dx**2 + dy**2) / dt
                    pred_speeds.append(speed)
            
            gt_speeds = []
            for i in range(len(gt_times) - 1):
                dt = gt_times[i+1] - gt_times[i]
                if dt > 0:
                    dx, dy = gt_displacements[i]
                    speed = np.sqrt(dx**2 + dy**2) / dt
                    gt_speeds.append(speed)
            
            if pred_speeds and gt_speeds:
                avg_pred_speed = np.mean(pred_speeds)
                avg_gt_speed = np.mean(gt_speeds)
                speed_score = compute_speed_fidelity(avg_pred_speed, avg_gt_speed)
            else:
                speed_score = 0.0
            
            # 3. Trajectory smoothness (acceleration penalty)
            smoothness_score = compute_trajectory_smoothness(pred_speeds)
            
            # Combine motion components
            motion_reward = (
                0.4 * direction_score +      # Direction is most important
                0.4 * speed_score +           # Speed matching
                0.2 * smoothness_score        # Smooth trajectories
            )
            
            motion_rewards.append(float(motion_reward))
        
        except Exception as e:
            print(f"Error computing motion reward for sample {idx}: {e}")
            motion_rewards.append(0.0)
        
        idx += 1
    
    return motion_rewards
```

### 2.3 Register Motion Reward

Edit: `src/r1-v/src/open_r1/reward_func.py`

**Add import at top:**
```python
from motion_reward_integration import motion_trajectory_reward
```

**Update registry at bottom:**
```python
reward_funcs_registry = {
    "ans_acc": ans_acc_reward,
    "ans_tiou": ans_tiou_reward,
    "ans_viou": ans_viou_reward,
    "thk_temporal_point": thk_temporal_point_reward,
    "thk_temporal_segment": thk_temporal_segment_reward,
    "thk_spatial": thk_spatial_reward,
    "motion_trajectory": motion_trajectory_reward,  # ⭐ NEW
    "format": format_reward
}
```

### 2.4 Enable Motion Reward

Edit: `src/r1-v/src/open_r1/grpo.py`

**Update default reward functions:**
```python
@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: [
            "ans_acc",
            "ans_tiou",
            "ans_viou",
            "thk_temporal_point",
            "thk_temporal_segment",
            "thk_spatial",
            "motion_trajectory",  # ⭐ ADD THIS
            "format"
        ],
        metadata={"help": "List of reward functions"},
    )
```

### 2.5 Train with Motion Reward

```bash
cd src/r1-v

# Edit run_grpo_video.sh
vim ../scripts/run_grpo_video.sh
# Set MODEL_PATH to SFT checkpoint
# Set OUT_DIR="/path/to/checkpoints/rl_motion"

# Run RL with motion reward
bash ../scripts/run_grpo_video.sh
```

**Monitor logs:**
```bash
tail -f /path/to/checkpoints/rl_motion/log.txt
```

Look for:
```
rewards/motion_trajectory: 0.45  ← Motion reward component
rewards/thk_spatial: 0.62        ← Their spatial reward
rewards/format: 1.0              ← Format reward
reward: 5.23                     ← Total reward (sum of all)
```

### 2.6 Evaluate Motion-Aware Model

```bash
cd ../../eval

# Update model path to motion-aware checkpoint
# bash scripts/eval_all.sh
bash scripts/eval_vstar.sh
```

**Compare results:**

| Metric | Baseline | +Motion Reward | Δ |
|--------|----------|----------------|---|
| V-STAR mAM | ___ | ___ | ___ |
| V-STAR mLGM | ___ | ___ | ___ |
| VideoMME | ___ | ___ | ___ |

---

## Phase 3: Ablation Studies (Day 4-5)

### 3.1 Ablation 1: Direction Only

Edit `motion_reward_integration.py`:
```python
# Combine motion components
motion_reward = (
    1.0 * direction_score  # Direction only
)
```

Train and evaluate.

### 3.2 Ablation 2: Direction + Speed

```python
motion_reward = (
    0.5 * direction_score +
    0.5 * speed_score
)
```

### 3.3 Ablation 3: Full Motion Reward

```python
motion_reward = (
    0.4 * direction_score +
    0.4 * speed_score +
    0.2 * smoothness_score
)
```

### 3.4 Ablation 4: Varying Weights

Try different λ values for motion reward weight in the overall sum.

**Currently:** All rewards are summed equally.

**To weight motion reward:**

Edit `grpo_trainer.py` line 658:
```python
# Current (equal weighting):
rewards = rewards_per_func.sum(dim=1)

# Weighted (emphasize motion):
weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0], device=device)  
# [ans_acc, ans_tiou, ans_viou, thk_temporal_point, thk_temporal_segment, thk_spatial, motion_trajectory, format]
rewards = (rewards_per_func * weights).sum(dim=1)
```

---

## Phase 4: Analysis & Paper Writing (Day 6-7)

### 4.1 Results Analysis

1. **Quantitative:**
   - V-STAR benchmark scores
   - Ablation study table
   - Training curves (reward, KL divergence)

2. **Qualitative:**
   - Visualize generated evidence chains
   - Compare motion descriptions with/without motion reward
   - Failure case analysis

### 4.2 Key Experiments

| Experiment | Purpose |
|------------|---------|
| Baseline (Open-o3) | Establish ceiling with their rewards |
| +Motion Reward | Show motion-aware improvement |
| Direction only | Isolate direction contribution |
| Direction+Speed | Show speed adds value |
| Full (Dir+Speed+Smooth) | Best configuration |
| Varying λ_motion | Find optimal weight |

### 4.3 Expected Improvements

**Conservative estimate:**
- V-STAR mAM: +2-5% absolute
- V-STAR mLGM: +3-7% absolute
- Motion-heavy tasks (NExT-QA causal): +5-10%

**Justification:**
- Open-o3's spatial reward is frame-independent
- Motion reasoning requires trajectory-level understanding
- Our reward directly optimizes for motion properties

---

## Troubleshooting

### Issue: Motion reward always 0

**Debug:**
```python
# Add debug prints in motion_trajectory_reward
print(f"Sample {idx}: parsed {len(parsed_claims)} claims")
print(f"Pred bboxes: {len(pred_bboxes)}, GT bboxes: {len(gt_bboxes)}")
print(f"Direction: {direction_score:.3f}, Speed: {speed_score:.3f}, Smooth: {smoothness_score:.3f}")
```

**Common causes:**
1. Task type filter too restrictive → Check `task` field
2. Not enough temporal points → Need ≥2 claims
3. Bbox format mismatch → Check normalization
4. Missing GT data → Check `key_items` and `key_frames`

### Issue: Training unstable

**Solutions:**
1. Reduce motion reward weight
2. Clip motion reward to [0, 1]
3. Add warmup for motion reward (start at 0, ramp up)

### Issue: GPU OOM

**Solutions:**
1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable DeepSpeed ZeRO-3 (in `run_grpo_video.sh`)

---

## Success Criteria

✅ **Baseline works**: Can reproduce Open-o3 results  
✅ **Motion reward computes**: Non-zero rewards during training  
✅ **Training stable**: Loss decreases, KL stays bounded  
✅ **Improvement**: +2-5% on V-STAR with motion reward  
✅ **Ablation makes sense**: Direction alone helps, full is best  

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Setup & Baseline | 1 day | Baseline V-STAR scores |
| Add Motion Reward | 2 days | Training with motion reward |
| Evaluate | 1 day | Comparison table |
| Ablations | 2 days | Ablation study results |
| Analysis | 2 days | Paper draft |
| **Total** | **8 days** | **Complete system + paper** |

---

## Next Steps

1. ⏳ Download STGR dataset
2. ⏳ Run baseline training
3. ⏳ Implement motion reward integration
4. ⏳ Train with motion reward
5. ⏳ Evaluate and compare
6. ⏳ Run ablations
7. ⏳ Write paper

**Ready to start!** 🚀
