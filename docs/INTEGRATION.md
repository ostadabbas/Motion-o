# Integration Guide: Motion-Aware Rewards → Open-o3 Video

Step-by-step guide to integrate motion-aware trajectory rewards into Open-o3 Video.

## Overview

Open-o3 Video provides:
- STGR dataset with spatial-temporal annotations
- Two-stage training (SFT + RL with GSPO)
- Infrastructure for grounded video reasoning

We add:
- **Motion-aware reward function** (trajectory-level)
- **Think-Predict bbox refinement**
- **Geometric motion metrics** (direction, speed, smoothness)

## Step-by-Step Integration

### 1. Clone and Setup Open-o3 Video

```bash
git clone https://github.com/marinero4972/Open-o3-Video
cd Open-o3-Video
conda create -n open-o3-video python=3.11
conda activate open-o3-video
bash setup.sh
```

### 2. Download STGR Dataset

Follow their data preparation instructions:
- Download STGR-SFT-30k (for cold-start)
- Download STGR-RL-36k (for RL training)
- Set DATA_ROOT in `src/r1-v/configs/data_root.py`

### 3. Copy Motion Reward Modules

```bash
# From your vlmm-mcot directory:
cp src/motion_metrics.py ../Open-o3-Video/src/
cp src/geometric_reward.py ../Open-o3-Video/src/
cp src/evidence_parser.py ../Open-o3-Video/src/
```

### 4. Modify RL Training Script

Edit `Open-o3-Video/src/scripts/run_grpo_video.sh` or the underlying Python script:

**Before (Open-o3's static spatial reward):**
```python
# In their RL reward computation:
reward = spatial_iou + temporal_iou + text_similarity
```

**After (Motion-aware trajectory reward):**
```python
from motion_metrics import compute_trajectory_iou, compute_direction_similarity
from geometric_reward import compute_geometric_reward

# Compute reward with motion awareness:
reward = compute_geometric_reward(
    pred_steps=parsed_predictions,
    gt_steps=ground_truth_evidence,
    frames=video_frames,
    fps=video_fps,
    lambda_spatial=0.25,   # Spatial IoU (per-frame)
    lambda_temporal=0.15,  # Temporal IoU
    lambda_motion=0.35,    # NEW: Trajectory-level motion
    lambda_caption=0.20    # Text similarity
)
```

### 5. Update Prompt Template

Modify the system prompt to include think/predict bboxes:

**Add to prompt:**
```
For each evidence step, provide:
1. Temporal interval [t_s–t_e]
2. Think bbox: Initial rough estimate (x1,y1),(x2,y2)
3. Predict bbox: Refined prediction (x1,y1),(x2,y2)
4. Motion: Quantitative motion descriptors (velocity, direction)
5. Description: What happened

Example:
Step 1: [0.0s–2.5s] Person picks up ball
  Think: (300,400),(500,700)
  Predict: (320,420),(480,680)
  Motion: velocity: 150px/s, direction: downward
  Description: Person reaches down and picks up the ball
```

### 6. Update Parser

Replace their evidence parser with ours:

```python
from evidence_parser import parse_think_predict_chain

# Parse model output:
steps = parse_think_predict_chain(model_output_text)

# Extract think and predict bboxes:
for step in steps:
    think_bbox = step.think_bboxes[0]  # Initial estimate
    pred_bbox = step.pred_bboxes[0]    # Refined prediction
    motion_text = step.motion_text
```

### 7. Run SFT Cold-Start

Use their SFT training as-is (no changes needed for cold-start):

```bash
cd Open-o3-Video
bash ./src/scripts/run_sft_video.sh
```

### 8. Run Motion-Aware RL Training

Now run RL with motion-aware rewards:

```bash
bash ./src/scripts/run_grpo_video.sh
```

Monitor logs for motion reward components:
```
Epoch 1, Step 100:
  R_spatial: 0.65
  R_temporal: 0.72
  R_motion: 0.58      ← Motion-aware trajectory score
    - Direction: 0.82
    - Speed: 0.61
    - Smoothness: 0.31
  R_caption: 0.44
  Total: 0.612
```

## Expected Modifications to Open-o3 Code

### File: `src/rl_trainer.py` (or equivalent)

**Add imports:**
```python
from motion_metrics import (
    compute_trajectory_iou,
    compute_direction_similarity,
    compute_speed_fidelity,
    compute_trajectory_smoothness
)
from geometric_reward import compute_geometric_reward
```

**Replace reward function:**
```python
def compute_reward(self, pred_text, gt_evidence, frames, fps):
    # Parse predictions
    from evidence_parser import parse_think_predict_chain
    pred_steps = parse_think_predict_chain(pred_text)
    
    # Compute motion-aware geometric reward
    reward = compute_geometric_reward(
        pred_steps=pred_steps,
        gt_steps=gt_evidence,
        frames=frames,
        fps=fps,
        lambda_spatial=self.config.lambda_spatial,
        lambda_temporal=self.config.lambda_temporal,
        lambda_motion=self.config.lambda_motion,  # NEW
        lambda_caption=self.config.lambda_caption
    )
    
    return reward
```

### File: `src/configs/training_config.py`

**Add hyperparameters:**
```python
lambda_spatial = 0.25
lambda_temporal = 0.15
lambda_motion = 0.35      # NEW: Motion-aware trajectory weight
lambda_caption = 0.20
```

## Evaluation

Use Open-o3's evaluation scripts with motion-focused benchmarks:

```bash
cd eval
bash ./scripts/eval_vstar.sh        # Open-o3's benchmark
bash ./scripts/eval_nextqa.sh       # Add NExT-QA causal/temporal
bash ./scripts/eval_groundmore.sh   # Add GroundMoRe (if available)
```

## Ablation Studies

Test motion reward contribution:

1. **Baseline**: λ_motion=0.0 (Open-o3's original reward)
2. **+Direction**: λ_motion=0.35, only direction matching
3. **+Speed**: Add speed fidelity
4. **+Smoothness**: Add trajectory smoothness
5. **Full**: All motion components

Expected improvements on motion reasoning tasks: +5-10% accuracy.

## Troubleshooting

**Issue**: Reward is always negative
- Check bbox coordinates are normalized [0,1]
- Verify FPS is correct
- Debug with `--debug-reward` flag

**Issue**: Motion reward is 0
- Ensure GT evidence has multiple temporal steps
- Check that bboxes exist for consecutive frames
- Verify displacement vectors are non-zero

**Issue**: Training unstable
- Reduce λ_motion from 0.35 to 0.20
- Increase KL penalty
- Use gradient clipping

## Next Steps

1. Train on STGR-RL-36k with motion rewards
2. Evaluate on V-STAR + motion-heavy benchmarks
3. Ablate motion reward components
4. Compare against Open-o3 baseline
5. Analyze failure cases (where motion-aware helps vs hurts)
