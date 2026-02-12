# Motion Reward Fix for Single-Frame GT

## Problem
STGR-RL dataset has 0% multi-frame GT, but motion_trajectory_reward needed multi-frame GT to compute direction/speed scores against ground truth.

## Solution
Modified `motion_trajectory_reward` to work in **two modes**:

### Mode 1: Multi-Frame GT (STGR-SFT-motion-v3)
- Full trajectory evaluation
- Direction similarity (cosine of displacement vectors)
- Speed fidelity (velocity magnitude matching)
- Smoothness penalty

### Mode 2: Single-Frame GT (STGR-RL) ⭐ NEW
Rewards based on model's predictions alone:

1. **Physical Plausibility** (40%): 
   - Smooth motion from model's trajectory
   - Penalizes jerky/implausible jumps
   - Uses trajectory_smoothness_penalty

2. **Spatial Consistency** (30%):
   - Model's predictions should be near GT anchor frame
   - Normalized distance to GT centroid
   - Tolerates motion within 50% image diagonal

3. **Motion Diversity** (30%):
   - **Penalizes identical bboxes** (encourages tracking)
   - Rewards 1-20% image diagonal displacement (realistic motion)
   - Penalizes excessive motion (>30% diagonal = implausible)

## Key Insight
Model generates multiple temporal observations even when GT has 1 frame.
We can evaluate motion quality from the model's OWN predictions!

## Benefits
- ✅ Works with STGR-RL (proper GRPO dataset)
- ✅ Follows Open-o3-Video paradigm (SFT vs RL data)
- ✅ Encourages model to predict DIFFERENT bboxes (not copy-paste)
- ✅ Rewards physically plausible motion

## Files Modified
- `training/motion_reward.py`: Added `_compute_plausibility_reward()`
- `scripts/sbatch_grpo_motion_full.sh`: Back to STGR-RL-subset.json

Ready to train!
