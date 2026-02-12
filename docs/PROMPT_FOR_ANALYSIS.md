# Prompt for Training Analysis

Copy and paste this prompt when SFT and GRPO training complete:

---

## Prompt:

```
I'm working on MotionR1, which adds motion reasoning to video VLMs using explicit <motion> tags with direction, speed, and acceleration.

Training pipeline just completed:
- Job 639: SFT on motion-v3 dataset (5,696 samples)
- Job 640: GRPO with motion_trajectory_reward

**Goal:** Analyze if the model learned to generate motion tags correctly and if the motion reward is working.

**SFT Analysis Tasks:**
1. Check if SFT training converged (loss trends from logs/sft_full_639.out)
2. Test the SFT model on sample #5283 (strong motion example) - should output: 
   <motion>up-left motion (speed: 2.621 units/s, accel: +3.556 units/s²)</motion>
   NOT: <motion>stationary (single frame)</motion>
3. Verify checkpoint exists at outputs/sft_motion_full/

**GRPO Analysis Tasks:**
1. Check GRPO training stability (logs/grpo_motion_full_640.out)
2. Verify motion_trajectory_reward is non-zero and improving
3. Check KL divergence stays under control (< 10)
4. Compare epoch 0 vs final epoch: reward trends, motion reward specifically

**Context:**
- Dataset v3: Multi-frame trajectories only, with acceleration (86.6% actual motion)
- Fixed LR: 1e-6 (not 1e-4 that caused collapse before)
- Motion reward: 0.4*direction + 0.4*speed + 0.2*smoothness

**Logs to analyze:**
@logs/sft_full_639.out
@logs/grpo_motion_full_640.out

**Model to test:**
@outputs/sft_motion_full/

**Test script:**
@scripts/test_sft_motion_mimics_training.py

Start by checking SFT convergence, then test the model, then analyze GRPO metrics.
```

---

## Alternative Shorter Version:

```
MotionR1 training completed. Need to analyze:

1. **SFT (Job 639):** Did it learn to generate motion tags with acceleration?
   - Check: @logs/sft_full_639.out (loss convergence)
   - Test: python scripts/test_sft_motion_mimics_training.py
   - Expected: <motion>up-left motion (speed: X, accel: +X)</motion>

2. **GRPO (Job 640):** Is motion_trajectory_reward working?
   - Check: @logs/grpo_motion_full_640.out
   - Metrics: motion_trajectory_reward trend, KL < 10, reward improvement

Dataset: motion-v3 (86.6% actual motion, acceleration-based)
Fixed: LR=1e-6, multi-frame only

Analyze logs and test model quality.
```

---

## Key Files to Reference:

- Logs: `logs/sft_full_639.out`, `logs/grpo_motion_full_640.out`
- Model: `outputs/sft_motion_full/`
- Test: `scripts/test_sft_motion_mimics_training.py`
- Dataset: `/mnt/data/stgr/json_data/STGR-SFT-subset-motion-v3.json`

## Success Criteria:

**SFT Success:**
- ✅ Loss decreases and converges
- ✅ Model outputs motion tags with acceleration values
- ✅ Not defaulting to "stationary" for moving objects

**GRPO Success:**
- ✅ `rewards/motion_trajectory_reward` > 0 and increasing
- ✅ KL divergence < 10 (stable training)
- ✅ Overall reward improves over epochs
- ✅ No reward collapse like Job 634

## What to Look For in Logs:

**SFT Log (`sft_full_639.out`):**
```
Epoch X | Loss: ~1.2-1.5 (should decrease)
Training Loss: (should decrease over steps)
```

**GRPO Log (`grpo_motion_full_640.out`):**
```
rewards/motion_trajectory_reward: X.XX (should be > 0)
rewards/kl: X.XX (should be < 10)
rewards/chosen: X.XX (should increase)
```

## Commands Ready to Run:

```bash
# Check SFT convergence
grep "Training Loss" logs/sft_full_639.out | tail -20

# Test SFT model
conda activate dora_cuda
python scripts/test_sft_motion_mimics_training.py

# Check GRPO motion reward
grep "motion_trajectory_reward" logs/grpo_motion_full_640.out

# Check GRPO KL divergence
grep "rewards/kl" logs/grpo_motion_full_640.out
```
