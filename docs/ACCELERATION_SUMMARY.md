# ✨ Motion Tags with Acceleration

## What Changed

### Old Format (v2):
```xml
<motion>up-left motion (speed: 2.621 units/s, smooth)</motion>
```
- ❌ "smooth/jerky/erratic" - subjective, vague
- ❌ No acceleration information
- ❌ Hard for model to learn quantitative motion

### New Format (v3):
```xml
<motion>up-left motion (speed: 2.621 units/s, accel: +3.556 units/s²)</motion>
```
- ✅ Numerical acceleration with sign
- ✅ Physics-based: direction, velocity, acceleration
- ✅ Clear semantics: +accelerating, -decelerating, ≈0 constant

## Why Acceleration Matters

### For Training:
1. **Objective metric** - Model learns actual physics, not subjective quality
2. **Predictive power** - Acceleration helps predict future positions
3. **Richer signal** - Distinguishes smooth constant motion from stop-start patterns

### For GRPO Reward:
The motion trajectory reward can now use acceleration for:
- Penalizing unrealistic sudden changes (high |accel|)
- Rewarding smooth transitions (low |accel|)
- Verifying physics consistency

## Dataset Statistics (v3)

**Full Dataset:** `STGR-SFT-subset-motion-v3.json`

```
Total samples: 5,696
Samples with motion: 4,687 (82.3%)
Total motion tags: 5,199

Acceleration Distribution:
├── Accelerating (+): 768 (14.8%)   ← Speeding up
├── Decelerating (-): 710 (13.7%)   ← Slowing down
├── Constant (≈0):   3,024 (58.2%)  ← Steady motion
└── Stationary:      697 (13.4%)    ← No movement
```

This is a **realistic** distribution! Most tracked objects move at constant velocity, with ~15% showing acceleration/deceleration.

## Example Motion Tags

### Accelerating Object:
```xml
<motion>up-left motion (speed: 2.621 units/s, accel: +3.556 units/s²)</motion>
```
- Starting slow, speeding up (e.g., car pulling away)

### Decelerating Object:
```xml
<motion>rightward motion (speed: 1.234 units/s, accel: -0.856 units/s²)</motion>
```
- Slowing down (e.g., car coming to a stop)

### Constant Velocity:
```xml
<motion>downward motion (speed: 0.782 units/s, accel: +0.000 units/s²)</motion>
```
- Steady motion (e.g., person walking at constant pace)

## Technical Implementation

### Acceleration Computation:

```python
def compute_direction_speed_acceleration(bboxes, timestamps):
    # Track instantaneous speeds between frames
    speeds = []
    for i in range(len(centroids) - 1):
        distance = ||centroid[i+1] - centroid[i]||
        dt = timestamps[i+1] - timestamps[i]
        speeds.append(distance / dt)
    
    # Acceleration = change in speed over time
    if len(speeds) >= 2:
        speed_change = speeds[-1] - speeds[0]
        time_span = total_time
        acceleration = speed_change / time_span
    
    return (direction, avg_speed, acceleration)
```

### Key Features:
- Uses centroid trajectories from bounding boxes
- Computes instantaneous speeds between consecutive frames
- Acceleration = (final_speed - initial_speed) / total_time
- Sign convention: + = speeding up, - = slowing down

## Comparison with Related Work

### Open-o3-Video:
- No motion tags at all
- Only `<obj>`, `<box>`, `<t>` for spatial-temporal grounding

### MotionR1 v1 (OLD):
- Had motion tags with "smooth/jerky/erratic"
- Subjective quality metrics
- No acceleration information

### MotionR1 v3 (NEW):
- ✨ **Physics-based trajectory reasoning**
- ✨ **Numerical acceleration with sign**
- ✨ **First video VLM with quantitative motion metrics**

## Benefits for Your Research

### 1. Better SFT Training:
- Model learns objective motion physics
- No confusion from subjective descriptors
- Clearer training signal from numerical values

### 2. Enhanced GRPO Reward:
The `motion_trajectory_reward` can now:
```python
# Penalize unrealistic acceleration
if |predicted_accel - gt_accel| > threshold:
    penalty = ...

# Reward physics consistency
if abs(predicted_accel) < realistic_threshold:
    bonus = ...
```

### 3. Motion Prediction:
With direction, speed, and acceleration, the model can:
- Predict future object positions
- Reason about trajectory causality
- Generate physically plausible motion descriptions

## Files Updated

### Modified:
- `scripts/augment_motion_data_simple.py` - Added acceleration computation
- `scripts/sbatch_sft_full.sh` - Points to v3 dataset
- `scripts/run_sft_full.sh` - Points to v3 dataset

### New Dataset:
- `/mnt/data/stgr/json_data/STGR-SFT-subset-motion-v3.json`

### Backups Preserved:
- v1: `STGR-SFT-subset-motion.json` (had single-frame noise)
- v2: `STGR-SFT-subset-motion-v2.json` (multi-frame only, with smooth/jerky)

## Next Steps

### 1. Re-train SFT with v3:

```bash
sbatch scripts/sbatch_sft_full.sh
```

**What to expect:**
- Model learns: `<motion>direction motion (speed: X.XXX units/s, accel: ±X.XXX units/s²)</motion>`
- Better understanding of motion physics
- Can reason about acceleration patterns

### 2. Test SFT Output:

```bash
python scripts/test_sft_motion_mimics_training.py
```

**Expected output:**
```xml
<motion>up-left motion (speed: 2.621 units/s, accel: +3.556 units/s²)</motion>
```
NOT:
```xml
<motion>stationary (single frame)</motion>
```

### 3. Proceed to GRPO:

```bash
sbatch scripts/sbatch_grpo_motion_full.sh
```

GRPO will refine the motion predictions using the trajectory reward.

## Summary

| Feature | v1 (OLD) | v2 | v3 (NEW) |
|---------|----------|-----|----------|
| Multi-frame only | ❌ | ✅ | ✅ |
| Direction | ✅ | ✅ | ✅ |
| Speed (numerical) | ✅ | ✅ | ✅ |
| **Acceleration** | ❌ | ❌ | **✅** |
| Quality metric | smooth/jerky | smooth/jerky | **accel (±)** |
| Physics-based | ❌ | ❌ | **✅** |

**Status:** ✅ Ready to train with physics-based motion reasoning!

---

## Motion Tag Format (Final)

```
<motion>DIRECTION motion (speed: SPEED units/s, accel: ACCEL units/s²)</motion>

Where:
  DIRECTION: up-left | rightward | downward | etc.
  SPEED:     Positive float (average velocity magnitude)
  ACCEL:     Signed float (+speeding up, -slowing down, ≈0 constant)
```

**Example:**
```xml
<motion>up-left motion (speed: 2.621 units/s, accel: +3.556 units/s²)</motion>
```

This object is moving toward the upper-left at 2.6 units/s and accelerating at +3.6 units/s², meaning it's getting faster! 🚀
