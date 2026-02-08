# Motion Chain of Thought (MCoT) Implementation

## Overview

This document describes the implementation of **Motion Chain of Thought (MCoT)** - a new reasoning pathway for video VLMs that explicitly models object trajectories through structured `<motion>` tags in the chain-of-thought reasoning process.

**Status**: ✅ Implementation Complete - Ready for Testing

**Philosophy**: Keep Open-o3 Video codebase intact, add minimal new components

---

## What is Motion Chain of Thought?

Traditional Chain-of-Thought for video understanding:
```xml
<think>
  Object at time T1: <obj>man</obj><box>[0.46,0.43,0.77,0.94]</box>at<t>47.5</t>s
  Object at time T2: <obj>man</obj><box>[0.41,0.38,0.77,0.99]</box>at<t>54.2</t>s
  Therefore, the man is moving...
</think>
```

**Motion Chain of Thought** adds explicit motion reasoning:
```xml
<think>
  Object at time T1: <obj>man</obj><box>[0.46,0.43,0.77,0.94]</box>at<t>47.5</t>s
  Object at time T2: <obj>man</obj><box>[0.41,0.38,0.77,0.99]</box>at<t>54.2</t>s
  <motion>leftward motion (speed: 0.004 units/s, smooth)</motion>
  Therefore, the man is moving left smoothly...
</think>
```

This creates a **structured reasoning pathway** where:
1. Temporal observations are made (existing Open-o3 format)
2. Motion is explicitly computed and described (new MCoT addition)
3. Final reasoning incorporates motion understanding

---

## Implementation Components

### Files Created (5 new files)

1. **`src/motion_text.py`** - Motion metrics → natural language converter
   - Converts displacement vectors to direction (leftward/rightward/upward/downward)
   - Computes average speed from bbox trajectories
   - Assesses trajectory smoothness (smooth/jerky/erratic)

2. **`scripts/augment_motion_data.py`** - Full data preprocessing pipeline
   - Original version (requires scipy - for production use)

3. **`scripts/augment_motion_data_simple.py`** - Simplified standalone version
   - No external dependencies beyond json
   - Computes motion inline (for environments with library conflicts)

4. **`scripts/test_motion_tokenization.py`** - Tokenizer compatibility test
   - Verifies `<motion>` tags work with Qwen2.5-VL tokenizer

5. **`scripts/run_sft_mcot.sh`** - SFT training with MCoT
   - Uses motion-augmented STGR-SFT-motion-test.json (10 samples)
   - Quick test mode enabled by default

6. **`scripts/run_grpo_mcot.sh`** - GRPO training with MCoT
   - Uses motion-augmented STGR-RL-motion-test.json (10 samples)
   - Includes motion_trajectory reward function
   - Loads from MCoT SFT checkpoint

### Files Modified (1 file only!)

1. **`training/data_loader.py`** - System prompt updates
   - Added motion reasoning guidance to 3 video task prompts:
     - "temporal-spatial free-form QA"
     - "General video QA MCQ"
     - "General video QA Free-form"
   - Change: ~30 words added per prompt instructing model to use `<motion>` tags

### Files Unchanged (All core code intact)

- ✅ `training/grpo_trainer.py` (858 lines)
- ✅ `training/train_grpo.py` (142 lines)
- ✅ `training/motion_reward.py` (264 lines)
- ✅ `src/motion_metrics.py` (641 lines)
- ✅ All model architecture code
- ✅ All reward computation functions
- ✅ All DeepSpeed configurations

---

## Data Augmentation Process

### Step 1: Extract Tracking Samples

```bash
# SFT dataset: Find 10 samples with temporal-spatial tracking
python3 << 'EOF'
import json

with open('/mnt/data/stgr/json_data/STGR-SFT.json') as f:
    data = json.load(f)

tracking_samples = []
for item in data:
    if (item.get('task') in ["temporal-spatial free-form QA", ...] and
        item.get('key_items') and 
        len(item.get('key_frames', [])) >= 2):
        tracking_samples.append(item)
        if len(tracking_samples) >= 10:
            break

with open('/mnt/data/stgr/json_data/STGR-SFT-tracking-samples.json', 'w') as f:
    json.dump(tracking_samples, f, indent=2)
EOF
```

### Step 2: Augment with Motion Tags

```bash
python3 scripts/augment_motion_data_simple.py \
    --input /mnt/data/stgr/json_data/STGR-SFT-tracking-samples.json \
    --output /mnt/data/stgr/json_data/STGR-SFT-motion-test.json \
    --inspect 3
```

**Results**:
- ✅ 10/10 samples augmented successfully
- ✅ Motion tags inserted after temporal-spatial claims
- ✅ Verified format: `<motion>direction motion (speed: X units/s, quality)</motion>`

### Example Augmented Sample

**Before**:
```
<obj>man</obj><box>[0.46, 0.43, 0.77, 0.94]</box>at<t>47.5</t>s, who is clearly 
recognizable as Obama. Further confirmation is seen as the same 
<obj>man</obj><box>[0.41, 0.38, 0.77, 0.99]</box>at<t>54.2</t>s smiles while...
```

**After**:
```
<obj>man</obj><box>[0.46, 0.43, 0.77, 0.94]</box>at<t>47.5</t>s, who is clearly 
recognizable as Obama. Further confirmation is seen as the same 
<obj>man</obj><box>[0.41, 0.38, 0.77, 0.99]</box>at<t>54.2</t>s<motion>leftward 
motion (speed: 0.004 units/s, smooth)</motion> smiles while...
```

---

## Training Pipeline

### Phase 1: SFT with MCoT (Quick Test)

```bash
bash scripts/run_sft_mcot.sh
```

**Configuration**:
- Model: Qwen/Qwen2.5-VL-7B-Instruct
- Dataset: 10 motion-augmented samples
- Output: `outputs/sft_mcot_test/`
- GPUs: 4x A100 32GB
- DeepSpeed ZeRO-2
- QUICK_TEST=true (enabled by default)

**Expected Behavior**:
- Model learns to generate `<motion>` tags after temporal-spatial claims
- Loss should decrease (validates format learning)
- Checkpoint saved for RL training

### Phase 2: GRPO with MCoT + Motion Rewards

```bash
bash scripts/run_grpo_mcot.sh
```

**Configuration**:
- Model: `outputs/sft_mcot_test/` (from Phase 1)
- Dataset: 10 motion-augmented RL samples
- Output: `outputs/grpo_mcot_test/`
- Rewards: **motion_trajectory** + all standard rewards
- LoRA enabled (r=16, alpha=32)

**Expected Behavior**:
- Model generates `<motion>` tags
- Motion reward scores should be > 0.0 for tracking samples
- Motion descriptions should align with ground truth trajectories

---

## Testing Commands

### 1. Tokenization Test
```bash
python3 scripts/test_motion_tokenization.py
```
Verifies `<motion>` tags don't break Qwen2.5-VL tokenizer.

### 2. Data Augmentation Test
```bash
# Test on 10 samples
python3 scripts/augment_motion_data_simple.py \
    --input /mnt/data/stgr/json_data/STGR-SFT-tracking-samples.json \
    --output /tmp/test_augmented.json \
    --inspect 3
```

### 3. SFT Quick Test
```bash
# 10 samples, ~5-10 minutes
bash scripts/run_sft_mcot.sh
```

### 4. GRPO Quick Test
```bash
# 10 samples, ~10-15 minutes
bash scripts/run_grpo_mcot.sh
```

---

## Motion Tag Format Specification

### Syntax
```xml
<motion>direction motion (speed: X.XXX units/s, quality)</motion>
```

### Components

1. **Direction**: 
   - Single axis: `leftward`, `rightward`, `upward`, `downward`
   - Diagonal: `up-left`, `up-right`, `down-left`, `down-right`
   - Stationary: `stationary`

2. **Speed**: 
   - Format: `X.XXX units/s` (3 decimal places)
   - Units: Normalized coordinates per second
   - Example: `0.004 units/s` = very slow, `0.150 units/s` = fast

3. **Quality**:
   - `smooth`: Low acceleration variance (CoV < 0.3)
   - `jerky`: Moderate acceleration variance (0.3 ≤ CoV < 0.7)
   - `erratic`: High acceleration variance (CoV ≥ 0.7)

### Special Cases

- **Single frame**: `stationary (single frame)`
- **No movement**: `stationary (no significant motion)`
- **Insufficient data**: `stationary (insufficient data)`

---

## Validation Results

### Data Augmentation (10 samples)
- ✅ Successfully augmented: 10/10
- ✅ Format correct: All `<motion>` tags properly inserted
- ✅ Motion metrics: Direction, speed, smoothness computed

### Example Motion Tags Generated
1. `leftward motion (speed: 0.004 units/s, smooth)` - Slow left movement
2. `stationary (single frame)` - Object appears once
3. `leftward motion (speed: 0.001 units/s, smooth)` - Very slow drift

### System Prompt Update
✅ Updated 3 video task prompts in `training/data_loader.py`:
- "temporal-spatial free-form QA"
- "General video QA MCQ"  
- "General video QA Free-form"

Added guidance: *"After describing multiple temporal observations of the same object, describe its motion with `<motion>description</motion>` tags..."*

---

## Next Steps

### For Full Training (After Validation)

1. **Create Full Augmented Datasets**:
```bash
# SFT dataset (31,166 samples)
python3 scripts/augment_motion_data_simple.py \
    --input /mnt/data/stgr/json_data/STGR-SFT.json \
    --output /mnt/data/stgr/json_data/STGR-SFT-motion.json

# RL dataset (37,231 samples)
python3 scripts/augment_motion_data_simple.py \
    --input /mnt/data/stgr/json_data/STGR-RL.json \
    --output /mnt/data/stgr/json_data/STGR-RL-motion.json
```

2. **Update Training Scripts**:
   - Modify `scripts/run_sft_mcot.sh`: Change dataset to `STGR-SFT-motion.json`
   - Modify `scripts/run_grpo_mcot.sh`: Change dataset to `STGR-RL-motion.json`
   - Set `export QUICK_TEST="false"`

3. **Full Training Run**:
```bash
# SFT (~21 hours on 4x A100)
bash scripts/run_sft_mcot.sh

# GRPO (~24 hours on 4x A100)
bash scripts/run_grpo_mcot.sh
```

---

## Comparison: Baseline vs. MCoT

| Aspect | Baseline (Open-o3) | MCoT (Our Contribution) |
|--------|-------------------|------------------------|
| **Temporal grounding** | ✅ Yes | ✅ Yes (preserved) |
| **Spatial grounding** | ✅ Yes | ✅ Yes (preserved) |
| **Motion reasoning** | ❌ Implicit | ✅ **Explicit with `<motion>` tags** |
| **Motion reward** | ✅ Yes (exists) | ✅ Yes (same reward function) |
| **Training data** | Original STGR | Augmented with motion tags |
| **System prompt** | Standard | Enhanced with motion guidance |
| **Code changes** | N/A | 1 file modified, 5 files added |

**Key Innovation**: We teach the model to **articulate** motion reasoning explicitly in its chain-of-thought, rather than only rewarding correct motion implicitly.

---

## File Locations

### Data Files
- Original: `/mnt/data/stgr/json_data/STGR-SFT.json`
- Test samples: `/mnt/data/stgr/json_data/STGR-SFT-tracking-samples.json`
- Augmented test: `/mnt/data/stgr/json_data/STGR-SFT-motion-test.json`
- Augmented full (TBD): `/mnt/data/stgr/json_data/STGR-SFT-motion.json`

### Training Scripts
- MCoT SFT: `scripts/run_sft_mcot.sh`
- MCoT GRPO: `scripts/run_grpo_mcot.sh`
- Baseline SFT: `scripts/run_sft.sh` (unchanged)
- Baseline GRPO: `scripts/run_grpo_motion.sh` (unchanged)

### Source Code
- Motion text generator: `src/motion_text.py`
- Augmentation (full): `scripts/augment_motion_data.py`
- Augmentation (simple): `scripts/augment_motion_data_simple.py`
- Tokenization test: `scripts/test_motion_tokenization.py`

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Video Input + Question                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Qwen2.5-VL-7B (Base Model)                     │
│           + LoRA Adapters (for RL fine-tuning)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Chain-of-Thought Output                    │
│  <think>                                                    │
│    Temporal: <obj>X</obj><box>...</box>at<t>T1</t>s        │
│    Temporal: <obj>X</obj><box>...</box>at<t>T2</t>s        │
│    Motion: <motion>direction (speed: X, quality)</motion>   │  ← MCoT
│    Reasoning: ...                                           │
│  </think>                                                   │
│  <answer>...</answer>                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Reward Computation                       │
│  - Answer accuracy                                          │
│  - Temporal IoU                                             │
│  - Spatial IoU (Visual)                                     │
│  - Motion trajectory  ← Evaluates <motion> tags             │
│  - Format correctness                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Issue: NumPy/SciPy compatibility error
**Solution**: Use `augment_motion_data_simple.py` instead of `augment_motion_data.py`

### Issue: Tokenization warnings
**Solution**: Normal - Qwen tokenizer treats `<motion>` as regular text tokens

### Issue: No samples augmented
**Cause**: Dataset samples don't have `key_items` with 2+ `key_frames`
**Solution**: Use tracking samples extracted with the provided script

### Issue: GRPO fails to load SFT checkpoint
**Cause**: SFT training not completed or checkpoint path incorrect
**Solution**: Run `bash scripts/run_sft_mcot.sh` first, verify output in `outputs/sft_mcot_test/`

---

## Summary

**Motion Chain of Thought (MCoT)** successfully implemented with:
- ✅ 5 new files created
- ✅ 1 file modified (data_loader.py only)
- ✅ 10 test samples augmented and verified
- ✅ Training scripts ready for quick testing
- ✅ Original Open-o3 Video codebase intact
- ✅ Minimal intervention philosophy maintained

**Ready for**: Quick testing (10 samples) and full training validation

**Innovation**: First structured motion reasoning pathway for video VLMs via explicit `<motion>` tags in chain-of-thought.
