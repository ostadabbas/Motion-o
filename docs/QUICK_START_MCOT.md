# 🚀 Quick Start: Motion Chain of Thought (MCoT)

## Run This Now! (Quick Test - 10 samples)

```bash
cd /home/bi.ga/Workspace/vlmm-mcot

# Step 1: SFT with MCoT (5-10 minutes)
bash scripts/run_sft_mcot.sh

# Step 2: GRPO with MCoT (10-15 minutes)
bash scripts/run_grpo_mcot.sh
```

That's it! The system is ready to run.

---

## What Just Happened?

### ✅ Files Created (5 new)
1. `src/motion_text.py` - Converts motion metrics → natural language
2. `scripts/augment_motion_data_simple.py` - Data augmentation (no dependencies)
3. `scripts/test_motion_tokenization.py` - Tokenizer test
4. `scripts/run_sft_mcot.sh` - SFT training with MCoT
5. `scripts/run_grpo_mcot.sh` - GRPO training with MCoT

### ✅ Files Modified (1 only!)
- `training/data_loader.py` - Added motion reasoning to system prompts

### ✅ Data Prepared
- 10 test samples with `<motion>` tags ready at:
  - `/mnt/data/stgr/json_data/STGR-SFT-motion-test.json`

---

## What is MCoT?

**Traditional video reasoning**:
```
<think>
Object at T1, Object at T2
Therefore, object moved...
</think>
```

**Motion Chain of Thought** (our innovation):
```
<think>
Object at T1: <obj>man</obj><box>[...]</box>at<t>47.5</t>s
Object at T2: <obj>man</obj><box>[...]</box>at<t>54.2</t>s
<motion>leftward motion (speed: 0.004 units/s, smooth)</motion>
Therefore, man moved left smoothly...
</think>
```

**Key insight**: Explicit motion reasoning in structured format!

---

## Verify It Worked

### Check Augmented Data
```bash
python3 -c "
import json
data = json.load(open('/mnt/data/stgr/json_data/STGR-SFT-motion-test.json'))
print(f'Samples: {len(data)}')
print(f'First sample has motion tags: {\"<motion>\" in data[0].get(\"reasoning_process\", \"\")}')
print(f'Sample reasoning preview:')
print(data[0]['reasoning_process'][:400])
"
```

### Check Training Scripts
```bash
ls -lh scripts/run_*_mcot.sh
```

### Check System Prompt
```bash
grep -A 2 "motion with" training/data_loader.py
```

---

## What to Expect During Training

### SFT (5-10 minutes)
- Loss: Should decrease from ~5.5 → ~5.0
- Output: `outputs/sft_mcot_test/`
- Model learns: `<motion>` tag format

### GRPO (10-15 minutes)
- Motion rewards: Should be > 0.0
- Output: `outputs/grpo_mcot_test/`
- Model improves: Motion description accuracy

---

## After Quick Test Succeeds

### Create Full Datasets
```bash
# SFT dataset (31,166 samples - takes ~10 minutes)
python3 scripts/augment_motion_data_simple.py \
    --input /mnt/data/stgr/json_data/STGR-SFT.json \
    --output /mnt/data/stgr/json_data/STGR-SFT-motion.json

# RL dataset (37,231 samples - takes ~15 minutes)
python3 scripts/augment_motion_data_simple.py \
    --input /mnt/data/stgr/json_data/STGR-RL.json \
    --output /mnt/data/stgr/json_data/STGR-RL-motion.json
```

### Update Scripts for Full Training
```bash
# Edit scripts/run_sft_mcot.sh
# Line 32: DATASET_JSON="${DATA_ROOT}/json_data/STGR-SFT-motion.json"
# Line 18: export QUICK_TEST="false"

# Edit scripts/run_grpo_mcot.sh
# Line 31: DATASET_JSON="${DATA_ROOT}/json_data/STGR-RL-motion.json"
# Line 17: export QUICK_TEST="false"
```

### Run Full Training
```bash
bash scripts/run_sft_mcot.sh    # ~21 hours
bash scripts/run_grpo_mcot.sh   # ~24 hours
```

---

## Compare with Baseline

### Run Baseline (Without MCoT)
```bash
bash scripts/run_sft.sh         # Original SFT
bash scripts/run_grpo_motion.sh # Original GRPO
```

### Compare Results
- Motion reward scores: MCoT vs. Baseline
- Answer accuracy: Should be similar or better
- Reasoning quality: Check if motion descriptions help

---

## Debug Commands

```bash
# Check GPU status
nvidia-smi

# Check training progress
tail -f outputs/sft_mcot_test/logs/train.log

# Check data location
ls -lh /mnt/data/stgr/json_data/*motion*.json

# Check augmentation quality
python3 scripts/augment_motion_data_simple.py \
    --input /mnt/data/stgr/json_data/STGR-SFT-tracking-samples.json \
    --output /tmp/test.json \
    --inspect 3
```

---

## Key Files

### Code
- Motion text: `src/motion_text.py`
- Augmentation: `scripts/augment_motion_data_simple.py`
- SFT script: `scripts/run_sft_mcot.sh`
- GRPO script: `scripts/run_grpo_mcot.sh`

### Data
- Test (10): `/mnt/data/stgr/json_data/STGR-SFT-motion-test.json`
- Full (TBD): `/mnt/data/stgr/json_data/STGR-SFT-motion.json`

### Docs
- Full details: `MCOT_IMPLEMENTATION.md`
- Status: `MCOT_READY.md`
- This guide: `QUICK_START_MCOT.md`

---

## One-Line Summary

**MCoT = Temporal-Spatial Grounding + Explicit Motion Tags in Chain-of-Thought**

🎯 Ready to run? → `bash scripts/run_sft_mcot.sh`
