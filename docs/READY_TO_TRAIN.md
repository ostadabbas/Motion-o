# ✅ Ready to Train - Quick Guide

## Setup Complete

Your training environment is ready with the subset dataset!

### Dataset Status

✅ **Filtered datasets created:**
- **SFT**: 5,696 samples → `/mnt/data/stgr/json_data/STGR-SFT-subset.json`
- **RL**: 5,819 samples → `/mnt/data/stgr/json_data/STGR-RL-subset.json`

✅ **Videos extracted:**
- PLM videos: 2,421 files (18GB)
- Temporal grounding videos: 454 files (7.4GB)
- Total: ~2,875 videos ready

✅ **Configuration:**
- Data root: `/mnt/data/stgr`
- Environment: `dora_cuda`
- GPUs: 0,1,2,3 (4 GPUs)
- Effective batch size: 8 (1 per-device × 4 GPUs × 2 accum steps)

## Start Training Now

### Step 1: SFT Training (Required First)

```bash
cd /home/bi.ga/Workspace/vlmm-mcot
bash scripts/run_sft.sh
```

**What it does:**
- Downloads and trains Qwen2.5-VL-7B-Instruct
- Learns to produce spatio-temporal evidence chains
- Output: `outputs/sft_subset/`
- Time: ~4-6 hours on 4xGPU

**Monitor progress:**
```bash
# Watch logs
tail -f outputs/sft_subset/training.log

# Check tensorboard
tensorboard --logdir outputs/sft_subset --port 6006
```

### Step 2: RL Training (Choose One)

#### Option A: Motion-Aware GRPO (Our Contribution)

```bash
bash scripts/run_grpo_motion.sh
```

**Includes:**
- ✅ Motion trajectory reward (direction, speed, smoothness)
- ✅ Trajectory-level geometric metrics
- Output: `outputs/rl_motion_subset/`

#### Option B: Baseline GRPO (For Comparison)

```bash
bash scripts/run_grpo_baseline.sh
```

**Includes:**
- ✅ Standard spatial + temporal rewards
- ❌ NO motion trajectory reward
- Output: `outputs/rl_baseline_subset/`

**Time:** ~6-8 hours each on 4xGPU

### Step 3: Compare Results

After training both models, compare their performance on motion reasoning tasks:

```bash
# Baseline
python training/eval.py \
    --model outputs/rl_baseline_subset \
    --dataset /mnt/data/stgr/json_data/STGR-RL-subset.json

# Motion-aware
python training/eval.py \
    --model outputs/rl_motion_subset \
    --dataset /mnt/data/stgr/json_data/STGR-RL-subset.json
```

## Quick Training Status Check

```bash
# Check if training is running
ps aux | grep train_sft
ps aux | grep train_grpo

# Check GPU usage
nvidia-smi

# Check disk space
df -h /mnt/data
df -h outputs/
```

## Training Configuration

### Hardware (4 GPUs)
- GPUs: 0,1,2,3
- Per-device batch size: 1
- Gradient accumulation: 2 steps
- Effective batch size: 8
- Mixed precision: BF16
- DeepSpeed: ZeRO-2 (SFT), ZeRO-3 (GRPO)

### Hyperparameters

**SFT:**
- Learning rate: 1e-6
- Epochs: 1
- Max grad norm: 5
- Optimizer: AdamW

**GRPO:**
- Learning rate: 1e-6
- LR scheduler: Cosine
- Weight decay: 0.01
- Beta (KL penalty): 0.04
- Num generations: 4
- Max prompt length: 16,384
- Max completion length: 768

## Troubleshooting

### OOM (Out of Memory) Errors

If you hit OOM, reduce batch size:

```bash
# Edit the script to use smaller batch size
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \  # Increase to maintain effective batch size
```

### Slow Data Loading

The first epoch might be slow as videos are loaded and cached. Subsequent epochs should be faster.

### Check Dataset Samples

```bash
python -c "
import json
with open('/mnt/data/stgr/json_data/STGR-SFT-subset.json') as f:
    data = json.load(f)
print(f'SFT samples: {len(data)}')
print(f'First sample:')
print(json.dumps(data[0], indent=2)[:500])
"
```

### Verify Video Loading

```bash
python -c "
import json
from pathlib import Path

with open('/mnt/data/stgr/json_data/STGR-SFT-subset.json') as f:
    data = json.load(f)

sample = data[0]
video_path = sample['video_path_full']
print(f'Video path: {video_path}')
print(f'Exists: {Path(video_path).exists()}')
print(f'Size: {Path(video_path).stat().st_size / 1024 / 1024:.1f} MB')
"
```

## What's Different from Full Dataset?

**Current (18% subset):**
- 5,696 SFT samples
- 5,819 RL samples
- Sources: PLM, ActivityNet, COIN, DiDeMo, QVHighlights, QueryD
- Good for: Initial experiments, debugging, proof of concept

**Full dataset (100%):**
- 31,166 SFT samples
- 37,231 RL samples
- Additional sources: VideoR1, GQA, VideoEspresso, TimerFT, TreeVGR
- Good for: Final paper results, comprehensive evaluation

## Next Steps After Training

1. ✅ Verify motion reward is improving trajectory accuracy
2. ✅ Run ablation studies on lambda values
3. ✅ Evaluate on motion-specific benchmarks
4. ✅ Download full dataset for final results
5. ✅ Write paper comparing baseline vs motion-aware

## File Locations

```
/home/bi.ga/Workspace/vlmm-mcot/
├── scripts/
│   ├── run_sft.sh                    # Start here!
│   ├── run_grpo_motion.sh            # Our contribution
│   └── run_grpo_baseline.sh          # Comparison
├── training/
│   ├── train_sft.py                  # SFT implementation
│   ├── train_grpo.py                 # GRPO implementation
│   ├── motion_reward.py              # Our novel reward
│   └── reward_func.py                # Reward registry
├── configs/
│   ├── data_root.py                  # Points to /mnt/data/stgr
│   ├── zero2.json                    # DeepSpeed ZeRO-2
│   └── zero3.json                    # DeepSpeed ZeRO-3
└── outputs/
    ├── sft_subset/                   # SFT checkpoints
    ├── rl_motion_subset/             # Motion-aware RL
    └── rl_baseline_subset/           # Baseline RL

/mnt/data/stgr/
├── json_data/
│   ├── STGR-SFT-subset.json         # Filtered SFT data
│   └── STGR-RL-subset.json          # Filtered RL data
└── videos/
    └── stgr/
        ├── plm/videos/*.mp4         # 2,421 PLM videos
        └── temporal_grounding/      # 454 videos
```

## Ready to Go!

Everything is configured. Just run:

```bash
bash scripts/run_sft.sh
```

Good luck with training! 🚀
