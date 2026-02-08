# Quick Start: Training with Subset Dataset

## Dataset Status

✅ **Successfully filtered dataset with available videos:**
- **SFT Training**: 5,696 samples (18.3% of full dataset)
- **RL Training**: 5,819 samples (15.6% of full dataset)
- **Sources**: PLM (3,168), ActivityNet (891), COIN (670), DiDeMo (629), QVHighlights (357), QueryD (104)

## Environment

Using conda environment: `dora_cuda`

All scripts automatically activate this environment.

## Training Steps

### 1. Supervised Fine-Tuning (SFT)

Train the base model to produce spatio-temporal evidence chains:

```bash
cd /home/bi.ga/Workspace/vlmm-mcot
bash scripts/run_sft.sh
```

**What it does:**
- Trains Qwen2.5-VL-7B on 5,696 samples with evidence chains
- Teaches basic bbox prediction and temporal grounding
- Output: `outputs/sft_baseline/` checkpoint

**Expected time:** ~4-6 hours on 8xA100 (40GB)

### 2a. Baseline GRPO (without motion reward)

Train with Open-o3's original reward (spatial + temporal + format):

```bash
bash scripts/run_grpo_baseline.sh
```

**What it does:**
- Uses static spatial IoU reward (frame-independent)
- Output: `outputs/grpo_baseline/` checkpoint

**Expected time:** ~6-8 hours on 8xA100

### 2b. Motion-Aware GRPO (with our novel reward)

Train with motion trajectory reward:

```bash
bash scripts/run_grpo_motion.sh
```

**What it does:**
- Adds motion trajectory reward (direction, speed, smoothness)
- Uses `training/motion_reward.py` for trajectory-level evaluation
- Output: `outputs/grpo_motion/` checkpoint

**Expected time:** ~6-8 hours on 8xA100

### 3. Evaluation

Compare the two RL checkpoints:

```bash
# Baseline
python training/eval.py --model outputs/grpo_baseline --dataset /mnt/data/stgr/json_data/STGR-RL-subset.json

# Motion-aware
python training/eval.py --model outputs/grpo_motion --dataset /mnt/data/stgr/json_data/STGR-RL-subset.json
```

## Training Configuration

### Hardware Requirements

**Minimum:**
- 8x A100 (40GB) GPUs
- 512GB RAM
- 500GB disk space

**Recommended:**
- 8x A100 (80GB) GPUs
- 1TB RAM
- 1TB disk space

### DeepSpeed Configuration

Using ZeRO-2 with:
- Gradient accumulation: 2 steps
- Effective batch size: 64 (8 GPUs × 4 per-device × 2 accum)
- Mixed precision: BF16

See `configs/ddp.yaml` for details.

## Monitoring Training

### Tensorboard

```bash
tensorboard --logdir outputs/ --port 6006
```

Then open: http://localhost:6006

### Key Metrics to Watch

**SFT:**
- Loss convergence (should drop below 1.0)
- Bbox format compliance (should be > 95%)
- Temporal grounding accuracy

**GRPO:**
- Reward curves (baseline vs motion-aware)
- KL divergence (keep < 0.1)
- Policy improvement rate

## Troubleshooting

### OOM Errors

Reduce batch size in training scripts:
```bash
--per_device_train_batch_size 2  # instead of 4
--gradient_accumulation_steps 4   # instead of 2
```

### Slow Data Loading

Pre-cache keyframes:
```bash
python scripts/cache_keyframes.py --dataset /mnt/data/stgr/json_data/STGR-SFT-subset.json
```

### Checkpoint Issues

If training crashes, resume from checkpoint:
```bash
--resume_from_checkpoint outputs/sft_baseline/checkpoint-1000
```

## Next Steps After Training

1. **Ablation Studies**: Test different lambda values for motion reward
2. **Full Dataset**: Download remaining videos and retrain
3. **Evaluation**: Run on motion-specific benchmarks (NExT-QA, GroundMoRe)
4. **Paper**: Write up results with baseline comparison

## Dataset Expansion

To download the full dataset (remaining 82% of samples):

See `DATASET_DOWNLOAD.md` for instructions on obtaining:
- VideoR1 videos (15k samples)
- GQA videos (5k samples)
- VideoEspresso videos (5k samples)
- TimerFT videos (2.3k samples)
- Additional temporal grounding videos

## Questions?

Check:
- `IMPLEMENTATION_PLAN.md` for detailed roadmap
- `OPEN_O3_ANALYSIS.md` for architecture details
- `INTEGRATION.md` for reward integration guide
