# Motion-Aware Trajectory Reasoning for Video Understanding

**Training VLMs to reason about video motion through verifiable, motion-aware evidence chains with trajectory-level geometric rewards.**

Built on [Open-o3 Video](https://github.com/marinero4972/Open-o3-Video) infrastructure with motion trajectory rewards as our core contribution.

## Core Contribution

**Motion-aware trajectory reward** — evaluating not just *where* objects are, but *how they moved*, using geometric motion metrics derived from predicted bbox sequences.

| Aspect | Open-o3 Video | This Work |
|--------|--------------|-----------|
| **Spatial Reward** | Per-frame IoU (static) | Trajectory-level motion |
| **Motion Modeling** | Implicit in text | Explicit (direction, speed, smoothness) |
| **Evaluation** | "Is bbox in right place?" | "Did object move correctly?" |
| **Key Question** | "What happened and where?" | "How did it move and why?" |

### Motion-Aware Reward Components

**R_motion (λ=0.35)**: Trajectory-level motion matching
- **Direction similarity** (0.4): Cosine similarity of displacement vectors
- **Speed fidelity** (0.4): Velocity magnitude matching
- **Trajectory smoothness** (0.2): Acceleration penalty for physically implausible motion

Combined with Open-o3's rewards:
- R_spatial (λ=0.25): Bbox IoU via Hungarian matching
- R_temporal (λ=0.15): Temporal interval IoU  
- R_caption (λ=0.20): Text similarity
- R_format (gate): Parseability validation

---

## Installation

```bash
conda create -n motion_vlm python=3.10
conda activate motion_vlm

# Install dependencies
pip install -r requirements.txt

# For DeepSpeed (multi-GPU training)
pip install deepspeed
```

---

## Quick Start

### 1. Setup Data

Download STGR dataset and update `configs/data_root.py`:

```python
DATA_ROOT = "/path/to/your/stgr/dataset"
```

Expected structure:
```
${DATA_ROOT}/
├── json_data/
│   ├── STGR-SFT.json (30k samples)
│   └── STGR-RL.json (36k samples)
└── videos/
    ├── gqa/, stgr/, timerft/, etc.
```

### 2. Train Baseline (without motion reward)

```bash
# SFT cold-start
bash scripts/run_sft.sh

# RL training (baseline)
bash scripts/run_grpo_baseline.sh
```

### 3. Train with Motion Reward

```bash
# RL training with motion-aware trajectory reward
bash scripts/run_grpo_motion.sh
```

### 4. Evaluate

```bash
cd evaluation
bash scripts/eval_all.sh
```

---

## Project Structure

```
vlmm-mcot/
├── src/                        # Core motion reasoning modules
│   ├── motion_metrics.py       # ⭐ Trajectory-level geometric metrics
│   ├── geometric_reward.py     # Multi-dimensional reward composition
│   ├── evidence_parser.py      # Evidence chain parsing
│   └── [supporting modules]
│
├── training/                   # Training infrastructure (from Open-o3)
│   ├── grpo_trainer.py         # GRPO trainer (808 lines, DeepSpeed)
│   ├── reward_func.py          # Modular reward functions
│   ├── motion_reward.py        # ⭐ Our motion trajectory reward
│   ├── train_grpo.py           # GRPO training script
│   ├── train_sft.py            # SFT training script
│   └── [data loader, vision processing]
│
├── evaluation/                 # Evaluation suite
│   ├── test/                   # V-STAR, VideoMME, VideoMMMU, WorldSense
│   └── scripts/eval_all.sh
│
├── configs/                    # Configuration files
│   ├── data_root.py            # Dataset path configuration
│   └── [DeepSpeed configs]
│
└── scripts/                    # Training launchers
    ├── run_sft.sh              # SFT cold-start
    ├── run_grpo_baseline.sh    # Baseline (no motion reward)
    └── run_grpo_motion.sh      # With motion reward ⭐
```

---

## Key Metrics

### Trajectory-Level Motion Metrics

**Direction Matching** (`src/motion_metrics.py`):
```python
cos_sim = dot(pred_direction, gt_direction) / (||pred|| * ||gt||)
```

**Speed Fidelity**:
```python
speed_score = exp(-|pred_speed - gt_speed| / (gt_speed + eps))
```

**Trajectory Smoothness**:
```python
accel = [v[t+1] - v[t] for t in range(T-1)]
smoothness = 1 / (1 + std(accel))
```

---

## Expected Results

### Quantitative Improvements

| Benchmark | Baseline | +Motion Reward | Δ |
|-----------|----------|----------------|---|
| V-STAR mAM | 35.5% | 37-40% | +2-5% |
| V-STAR mLGM | 49.0% | 52-56% | +3-7% |
| Motion-heavy tasks | - | - | +5-10% |

**Justification:** Open-o3's spatial reward is frame-independent. Our reward optimizes trajectory-level motion properties.

---

## Training Tips

### Multi-GPU Setup

```bash
# 8x GPUs with DeepSpeed ZeRO-3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_grpo_motion.sh
```

### Memory Optimization

- **DeepSpeed ZeRO-3**: Enabled in `configs/zero3.json`
- **Gradient checkpointing**: `--gradient_checkpointing true`
- **Flash Attention 2**: `--attn_implementation flash_attention_2`
- **4-bit quantization**: Add `--load_in_4bit` flag

### Monitoring Training

```bash
# Check reward components
tail -f /path/to/checkpoints/rl_motion/log.txt | grep "rewards/"
```

Expected output:
```
rewards/ans_acc: 0.75
rewards/thk_spatial: 0.62
rewards/motion_trajectory: 0.48  ← Motion-aware trajectory score
rewards/format: 1.0
reward: 5.23                      ← Total (sum of all rewards)
```

---

## Ablation Studies

Run experiments with different motion components:

```bash
# Edit training/motion_reward.py, line ~235:
# Full motion reward
motion_reward = 0.4*direction + 0.4*speed + 0.2*smoothness

# Direction only
motion_reward = 1.0*direction

# Direction + Speed
motion_reward = 0.5*direction + 0.5*speed
```

---

## Hardware Requirements

- **GPU**: 8x A100 (40GB) or 8x V100 (32GB)
- **CPU**: 32+ cores for data loading
- **Storage**: ~500GB for STGR dataset
- **Memory**: ~200GB RAM recommended

**Smaller setups:**
- 4x GPUs: Reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps`
- Single GPU: Use 4-bit quantization + LoRA

---

## Citation

```bibtex
@article{galoaa2026motion,
  title={Motion-Aware Trajectory Reasoning for Video Understanding},
  author={Galoaa, Bishoy and Bai, Xiangyu and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

Built upon [Open-o3 Video](https://github.com/marinero4972/Open-o3-Video):
```bibtex
@article{meng2025openo3,
  title={Open-o3 Video: Grounded Video Reasoning with Explicit Spatio-Temporal Evidence},
  author={Meng, Jiahao and Li, Xiangtai and Wang, Haochen and others},
  journal={arXiv preprint arXiv:2510.20579},
  year={2025}
}
```

---

## Documentation

- **`OPEN_O3_ANALYSIS.md`** - Analysis of Open-o3 infrastructure
- **`IMPLEMENTATION_PLAN.md`** - Step-by-step integration guide
- **`INTEGRATION.md`** - Technical integration details
- **`SUMMARY.md`** - Executive summary

---

## License

MIT License

---

## Acknowledgments

This work builds on the excellent infrastructure provided by [Open-o3 Video](https://github.com/marinero4972/Open-o3-Video). We use their GRPO trainer, STGR dataset, and evaluation suite while contributing motion-aware trajectory rewards for improved motion reasoning.
