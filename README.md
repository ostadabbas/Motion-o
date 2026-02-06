# Spatio-Temporal Motion Reasoning with GRPO


Training vision-language models to reason about video motion through verifiable, spatially-grounded evidence chains using Group Relative Policy Optimization (GRPO).

## Overview

This project trains VLMs (Qwen3-VL-8B) to generate **spatio-temporal evidence chains** with:
- **Bounding boxes** for spatial grounding (`<bbox>[x1,y1,x2,y2]</bbox>`)
- **Motion descriptors** (centroid displacement, velocity, direction)
- **Temporal intervals** for each reasoning step
- **Verifiable reasoning** anchored to real coordinates

The system uses **RL-only alignment** with geometric rewards (no SFT), training on the PLM-STC dataset with masklets and motion annotations.

## Architecture

```
PLM-STC Dataset → Preprocessing → Motion Dataset
                                        ↓
                                   GRPO Trainer
                                        ↓
                    Model Output: Evidence Chain with Bboxes
                                        ↓
                                  Evidence Parser
                                        ↓
                               Geometric Reward Function
                                        ↓
                    (Spatial IoU + Temporal IoU + Motion + Caption)
```

### Geometric Rewards

Multi-dimensional reward combining:
- **R_spatial** (λ=0.25): Bbox IoU via Hungarian matching
- **R_temporal** (λ=0.15): Temporal interval IoU
- **R_motion** (λ=0.35): Trajectory matching (direction + speed + smoothness)
- **R_caption** (λ=0.20): Text similarity (token F1 + Levenshtein)
- **R_format** (gate): Parseability validation

## Installation

```bash
# Create conda environment
conda create -n motion_vlm python=3.10
conda activate motion_vlm

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft trl datasets pillow opencv-python scipy tqdm
```

## Quick Start

### 1. Preprocess PLM-STC Dataset

Convert raw PLM-STC data (masklets, annotations) into training-ready format:

```bash
python scripts/preprocess_plm_stc.py \
    /path/to/plm_stc \
    /path/to/output \
    --split train \
    --max-frames 32
```

**Expected PLM-STC structure:**
```
plm_stc/
├── videos/
│   └── {video_id}.mp4
├── annotations/
│   └── train.json  # Contains: video_id, question, answer, evidence_steps
└── masklets/
    └── {video_id}_{step_idx}.npy  # Masklet arrays
```

### 2. Test Pipeline

Verify all components work with synthetic data:

```bash
python scripts/test_motion_pipeline.py
```

### 3. Train Model

Train Qwen3-VL-8B with GRPO on preprocessed data:

```bash
bash shell_scripts/train_motion.sh /path/to/preprocessed/train
```

Or manually:

```bash
python scripts/train_motion_grpo.py /path/to/preprocessed/train \
    --output-dir ./outputs/motion_grpo \
    --model-id Qwen/Qwen3-VL-8B-Instruct \
    --use-4bit \
    --use-lora \
    --num-generations 8 \
    --max-steps 1000 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-5 \
    --max-response-length 512 \
    --max-frames 16 \
    --lambda-spatial 0.25 \
    --lambda-temporal 0.15 \
    --lambda-motion 0.35 \
    --lambda-caption 0.20
```

### 4. Quick Validation (10 steps)

```bash
python scripts/train_motion_grpo.py /path/to/dataset \
    --max-steps 10 \
    --use-lora \
    --use-4bit \
    --debug-reward
```

## Project Structure

```
vlmm-mcot/
├── scripts/
│   ├── train_motion_grpo.py         # Main GRPO training script
│   ├── preprocess_plm_stc.py        # PLM-STC preprocessing
│   ├── test_motion_pipeline.py      # End-to-end tests
│   └── archived/                    # Old Dora scripts
├── src/
│   ├── motion_dataset.py            # Motion GRPO dataset
│   ├── evidence_parser.py           # Parse evidence chains
│   ├── motion_metrics.py            # Geometric metrics
│   ├── geometric_reward.py          # Multi-dim reward
│   ├── model_loader.py              # VLM model loader
│   ├── video_utils.py               # Frame extraction
│   ├── text_cleaning.py             # Text utilities
│   └── archived/                    # Old modules
├── shell_scripts/
│   └── train_motion.sh              # Training launcher
└── config/
    └── motion_config.yaml           # Configuration (optional)
```

## Example Output

The model generates evidence chains like:

```
Step 1: [2.1–3.4] Person <bbox>[120,80,220,350]</bbox> picks up ball <bbox>[200,300,240,340]</bbox>
Motion: ball centroid shifts from (220,320) to (180,120) over 1.3s, velocity 150px/s
Description: Person reaches down and picks up the ball from the ground

Step 2: [3.4–5.0] Ball <bbox>[160,50,210,100]</bbox> hits wall
Motion: velocity direction flips from (-30,-150)/s to (+20,+80)/s, motion reversal detected
Description: Ball bounces off the wall and changes direction

Answer: The ball changed direction because it hit the wall at t=3.4s
```

## Key Features

### RL-Only Training
No supervised fine-tuning needed - Qwen2.5-VL already has bbox generation capability, GRPO composes it into motion chains through reward signals.

### Verifiable Reasoning
Every reasoning step is falsifiable with real coordinates:
- Bboxes can be overlaid on frames
- Motion descriptors can be computed from predicted bboxes
- Temporal intervals can be validated against video timestamps

### Modular Design
Each component is independent and testable:
- `evidence_parser.py`: Parsing only
- `motion_metrics.py`: Individual metrics
- `geometric_reward.py`: Reward composition
- Easy to tune weights or swap metrics

## Configuration

Default reward weights (can be adjusted via CLI):
```python
lambda_spatial = 0.25   # Bbox IoU
lambda_temporal = 0.15  # Interval IoU  
lambda_motion = 0.35    # Trajectory (direction + speed + smoothness)
lambda_caption = 0.20   # Text similarity (F1 + Levenshtein)
```

## Training Tips

1. **Memory**: Use 4-bit quantization (`--use-4bit`) and LoRA (`--use-lora`) for 7B model
2. **Batch size**: 4-8 with gradient accumulation
3. **Response length**: Set to 512+ tokens for evidence chains (longer than Q&A)
4. **Generations**: 8 per prompt for diversity
5. **KL beta**: 0.01 prevents drift from base model
6. **Debug**: Use `--debug-reward` to see component rewards

## Hardware Requirements

- GPU: 40GB+ VRAM (A100/H100) for 8B model with 4-bit, or 4x V100 (16GB each)
- CPU: 32+ cores for data loading
- Storage: ~500GB for PLM-STC dataset + preprocessed data

## Citation

If you use this work, please cite:

```bibtex
@article{structured2026,
  title={Structured Over Scale: Learning Spatial Reasoning from Educational Video},
  author={Galoaa, Bishoy and Bai, Xiangyu and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:2601.23251},
  year={2026}
}
```

## License

MIT License
