# Quick Start Guide

## 1. Test Motion Metrics (Standalone)

Test the motion reward computation without training:

```bash
python scripts/test_motion_pipeline.py
```

This will:
- Create synthetic GT and predicted evidence chains
- Compute spatial, temporal, motion, and caption rewards
- Show component breakdowns

Expected output:
```
=== Geometric Reward Computation ===
R_spatial: 0.750 (bbox IoU)
R_temporal: 0.850 (interval IoU)
R_motion: 0.620 (trajectory matching)
  - Direction: 0.82
  - Speed: 0.61
  - Smoothness: 0.31
R_caption: 0.440 (text similarity)
Total Reward: 0.653
```

## 2. Clone and Integrate with Open-o3 Video

```bash
# Clone Open-o3 Video
git clone https://github.com/marinero4972/Open-o3-Video
cd Open-o3-Video

# Setup environment
conda create -n open-o3-video python=3.11
conda activate open-o3-video
bash setup.sh

# Copy motion-aware modules
cp ../vlmm-mcot/src/motion_metrics.py ./src/
cp ../vlmm-mcot/src/geometric_reward.py ./src/
cp ../vlmm-mcot/src/evidence_parser.py ./src/

# Download STGR dataset (follow their instructions)
# Set DATA_ROOT in src/r1-v/configs/data_root.py
```

## 3. Modify Open-o3's RL Reward

Edit their RL training script (likely `src/rl_trainer.py` or similar):

```python
# Add at top:
from motion_metrics import compute_trajectory_iou, compute_direction_similarity
from geometric_reward import compute_geometric_reward

# In reward computation function, replace:
def compute_reward(pred_text, gt_evidence, frames, fps):
    from evidence_parser import parse_think_predict_chain
    
    pred_steps = parse_think_predict_chain(pred_text)
    
    reward = compute_geometric_reward(
        pred_steps=pred_steps,
        gt_steps=gt_evidence,
        frames=frames,
        fps=fps,
        lambda_spatial=0.25,
        lambda_temporal=0.15,
        lambda_motion=0.35,      # Motion-aware trajectory reward
        lambda_caption=0.20
    )
    
    return reward
```

## 4. Update Prompt Template

Modify the system prompt to include think/predict format:

```python
PROMPT_TEMPLATE = """
Given a video and a question, provide step-by-step reasoning with spatio-temporal evidence.

For each step, provide:
1. Temporal interval: [t_s–t_e]
2. Think bbox: Initial rough spatial estimate
3. Predict bbox: Refined spatial prediction
4. Motion: Quantitative motion descriptors
5. Description: What happened in this step

Format:
Step 1: [0.0s–2.5s] Brief description
  Think: (x1,y1),(x2,y2)
  Predict: (x1,y1),(x2,y2)
  Motion: velocity: 150px/s, direction: northeast
  Description: Detailed description of the event

Answer: Final answer to the question
"""
```

## 5. Train

```bash
# Cold-start SFT (use their original script)
bash ./src/scripts/run_sft_video.sh

# Motion-aware RL training (with your modified reward)
bash ./src/scripts/run_grpo_video.sh
```

## 6. Evaluate

```bash
cd eval
bash ./scripts/eval_vstar.sh        # V-STAR benchmark
bash ./scripts/eval_videomme.sh     # VideoMME
bash ./scripts/eval_nextqa.sh       # Motion-heavy benchmark
```

## Key Configuration

**Reward weights** (tune these):
- `lambda_spatial = 0.25`  # Bbox IoU
- `lambda_temporal = 0.15` # Interval IoU
- `lambda_motion = 0.35`   # **Trajectory-level motion** (your contribution)
- `lambda_caption = 0.20`  # Text similarity

**Motion components** (in motion_metrics.py):
- Direction matching: Cosine similarity of displacement vectors
- Speed fidelity: Velocity magnitude matching  
- Smoothness: Penalize acceleration spikes

## Troubleshooting

**Reward is negative:**
- Bboxes must be normalized [0,1]
- Check FPS is correct (usually 30)
- Verify temporal intervals are in seconds

**Motion reward is 0:**
- Need multiple frames with bboxes
- Check displacement vectors are non-zero
- Verify GT evidence has motion (not static objects)

**Training unstable:**
- Reduce `lambda_motion` from 0.35 to 0.20
- Increase KL penalty in RL config
- Use gradient clipping (default: 1.0)

## Expected Results

On V-STAR benchmark:
- **Baseline (Open-o3)**: mAM: 35.5%, mLGM: 49.0%
- **+Motion rewards**: Expected +2-5% on motion-heavy tasks

On NExT-QA (causal/temporal splits):
- Expected improvement: +5-10% accuracy on motion reasoning questions

## Next Steps

1. ✅ Test motion metrics standalone
2. ✅ Clone and setup Open-o3 Video
3. ✅ Integrate motion reward modules
4. ⏳ Train on STGR dataset
5. ⏳ Evaluate on benchmarks
6. ⏳ Ablate motion components
7. ⏳ Write paper

See `INTEGRATION.md` for detailed integration instructions.
