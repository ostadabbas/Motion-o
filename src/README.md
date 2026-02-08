# Core Motion-Aware Reasoning Modules

## Overview

These modules implement trajectory-level geometric rewards for video motion reasoning.

## Core Modules

### `motion_metrics.py`
**Trajectory-level geometric metrics for motion reasoning.**

Key functions:
- `compute_bbox_iou()`: Spatial IoU between predicted and GT bboxes
- `compute_trajectory_iou()`: Hungarian matching of trajectories across frames
- `compute_direction_similarity()`: Cosine similarity of displacement vectors
- `compute_speed_fidelity()`: Velocity magnitude matching
- `compute_trajectory_smoothness()`: Penalize physically implausible jumps

**Innovation**: Evaluates *how* objects moved, not just *where* they are.

### `geometric_reward.py`
**Multi-dimensional reward function combining spatial, temporal, motion, and caption rewards.**

Main function:
```python
compute_geometric_reward(
    pred_steps,        # Parsed evidence steps
    gt_steps,          # Ground truth evidence
    frames,            # Video frames
    fps,               # Frames per second
    lambda_spatial,    # Weight for spatial IoU
    lambda_temporal,   # Weight for temporal IoU
    lambda_motion,     # Weight for trajectory motion (KEY)
    lambda_caption     # Weight for text similarity
)
```

Returns: Scalar reward in [0, 1]

### `evidence_parser.py`
**Parse model outputs into structured evidence chains with think/predict bboxes.**

Key functions:
- `parse_evidence_chain()`: Parse standard evidence format
- `parse_think_predict_chain()`: Parse think→predict bbox refinement format
- `extract_bboxes_from_text()`: Extract bboxes from various formats
- `extract_time_interval()`: Parse temporal intervals

Supports formats:
- `<bbox>[x1,y1,x2,y2]</bbox>` (normalized [0,1])
- Think/Predict dual-phase bboxes on 0-1000 scale

### `motion_dataset.py`
**Dataset wrapper for motion reasoning with evidence chains.**

Loads:
- Video frames
- Ground truth evidence steps with bboxes and captions
- Temporal intervals
- Motion descriptors

Can be adapted for STGR dataset format.

### `model_loader.py`
**VLM model loader with LoRA and quantization support.**

Supports:
- Qwen2.5-VL / Qwen3-VL
- 4-bit quantization
- LoRA adapters
- Flash attention

## Utility Modules

### `video_utils.py`
Frame extraction from videos with temporal sampling.

### `text_cleaning.py`
Text preprocessing for caption matching.

### `eval_utils.py`
Evaluation metrics for motion reasoning.

## Usage Example

```python
from evidence_parser import parse_think_predict_chain
from geometric_reward import compute_geometric_reward

# Parse model output
pred_steps = parse_think_predict_chain(model_output_text)

# Compute reward
reward = compute_geometric_reward(
    pred_steps=pred_steps,
    gt_steps=gt_evidence,
    frames=video_frames,
    fps=30.0,
    lambda_spatial=0.25,
    lambda_temporal=0.15,
    lambda_motion=0.35,   # Motion-aware trajectory reward
    lambda_caption=0.20
)
```

## Integration with Open-o3 Video

See `../INTEGRATION.md` for detailed integration guide.

Key steps:
1. Copy these modules to Open-o3 codebase
2. Replace their reward function with `compute_geometric_reward()`
3. Update prompt template for think/predict bboxes
4. Use `parse_think_predict_chain()` for parsing
5. Train with motion-aware rewards
