# VLM Spatial Grounding Testing Suite

This repository contains diagnostic tests to evaluate Vision-Language Models' ability to perform spatio-temporal reasoning with grounded evidence chains.

## Quick Start

All tests use the `dora_cuda` conda environment:

```bash
conda activate dora_cuda

# Or run directly with:
conda run -n dora_cuda python scripts/test_xxx.py ...
```

---

## Test Scripts

### 1. Final Ball Tracking Test (Recommended Starting Point)
**Script**: `scripts/test_final_ball_tracking.py`

Tests single-frame detection capability and basic motion tracking.

```bash
python scripts/test_final_ball_tracking.py \
    test_videos/Ball_Animation_Video_Generation.mp4 \
    --frames "0,48,96,144,191" \
    --model-id "Qwen/Qwen2-VL-2B-Instruct" \
    --device cuda:0 \
    --output-dir outputs/final_tracking
```

**Output**:
- `outputs/final_tracking/frame_XXXX_tracked.jpg` - Annotated frames with bboxes
- `outputs/final_tracking/tracking_data.json` - Structured tracking results

**What it shows**: Whether the model can detect objects and track them across frames

---

### 2. Motion Chain Inference Test (Key for GRPO Training)
**Script**: `scripts/test_motion_chain_inference.py`

Tests the model's ability to generate structured spatio-temporal evidence chains - the target format for GRPO training.

```bash
python scripts/test_motion_chain_inference.py \
    test_videos/Ball_Animation_Video_Generation.mp4 \
    "Describe the motion trajectory of the red ball. Where does it start, where does it end, and how does it move across the scene? Provide spatial evidence with bounding boxes for each key position." \
    --num-frames 8 \
    --max-tokens 1024 \
    --model-id "Qwen/Qwen2-VL-2B-Instruct" \
    --output-dir outputs/motion_chain_inference
```

**Output**:
- `outputs/motion_chain_inference/motion_chain_response.txt` - Full reasoning chain
- `outputs/motion_chain_inference/motion_chain_result.json` - Structured result

**What it shows**: Whether the model can produce multi-step evidence chains with temporal intervals, bboxes, and motion descriptors

**Key insight**: Base model can generate the FORMAT but fills it with hallucinated/copied values. GRPO training will teach real spatial grounding.

---

### 3. Hallucination Diagnostic Test
**Script**: `scripts/diagnose_detection_hallucination.py`

Compares VLM predictions against classical computer vision ground truth to detect hallucinations.

```bash
python scripts/diagnose_detection_hallucination.py \
    test_videos/Ball_Animation_Video_Generation.mp4 \
    --frames "0,48,96,144,191" \
    --model-id "Qwen/Qwen2-VL-2B-Instruct"
```

**Output**:
- `outputs/hallucination_diagnosis/comparison_frame_XXXX.jpg` - Side-by-side comparison (original | CV ground truth | VLM prediction)

**What it shows**:
- False positives (hallucinated bboxes)
- False negatives (missed detections)
- IoU between VLM and ground truth

---

### 4. Ball Grounding Test (Multiple Prompts)
**Script**: `scripts/test_ball_grounding.py`

Tests different prompting strategies for spatial grounding.

```bash
# Test prompt version 1 (explicit bbox format)
python scripts/test_ball_grounding.py \
    test_videos/Ball_Animation_Video_Generation.mp4 \
    --prompt-version 1 \
    --num-frames 8

# Test prompt version 2 (JSON format)
python scripts/test_ball_grounding.py \
    test_videos/Ball_Animation_Video_Generation.mp4 \
    --prompt-version 2 \
    --num-frames 8

# Test prompt version 3 (example-based)
python scripts/test_ball_grounding.py \
    test_videos/Ball_Animation_Video_Generation.mp4 \
    --prompt-version 3 \
    --num-frames 8
```

**Output**:
- `outputs/grounding_test/frame_first_with_boxes.jpg`
- `outputs/grounding_test/frame_last_with_boxes.jpg`
- `outputs/grounding_test/response.txt`

**What it shows**: How prompt engineering affects bbox generation quality

---

### 5. Frame-by-Frame Tracking Test
**Script**: `scripts/test_frame_by_frame_tracking.py`

Tests independent frame detection to identify if the model can track motion or just detects static positions.

```bash
python scripts/test_frame_by_frame_tracking.py \
    test_videos/Ball_Animation_Video_Generation.mp4 \
    --frames "0,48,96,144,191"
```

**Output**:
- `outputs/frame_by_frame_tracking/frame_XXXX_tracked.jpg`
- `outputs/frame_by_frame_tracking/tracking_data.json`

**What it shows**: Frame-by-frame detection accuracy and motion consistency

---

## Understanding the Results

### Good Signs ✓
- High IoU (> 0.5) with ground truth
- Monotonic motion (positions consistently increase/decrease)
- Detections in most frames (> 80%)
- Smooth trajectories (no sudden jumps)

### Warning Signs ⚠
- Identical bboxes across frames (copying from examples)
- Non-monotonic motion (random position jumps)
- Temporal intervals beyond video duration (hallucination)
- Low IoU (< 0.3) with ground truth

### Critical Issues ✗
- False positives (detecting objects that don't exist)
- Completely missing detections (0% recall)
- Random bbox guessing
- No spatial grounding at all

---

## Diagnostic Summary

After running tests, check `outputs/DIAGNOSTIC_SUMMARY.md` for:
- Consolidated test results
- What's missing for GRPO training
- Recommended training strategy
- Success criteria

---

## Test Video

### Ball Animation Video
- **Path**: `test_videos/Ball_Animation_Video_Generation.mp4`
- **Properties**: 1280x720, 192 frames (8 seconds), 24 fps
- **Content**: Red ball moving smoothly from left to right across a wooden floor
- **Ground truth motion**: LEFT (x≈0.15) → RIGHT (x≈0.95), monotonic horizontal motion

---

## Expected Results (Base Model)

Based on our diagnostics with Qwen2-VL-2B:

| Test | Result | Score |
|------|--------|-------|
| Single-frame detection | Partial | 60% recall |
| Motion tracking | Poor | Non-monotonic |
| Bbox accuracy (IoU) | Low | 0.14-0.29 |
| Format generation | Good | 100% |
| Spatial grounding | Poor | Hallucinated |
| Multi-step chains | Format only | No real grounding |

**Conclusion**: Base model understands OUTPUT FORMAT but lacks SPATIAL GROUNDING.

---

## For GRPO Training

### What These Tests Reveal

1. **Baseline Capability**
   - Model can generate structured format
   - Model has limited bbox detection
   - Model lacks motion tracking

2. **Training Targets**
   - Improve spatial accuracy (IoU)
   - Enable motion tracking (consistency)
   - Ground all bbox predictions in visual evidence
   - Quantify motion descriptors accurately

3. **Reward Function Inputs**
   - Ground truth bboxes (from tracker or masklets)
   - Predicted bboxes (from model output)
   - Temporal intervals (from video timestamps)
   - Motion descriptors (from trajectory analysis)

### Using Results for Training

1. **Collect ground truth**: Run object tracker on training videos
2. **Generate baselines**: Run motion chain inference on training examples
3. **Compute rewards**: Compare predicted vs ground truth bboxes (IoU)
4. **Train with GRPO**: Maximize geometric reward
5. **Evaluate**: Re-run these tests to measure improvement

---

## Model Support

These scripts support:
- Qwen2-VL (Qwen2-VL-2B-Instruct, Qwen2-VL-7B-Instruct)
- Qwen2.5-VL (Qwen2.5-VL-*)
- Qwen3-VL (Qwen3-VL-8B-Instruct)

The model class is automatically selected based on the model ID.

---

## Troubleshooting

### Model Loading Errors
```
ImportError: cannot import name 'AutoModelForVision2Seq'
```
**Fix**: Models are loaded using specific classes (Qwen2VLForConditionalGeneration, etc.) which are automatically selected.

### CUDA Out of Memory
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Fix**: Reduce `--num-frames` or use smaller model (2B instead of 7B/8B)

### No Detections
If model returns no bboxes:
- Try different `--prompt-version` (for grounding test)
- Check if video actually contains the object
- Verify frames are being extracted correctly

---

## Citation

If you use these tests or insights, please cite:

```bibtex
@article{structured2026,
  title={Structured Over Scale: Learning Spatial Reasoning from Educational Video},
  author={Galoaa, Bishoy and Bai, Xiangyu and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:2601.23251},
  year={2026}
}
```

---

## Summary

Run these tests to:
1. ✓ Establish baseline VLM spatial grounding capability
2. ✓ Identify what GRPO training needs to fix
3. ✓ Generate example outputs for comparison
4. ✓ Design reward functions based on gaps
5. ✓ Evaluate training progress by re-running tests

**Key takeaway**: Base VLMs can format evidence chains but need GRPO to ground them in real spatial coordinates. 🎯
