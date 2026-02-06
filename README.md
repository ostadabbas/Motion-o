# Spatio-Temporal Motion Reasoning with GRPO

Training vision-language models to reason about video motion through verifiable, spatially-grounded evidence chains using Group Relative Policy Optimization (GRPO).

## ⚠️ CRITICAL ISSUE - DATASET PROBLEM

**STATUS**: The PLM-Video-Human RDCap dataset has **INCORRECT masklet_id mappings**. The `masklet_id` field in RDCap does NOT correspond to the objects described in `dense_captions`. SA-V masklets are tracking wrong objects (furniture, background, etc.) instead of people mentioned in captions.

**Data Status:**
- ✅ SA-V videos downloaded: ~50,232 videos
- ✅ SA-V annotations: ~98,315 JSON files (_manual.json and _auto.json)
- ❌ RDCap masklet_id mappings: INCORRECT

**Examples found**:
- `sav_015834`: Caption says "person walking" but masklet_id=0 tracks walkway shadow/grate
- `sav_003127`: Caption says "man walking into mall" but masklet_id=1 is EMPTY
- `sav_017599`: Caption says "boy enters frame" but all masklets are EMPTY at described timestamps

**Root Cause:** The `masklet_id` field in RDCap does not correctly index the SA-V masklets. For `sav_015834`:
- RDCap says use `masklet_id=0`
- SA-V JSON has `masklet_id: [0, 1, 2, 3]` (4 tracked objects)
- Object 0 = walkway patch (WRONG!)
- Object 1 = grate on walkway (WRONG!)
- The actual person walking is NOT being tracked by any masklet

**Impact**: Cannot train on this dataset without fixing masklet associations.

**Options**:
1. Implement heuristic search across ALL masklets per frame (find largest moving person-shaped object)
2. Report issue to Meta/Facebook and wait for fix
3. Switch to different dataset (e.g., pure SA-V with auto-generated captions)

**See**: `scripts/debug_all_masklets.py` to visualize ALL masklets and verify the mismatch

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
pip install transformers accelerate peft trl datasets pillow opencv-python scipy tqdm pycocotools
```

## Dataset Setup (PLM-Video-Human RDCap)

### Step 1: Download RDCap Annotations

```bash
# Annotations are downloaded automatically via HuggingFace
python -c "from datasets import load_dataset; ds = load_dataset('facebook/PLM-Video-Human', 'rdcap', split='train', streaming=True)"
```

### Step 2: Download SA-V Videos

The RDCap annotations reference videos from the Segment Anything Video (SA-V) dataset, which must be downloaded manually.

```bash
# 1. Go to https://ai.meta.com/datasets/segment-anything-video/
# 2. Accept terms and download videos
# 3. Extract to /mnt/data/plm_stc/raw/sa-v/
# 4. Verify structure:
#    /mnt/data/plm_stc/raw/sa-v/
#    ├── sav_000001.mp4
#    ├── sav_000001_manual.json  (masklet annotations)
#    ├── sav_000001_auto.json
#    └── ... (~50K videos + ~98K JSON files)

# Or use download script (requires authentication):
bash scripts/download_sav_videos.sh
```

**Verify Download:**
```bash
# Should show ~50K videos
find /mnt/data/plm_stc/raw/sa-v/ -name "*.mp4" | wc -l

# Should show ~98K JSON files
find /mnt/data/plm_stc/raw/sa-v/ -name "*.json" | wc -l

# Check specific video exists
ls -lh /mnt/data/plm_stc/raw/sa-v/sav_015834*
```

### Step 3: Convert RDCap + SA-V to Training Format

```bash
python scripts/convert_plm_stc_to_format.py \
    --input-annotations /mnt/data/plm_stc/raw/rdcap \
    --input-videos /mnt/data/plm_stc/raw/sa-v \
    --output-dir /mnt/data/plm_stc/formatted_test \
    --limit 100  # Start small for testing

# Output structure:
#   /mnt/data/plm_stc/formatted_test/
#   ├── videos/ (symlinks to SA-V videos)
#   ├── masklets/ (converted .npy files)
#   └── annotations/train.json
```

**Note**: This script:
- Loads masklets from SA-V JSON files
- Decodes RLE masks to numpy arrays
- Slices masklets by temporal segments from dense_captions
- ⚠️ **USES INCORRECT masklet_id FROM RDCAP** (known issue)

### Step 4: Preprocess for Training

```bash
python scripts/preprocess_plm_stc.py \
    /mnt/data/plm_stc/formatted_test \
    /mnt/data/plm_stc/preprocessed_test \
    --split train \
    --max-frames 8

# Output: HuggingFace dataset with frames + evidence_steps + bboxes
#   /mnt/data/plm_stc/preprocessed_test/train/
```

### Debug Visualizations

```bash
# Visualize all masklets for a specific frame (to find correct object)
python scripts/debug_all_masklets.py

# Visualize final preprocessed dataset
python scripts/visualize_clean_dataset.py
```

## Quick Start

### 1. Test Pipeline

Verify all components work with synthetic data:

```bash
python scripts/test_motion_pipeline.py
```

### 2. Train Model

⚠️ **WARNING**: Training will not work correctly due to incorrect masklet mappings in RDCap dataset. See "CRITICAL ISSUE" section above.

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

### 3. Quick Validation (10 steps)

```bash
python scripts/train_motion_grpo.py /path/to/dataset \
    --max-steps 10 \
    --use-lora \
    --use-4bit \
    --debug-reward
```

## Debugging & Visualization Tools

### Check Raw RDCap Data

```bash
# View RDCap sample structure
python -c "
from datasets import load_dataset
ds = load_dataset('facebook/PLM-Video-Human', 'rdcap', split='train', streaming=True)
sample = next(iter(ds))
print('Video:', sample['video'])
print('Masklet ID:', sample['masklet_id'])
print('Dense captions:', sample['dense_captions'])
"
```

### Visualize SA-V Masklets

```bash
# Show all masklets for a specific frame
python scripts/debug_all_masklets.py

# This reveals which masklet_id actually contains the person
# (often different from what RDCap says!)
```

### Visualize Preprocessed Dataset

```bash
python scripts/visualize_clean_dataset.py

# Shows final GT bboxes overlaid on frames
# Helps verify if bboxes match captions
```

### Check Coordinate Systems

```bash
python scripts/debug_coordinates.py

# Verifies pixel → normalized → pixel conversions
```

## Known Issues & Workarounds

### Issue 1: Incorrect Masklet IDs (CRITICAL)

**Problem**: RDCap's `masklet_id` does not match objects in `dense_captions`

**Evidence**:
- Run `scripts/debug_all_masklets.py` to see all masklets for a frame
- Compare visual objects with caption text
- Most samples have mismatched masklets

**Workarounds**:
1. **Heuristic Search**: Modify `convert_plm_stc_to_format.py` to search ALL masklets and pick the largest moving person-shaped object
2. **Manual Filtering**: Curate subset of videos where masklets are correct
3. **Use Different Dataset**: Switch to pure SA-V with auto-generated captions

### Issue 2: "Out of frame" Segments

**Problem**: RDCap includes temporal segments where subject is not visible

**Solution**: Current preprocessing filters out segments with "Out of frame" caption

```python
# In convert_plm_stc_to_format.py
if "out of frame" in caption.lower():
    continue  # Skip this segment
```

### Issue 3: Empty Masklets

**Problem**: Some masklets have zero non-zero pixels despite non-empty captions

**Detection**:
```python
masklets = np.load('masklet.npy')
if np.count_nonzero(masklets) == 0:
    print("Empty masklet!")
```

**Solution**: Preprocessing skips empty masklets

### Issue 4: Video Decoding Errors

**Problem**: Some SA-V videos have corrupted frames (`error while decoding MB`)

**Workaround**: Use `try/except` in frame extraction and skip corrupted videos

## Project Structure

```
vlmm-mcot/
├── scripts/
│   ├── train_motion_grpo.py              # Main GRPO training script
│   ├── convert_plm_stc_to_format.py      # Convert RDCap+SA-V to intermediate format
│   ├── preprocess_plm_stc.py             # Preprocess to HF dataset with frames
│   ├── test_motion_pipeline.py           # End-to-end tests
│   ├── debug_all_masklets.py             # Visualize ALL masklets per frame
│   ├── visualize_clean_dataset.py        # Visualize final preprocessed data
│   ├── debug_coordinates.py              # Test coordinate conversions
│   └── archived/                         # Old Dora scripts
├── src/
│   ├── motion_dataset.py                 # Motion GRPO dataset
│   ├── evidence_parser.py                # Parse evidence chains
│   ├── motion_metrics.py                 # Geometric metrics
│   ├── geometric_reward.py               # Multi-dim reward
│   ├── model_loader.py                   # VLM model loader
│   ├── video_utils.py                    # Frame extraction
│   ├── text_cleaning.py                  # Text utilities
│   └── archived/                         # Old modules
├── shell_scripts/
│   └── train_motion.sh                   # Training launcher
└── config/
    └── motion_config.yaml                # Configuration (optional)
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
- Storage: ~500GB for SA-V videos + ~100GB for preprocessed data

## Next Steps to Fix Dataset Issue

### Option 1: Implement Heuristic Masklet Search

Modify `scripts/convert_plm_stc_to_format.py`:

```python
def find_best_masklet(masklet_data, caption, frame_range):
    """
    Search ALL masklets and pick the most likely match based on:
    - Size (prefer person-sized objects: 5,000-50,000 pixels)
    - Motion (prefer objects that move consistently)
    - Position (prefer objects in center/foreground)
    - Temporal consistency (prefer objects visible throughout segment)
    """
    # Implementation needed
    pass
```

### Option 2: Manual Curation

1. Run `scripts/debug_all_masklets.py` on entire dataset
2. Create mapping file: `{video_id: {correct_masklet_id}}`
3. Update conversion script to use corrected mappings

### Option 3: Switch Dataset

Use pure SA-V with auto-generated captions:
- SA-V has correct masklet annotations
- Generate captions using existing VLM (e.g., Qwen2-VL)
- Trade caption quality for correct spatial grounding

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
