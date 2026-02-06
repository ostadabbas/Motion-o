# Multi-GPU Inference Guide

## Your Hardware Setup

```
GPU 0: NVIDIA GeForce GTX 745 (4GB)   ← Skip this (weak)
GPU 1: Tesla V100-SXM2-32GB          ← Use these!
GPU 2: Tesla V100-SXM2-32GB
GPU 3: Tesla V100-SXM2-32GB
GPU 4: Tesla V100-SXM2-32GB

Total V100 VRAM: 128GB (4 × 32GB)
```

---

## Recommended Models

With 4x V100-32GB, you can run:

| Model | Size | VRAM w/ 4-bit | VRAM w/o 4-bit | Recommended |
|-------|------|---------------|----------------|-------------|
| Qwen2-VL-2B | 2B | ~2GB | ~4GB | Too small |
| Qwen2-VL-7B | 7B | ~7GB | ~14GB | ✓ Good |
| **Qwen2.5-VL-7B** | 7B | ~7GB | ~14GB | ✓✓ **Best choice** |
| Qwen3-VL-8B | 8B | ~8GB | ~16GB | ✓ Newest |

**Recommendation**: Use **Qwen2.5-VL-7B-Instruct** with 4-bit quantization
- Better spatial grounding than 2B model
- Fits easily on single V100 with 4-bit (~7GB)
- Multi-GPU auto-splits for faster inference
- Newer than Qwen2-VL, same API

---

## Quick Start

### Option 1: Using the Helper Script (Easiest)

```bash
# Run with Qwen2.5-VL-7B (recommended)
./scripts/run_multi_gpu_test.sh

# Or specify model explicitly:
./scripts/run_multi_gpu_test.sh Qwen/Qwen2.5-VL-7B-Instruct

# Or test with Qwen3-VL-8B:
./scripts/run_multi_gpu_test.sh Qwen/Qwen3-VL-8B-Instruct
```

### Option 2: Direct Python Command

```bash
# Set GPUs (skip GTX 745)
export CUDA_VISIBLE_DEVICES=1,2,3,4

# Run with multi-GPU
conda run -n dora_cuda python scripts/test_think_bbox_inference.py \
    test_videos/Ball_Animation_Video_Generation.mp4 \
    "Describe the motion trajectory of the red ball" \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --device auto \
    --use-4bit \
    --multi-gpu \
    --strategies explicit_binding freeform_chain \
    --num-frames 8
```

---

## How Multi-GPU Works

### Automatic Device Mapping

When you use `--device auto --multi-gpu`, the model automatically splits across GPUs:

```python
device_map="auto"
# Transformers library will:
# 1. Calculate model size
# 2. Detect available GPUs
# 3. Split layers evenly across GPUs
# 4. Handle data transfer automatically
```

**Example allocation** for 7B model across 4 GPUs:
```
GPU 1 (cuda:0 after CUDA_VISIBLE_DEVICES): Layers 0-10, embeddings
GPU 2 (cuda:1): Layers 11-20
GPU 3 (cuda:2): Layers 21-30  
GPU 4 (cuda:3): Layers 31-40, head
```

### Memory Estimates

**Qwen2.5-VL-7B with 4-bit quantization:**
- Model weights: ~7GB
- Activations: ~3-5GB (depends on batch size, num frames)
- Total per GPU: ~3-4GB (split across 4 GPUs)
- **Fits easily!** Each V100 has 32GB

**Without 4-bit:**
- Model weights: ~14GB
- Still fits fine on 4x V100s

---

## GPU Selection Options

### Use All 4 V100s (Recommended)
```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4
python script.py --device auto --multi-gpu
```

### Use Specific V100s
```bash
# Use only GPUs 1 and 2
export CUDA_VISIBLE_DEVICES=1,2
python script.py --device auto --multi-gpu

# Use single V100 (GPU 1)
export CUDA_VISIBLE_DEVICES=1
python script.py --device cuda:0  # cuda:0 maps to physical GPU 1
```

### Don't Use GTX 745
The GTX 745 is much slower than V100s. Always skip it:
```bash
# ✗ DON'T DO THIS (includes GTX 745):
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# ✓ DO THIS (only V100s):
export CUDA_VISIBLE_DEVICES=1,2,3,4
```

---

## Performance Comparison

Expected inference time for 8 frames:

| Setup | Model | Time per inference | Notes |
|-------|-------|-------------------|-------|
| 1x V100 | 2B (no quant) | ~3s | Current baseline |
| 1x V100 | 7B (4-bit) | ~8-10s | Single GPU |
| 4x V100 | 7B (4-bit) | ~6-8s | Faster with parallelism |
| 4x V100 | 7B (no quant) | ~5-7s | Even faster |

**Multi-GPU benefits:**
- Faster inference (layers compute in parallel)
- Handles larger models
- Better for training (can use larger batch sizes)

---

## Testing the Bigger Model

### Run the Test

```bash
# Clean old outputs
rm -rf outputs/think_bbox_test

# Run with Qwen2.5-VL-7B on 4 V100s
export CUDA_VISIBLE_DEVICES=1,2,3,4

conda run -n dora_cuda python scripts/test_think_bbox_inference.py \
    test_videos/Ball_Animation_Video_Generation.mp4 \
    "Describe the motion trajectory of the red ball" \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --device auto \
    --use-4bit \
    --multi-gpu \
    --strategies explicit_binding freeform_chain \
    --num-frames 8 \
    --output-dir outputs/think_bbox_7b_test
```

### What to Expect from 7B Model

Larger models typically have:
- ✓ **Better spatial grounding** (more accurate bbox predictions)
- ✓ **Stronger visual understanding** (less hallucination)
- ✓ **More consistent tracking** (smoother trajectories)
- ✓ **Better following complex prompts** (Think→Pred structure)

**Hypothesis**: 7B model will produce bboxes that are:
1. Less random (more grounded in actual ball position)
2. Smoother motion (less jumping around)
3. Higher quality Think→Pred refinement

---

## For GRPO Training

Once you find a good model, use multi-GPU for training too:

```python
# In your training script
accelerator = Accelerator(
    mixed_precision="bf16",
    # Will automatically use all GPUs in CUDA_VISIBLE_DEVICES
)

# Model will be distributed automatically
model = accelerator.prepare(model)
```

Or with Hugging Face Trainer:

```python
training_args = GRPOConfig(
    # ... other args ...
    ddp_find_unused_parameters=False,
    # Trainer will use all available GPUs automatically
)
```

---

## Troubleshooting

### Out of Memory
```
torch.cuda.OutOfMemoryError
```
**Solution**: Add `--use-4bit` flag

### Model Not Splitting
```
All layers on GPU 0
```
**Solution**: 
1. Check `CUDA_VISIBLE_DEVICES` is set
2. Use `--device auto` (not `cuda:0`)
3. Add `--multi-gpu` flag

### Slow Inference
```
Taking too long
```
**Solution**:
1. Verify using V100s, not GTX 745
2. Check `nvidia-smi` during inference
3. Reduce `--num-frames` if needed

---

## Quick Commands Reference

```bash
# See GPU allocation in real-time
watch -n 1 nvidia-smi

# Test with different models
export CUDA_VISIBLE_DEVICES=1,2,3,4

# 7B model (recommended)
./scripts/run_multi_gpu_test.sh Qwen/Qwen2.5-VL-7B-Instruct

# 8B model (newest)  
./scripts/run_multi_gpu_test.sh Qwen/Qwen3-VL-8B-Instruct

# Check device mapping after model loads
python -c "
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-7B-Instruct',
    device_map='auto',
    load_in_4bit=True,
    trust_remote_code=True
)
print(model.hf_device_map)
"
```

---

## Expected Results

With the 7B model, you should see:
- Better spatial grounding (bboxes closer to actual ball)
- More consistent motion (less random jumping)
- Smoother trajectories (monotonic L→R progression)

This will give you a **stronger baseline** for GRPO training, requiring less reward signal to achieve good spatial grounding.
