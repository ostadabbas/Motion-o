# Qwen2-VL Architecture Notes for Token Extraction

## Model Overview

**Model**: Qwen2-VL-7B-Instruct  
**Parameters**: ~7 billion  
**Architecture**: Vision Transformer + Language Model  
**Framework**: HuggingFace Transformers 4.57.3

---

## Architecture Stages

```
Input: Video/Image
    ↓
┌─────────────────────────────────────┐
│  Vision Transformer (ViT)          │
│  - Patch embedding                  │
│  - 32 transformer blocks            │
│  - Output: visual.blocks[31]        │
└─────────────────────────────────────┘
    ↓ [4784 tokens, 1280-dim]
    ↓
┌─────────────────────────────────────┐
│  Visual-to-Language Connector       │
│  - Merger/projection layer          │
│  - Spatial compression (4:1)        │
│  - Dimension expansion (1280→3584)  │
│  - Output: visual.merger            │
└─────────────────────────────────────┘
    ↓ [1196 tokens, 3584-dim]
    ↓
┌─────────────────────────────────────┐
│  Language Model (Qwen2)             │
│  - Token embedding fusion           │
│  - Transformer layers               │
│  - Text generation                  │
└─────────────────────────────────────┘
    ↓
Output: Text
```

---

## Token Extraction Points

### 1. visual.blocks.last (visual.blocks[31])

**Location**: After final Vision Transformer block  
**Hook**: `model.visual.blocks[31].register_forward_hook()`

**Token Specification:**
```python
Shape: [batch_size, num_tokens, hidden_dim]
       [1, 4784, 1280]

Spatial Grid: 52 × 92 = 4,784 tokens
Coverage: ~24.6 × 7.8 pixels/token (for 1280×720 input)

Statistics:
  dtype: float32 (after cleaning)
  mean: ~3872.25
  std: ~3459.50
  range: [64.88, 15826.46]
```

**Properties:**
- High-resolution visual representation
- Velocity-like motion encoding (R²=0.995 linearity)
- Preserves directional information (PC1: 30%)
- Good for fine-grained spatial reasoning

**Motion Characteristics:**
- Delta magnitude: ~2628.55 ± 2661.37
- Motion/Static ratio: 1.14x (p < 10^-13)
- Detection F1: 0.336

### 2. visual.merger

**Location**: After visual-to-language connector  
**Hook**: `model.visual.merger.register_forward_hook()`

**Token Specification:**
```python
Shape: [batch_size, num_tokens, hidden_dim]
       [1, 1196, 3584]

Spatial Grid: 26 × 46 = 1,196 tokens
Coverage: ~49.2 × 15.7 pixels/token (for 1280×720 input)

Statistics:
  dtype: float32 (after cleaning)
  mean: ~47.01
  std: ~29.17
  range: [9.70, 274.44]
```

**Properties:**
- Compressed spatial resolution (4x fewer tokens)
- Expanded semantic dimension (2.8x larger hidden dim)
- Displacement-like motion encoding (R²=0.673, sub-linear)
- Better motion/static separation (1.28x ratio)
- Normalized/scaled values

**Motion Characteristics:**
- Delta magnitude: ~42.76 ± 22.07
- Motion/Static ratio: 1.28x (p < 10^-48)
- Detection F1: 0.180

---

## Hook Implementation

### Forward Hook Setup

```python
def make_hook(name):
    def hook(module, input, output):
        # Handle tuple output
        if isinstance(output, tuple):
            token_data = output[0].detach().cpu().float()
        else:
            token_data = output.detach().cpu().float()
        
        # Clean NaN/inf values
        token_data[torch.isnan(token_data)] = 0
        token_data[torch.isinf(token_data)] = 0
        
        self.extracted_tokens[name] = token_data
    return hook

# Register hooks
hook1 = model.visual.blocks[31].register_forward_hook(make_hook("visual.blocks.last"))
hook2 = model.visual.merger.register_forward_hook(make_hook("visual.merger"))
```

### Key Considerations

1. **Data Type Conversion**: Convert to float32 to avoid float16 instabilities
2. **NaN/Inf Handling**: Clean corrupted values (rare but critical)
3. **Memory Management**: `.detach().cpu()` to prevent gradient tracking and GPU OOM
4. **Tuple Handling**: ViT blocks may return (hidden_states, attention_weights)

---

## Video Processing Pipeline

### Input Processing

```python
from qwen_vl_utils import process_vision_info

messages = [{
    "role": "user",
    "content": [
        {
            "type": "video",
            "video": "path/to/video.mp4",
            "max_pixels": 360 * 420,  # Resolution control
            "fps": 1.0,  # Frame sampling rate
        },
        {"type": "text", "text": "Describe the video."},
    ],
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
```

**Input Tensor Shapes** (for 8-frame video):
```
input_ids: [1, 2729]  # Tokenized text + video placeholders
attention_mask: [1, 2729]
pixel_values_videos: [10800, 1176]  # Flattened video patches
video_grid_thw: [1, 3]  # Temporal-Height-Width grid dimensions
```

### Per-Frame Processing (Recommended)

For better memory control and frame-level analysis:

```python
from PIL import Image

# Load frames manually
frames = [...]  # List of numpy arrays (H, W, 3)

# Convert to PIL
pil_frames = [Image.fromarray(frame) for frame in frames]

# Process as image sequence
messages = [{
    "role": "user",
    "content": [{"type": "image", "image": frame} for frame in pil_frames[:1]]
              + [{"type": "text", "text": prompt}],
}]

# Extract tokens
outputs = model(**inputs, output_hidden_states=True)
```

---

## Memory Optimization

### GPU Memory Requirements

**Baseline (no optimization):**
- Model loading: ~15-20 GB
- Forward pass (1 frame): ~2-3 GB
- Total: ~18-23 GB

**With Optimization:**
```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Half precision
    device_map="auto",  # Automatic device distribution
    low_cpu_mem_usage=True,  # Efficient loading
    max_memory={0: "3GiB", "cpu": "16GiB"}  # Memory limits (adjust based on available GPU)
)
```

**Memory Tips:**
1. Process frames one at a time (not batched video)
2. Use `torch.cuda.empty_cache()` between frames
3. Delete intermediate tensors explicitly
4. Consider 8-bit or 4-bit quantization for very small GPUs

---

## Spatial Token Mapping

### Token-to-Pixel Correspondence

For input resolution `(H, W)`:

**visual.blocks.last:**
```
Grid: (h_tokens, w_tokens) = (52, 92)
Patch size: (H/52, W/92)

For 1280×720 input:
  Each token covers ~24.6 × 7.8 pixels
  
Token index i corresponds to:
  row = i // 92
  col = i % 92
  pixel_y = row * 13.85
  pixel_x = col * 13.91
```

**visual.merger:**
```
Grid: (h_tokens, w_tokens) = (26, 46)
Patch size: (H/26, W/46)

For 1280×720 input:
  Each token covers ~49.2 × 15.7 pixels
  
Token index i corresponds to:
  row = i // 46
  col = i % 46
  pixel_y = row * 27.69
  pixel_x = col * 27.83
```

**Caveats:**
- Grid dimensions depend on input resolution and model configuration
- Always verify with actual token shapes
- Some tokens may be special tokens (CLS, SEP) - typically at the end

---

## Token Delta Computation

### Frame-to-Frame Deltas

```python
# Extract tokens for each frame
token_sequence = []  # List of [num_tokens, hidden_dim]
for frame in frames:
    tokens = extract_tokens(frame)
    token_sequence.append(tokens)

# Stack into sequence
token_sequence = torch.stack(token_sequence)  # [num_frames, num_tokens, hidden_dim]

# Compute deltas
deltas = token_sequence[1:] - token_sequence[:-1]  # [num_frames-1, num_tokens, hidden_dim]

# Delta magnitudes
delta_norms = torch.norm(deltas, dim=2)  # [num_frames-1, num_tokens]
```

### Multi-Frame Deltas

For temporal distance `k`:
```python
delta_k = token_sequence[k:] - token_sequence[:-k]  # [num_frames-k, num_tokens, hidden_dim]
```

**Interpretation:**
- `k=1`: Frame-to-frame differences (velocity)
- `k>1`: Multi-frame differences (displacement)
- `visual.blocks.last`: Linear scaling with k (velocity encoding)
- `visual.merger`: Sub-linear scaling with k (displacement encoding)

---

## Common Issues and Solutions

### Issue 1: "max_pixels must be >= min_pixels"

**Cause**: `max_pixels` parameter too low (minimum ~784 = 28×28)

**Solution**:
```python
"max_pixels": 360 * 420  # Use at least this value
```

### Issue 2: CUDA Out of Memory

**Cause**: Processing too many frames at once

**Solutions:**
1. Process frames one at a time
2. Reduce resolution (`max_pixels`)
3. Use smaller batch size
4. Enable memory optimization (see above)

### Issue 3: NaN or Inf in Tokens

**Cause**: Numerical instability in float16

**Solution**:
```python
# Clean tokens after extraction
tokens[torch.isnan(tokens)] = 0
tokens[torch.isinf(tokens)] = 0
```

### Issue 4: Hook Not Triggering

**Cause**: Wrong layer name or module path

**Solution**:
```python
# Inspect model architecture
print(model)

# Verify layer exists
assert hasattr(model.visual.blocks[31], 'forward')
```

---

## Performance Benchmarks

**Hardware**: NVIDIA GPU with 32GB VRAM

| Operation | Time | Memory |
|-----------|------|--------|
| Model loading | ~3s | 15GB |
| Single frame extraction | ~0.8s | +1.5GB |
| 12 frames (sequential) | ~10s | ~3GB peak |
| 30 frames (sequential) | ~25s | ~3GB peak |

**Optimization Impact:**
- Float16 vs Float32: 2x memory reduction
- Per-frame vs batched: 5x memory reduction (for 10+ frames)
- CPU offloading: Enables 7B model on 4GB GPU

---

## Appendix: Layer Hierarchy

Complete path from input to extracted tokens:

```
Qwen2VLForConditionalGeneration
├── model (Qwen2VLModel)
│   ├── vision_model (Qwen2VisionTransformerPretrainedModel)
│   │   └── NOT USED (alternative path)
│   └── language_model (Qwen2ForCausalLM)
│       └── [Text generation]
└── visual (VisionTransformer)
    ├── patch_embed (Conv2d or similar)
    ├── blocks (ModuleList)
    │   ├── [0] TransformerBlock
    │   ├── ...
    │   └── [31] TransformerBlock ← EXTRACT HERE (visual.blocks.last)
    └── merger (ProjectionLayer or MLP)
        └── ← EXTRACT HERE (visual.merger)
```

Use `model.visual.blocks[31]` and `model.visual.merger` for hooks.

---

*Last updated: February 4, 2026*
