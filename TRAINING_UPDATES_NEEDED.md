# Training Updates Required

Based on the successful "Think-then-Predict" scaffolding tests with the 7B model, here are the critical updates needed before GRPO training:

---

## ✅ What We Learned from Testing

1. **"Think-then-Predict" format works!** The model can generate varying bboxes when using:
   ```
   Step N: [time] Description
     Think: (x1,y1),(x2,y2) - rough estimate
     Predict: (x1,y1),(x2,y2) - refined bbox
     Motion: how the object moved
   ```

2. **Qwen native coordinates** `(x1,y1),(x2,y2)` on scale 0-1000 work better than JSON format

3. **7B model** generates motion-varying bboxes (2B was copying examples)

4. **Multi-GPU** works: 7B model splits across 4 V100s successfully

---

## 🔧 Required Code Updates

### 1. Update Prompt Format (`src/motion_dataset.py`)

**Current (broken)**:
```python
def _build_chain_prompt(self, question: str) -> str:
    prompt = f"""
**Task**: Analyze the video motion to answer this question: {question}

For each evidence step, provide:
1. **Time Interval**: [start_time–end_time]
2. **Object Detection**: JSON format:
   ```json
   {{"bbox_2d": [x1, y1, x2, y2], "label": "object_name"}}
   ```
```

**Update to (works!)**:
```python
def _build_chain_prompt(self, question: str, num_frames: int, frame_times: List[float]) -> str:
    """
    Build Think-then-Predict prompt that generates varying spatial evidence.
    
    Based on successful test_think_bbox_inference.py format.
    """
    # Build frame time listing
    frame_list = "\n".join([
        f"  Frame {i+1} at t={t:.2f}s" 
        for i, t in enumerate(frame_times)
    ])
    
    prompt = f"""{question}

You are shown {num_frames} frames from the video at these times:
{frame_list}

For EACH frame, locate relevant objects and provide:

Step N: [time] Description
  Think: (x1,y1),(x2,y2) - rough estimate
  Predict: (x1,y1),(x2,y2) - refined bbox
  Motion: how the object moved

Coordinates use 0-1000 scale (0=left/top, 1000=right/bottom).

Example format:
Step 1: [0.0s] Ball on left side
  Think: (150,400),(250,600)
  Predict: (160,420),(240,580)
  Motion: Starting position

Step 2: [1.0s] Ball moved right
  Think: (450,390),(550,590)
  Predict: (460,410),(540,570)
  Motion: Moved 300 units right, velocity 150 units/s

CRITICAL: Each step must have DIFFERENT coordinates showing actual object position in that frame.
Do NOT copy coordinates between steps!

After all steps: Answer: your final answer"""
    
    return prompt
```

---

### 2. Update Evidence Parser (`src/evidence_parser.py`)

**Add new parser for Think-Predict format**:

```python
@dataclass
class ThinkPredictStep:
    """Evidence step with Think-then-Predict bboxes."""
    step_num: int
    time: Optional[float]
    description: str
    think_bboxes: List[List[int]]  # Rough estimates
    pred_bboxes: List[List[int]]   # Refined predictions
    motion_text: str

def parse_think_predict_chain(text: str, img_width: int = 1280, img_height: int = 720) -> List[ThinkPredictStep]:
    """
    Parse Think-Predict format from model output.
    
    Format:
        Step 1: [0.0s] Description
          Think: (x1,y1),(x2,y2)
          Predict: (x1,y1),(x2,y2)
          Motion: motion description
    
    Returns:
        List of ThinkPredictStep objects with parsed bboxes
    """
    steps = []
    
    # Split by "Step N:"
    step_pattern = r'Step\s+(\d+):\s*\[([^\]]+)\]\s*([^\n]+)'
    step_matches = re.finditer(step_pattern, text, re.IGNORECASE)
    
    for match in step_matches:
        step_num = int(match.group(1))
        time_str = match.group(2).strip()
        description = match.group(3).strip()
        
        # Parse time
        time = None
        try:
            time = float(time_str.replace('s', ''))
        except:
            pass
        
        # Find Think and Predict bboxes for this step
        # Look ahead from current match
        start_pos = match.end()
        next_match = re.search(r'Step\s+\d+:', text[start_pos:], re.IGNORECASE)
        end_pos = start_pos + next_match.start() if next_match else len(text)
        step_content = text[start_pos:end_pos]
        
        # Extract Think bboxes
        think_bboxes = []
        think_pattern = r'Think:\s*\((\d+),(\d+)\),\((\d+),(\d+)\)'
        for bbox_match in re.finditer(think_pattern, step_content):
            x1, y1, x2, y2 = map(int, bbox_match.groups())
            # Convert from 0-1000 scale to pixels
            think_bboxes.append([
                int(x1 * img_width / 1000),
                int(y1 * img_height / 1000),
                int(x2 * img_width / 1000),
                int(y2 * img_height / 1000)
            ])
        
        # Extract Predict bboxes
        pred_bboxes = []
        pred_pattern = r'Predict:\s*\((\d+),(\d+)\),\((\d+),(\d+)\)'
        for bbox_match in re.finditer(pred_pattern, step_content):
            x1, y1, x2, y2 = map(int, bbox_match.groups())
            # Convert from 0-1000 scale to pixels
            pred_bboxes.append([
                int(x1 * img_width / 1000),
                int(y1 * img_height / 1000),
                int(x2 * img_width / 1000),
                int(y2 * img_height / 1000)
            ])
        
        # Extract motion text
        motion_pattern = r'Motion:\s*([^\n]+)'
        motion_match = re.search(motion_pattern, step_content)
        motion_text = motion_match.group(1).strip() if motion_match else ""
        
        steps.append(ThinkPredictStep(
            step_num=step_num,
            time=time,
            description=description,
            think_bboxes=think_bboxes,
            pred_bboxes=pred_bboxes,
            motion_text=motion_text
        ))
    
    return steps
```

---

### 3. Update Model Loading (`scripts/train_motion_grpo.py`)

**Current (broken)**:
```python
from transformers import AutoProcessor, AutoModelForVision2Seq  # ❌ Doesn't exist
```

**Update to**:
```python
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

def load_model_for_training(model_id: str, use_lora: bool = True, use_4bit: bool = False):
    """
    Load VLM model with proper class selection (not AutoModelForVision2Seq).
    
    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
        use_lora: Enable LoRA for efficient training
        use_4bit: Enable 4-bit quantization (optional with 4x V100-32GB)
    """
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Select correct model class
    mid = model_id.lower()
    if "qwen3" in mid:
        model_cls = Qwen3VLForConditionalGeneration
    elif "qwen2.5" in mid or "qwen2_5" in mid:
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        model_cls = Qwen2VLForConditionalGeneration
    
    # Configure quantization (optional)
    quant_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    # Load model with multi-GPU support
    model = model_cls.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map="auto",  # Automatically split across GPUs
        trust_remote_code=True,
    )
    
    # Setup LoRA if requested
    if use_lora:
        if use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, processor
```

---

### 4. Update GPU Selection

**Current**:
```python
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # ❌ GTX 745!
```

**Update to**:
```python
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"  # ✅ V100s only
```

---

### 5. Update Reward Calculation (`src/geometric_reward.py`)

**Need to handle both Think and Predict bboxes**:

```python
def compute_geometric_reward(
    generated_text: str,
    gt_evidence_steps: List[Dict],
    img_width: int = 1280,
    img_height: int = 720,
    weights: Dict[str, float] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute geometric reward for Think-Predict format.
    
    Rewards:
    1. Spatial IoU (Predict vs GT): Accuracy of refined prediction
    2. Temporal IoU: Time alignment
    3. Think→Pred refinement: Reward improvement from Think to Predict
    4. Motion similarity: Text similarity of motion descriptors
    5. Format compliance: Proper Think-Predict structure
    
    Returns:
        (total_reward, reward_breakdown)
    """
    from src.evidence_parser import parse_think_predict_chain
    
    if weights is None:
        weights = {
            "spatial_iou": 0.35,      # Main spatial grounding
            "temporal_iou": 0.15,     # Time alignment
            "refinement": 0.15,       # Think→Pred improvement
            "motion_sim": 0.20,       # Motion descriptor accuracy
            "format": 0.15,           # Proper structure
        }
    
    # Parse generated chain
    try:
        pred_steps = parse_think_predict_chain(generated_text, img_width, img_height)
        format_reward = 1.0 if len(pred_steps) > 0 else 0.0
    except Exception as e:
        return 0.0, {"error": "parse_failed", "spatial_iou": 0.0}
    
    # Initialize rewards
    spatial_ious = []
    temporal_ious = []
    refinement_scores = []
    motion_sims = []
    
    # Match predicted steps to GT steps
    for pred_step in pred_steps:
        # Find best matching GT step by time
        best_gt_step = find_closest_gt_step(pred_step, gt_evidence_steps)
        
        if best_gt_step and len(pred_step.pred_bboxes) > 0:
            # 1. Spatial IoU (Predict bbox vs GT bbox)
            pred_bbox = pred_step.pred_bboxes[0]
            gt_bbox = best_gt_step["bboxes"][0]
            spatial_iou = compute_bbox_iou(pred_bbox, gt_bbox)
            spatial_ious.append(spatial_iou)
            
            # 2. Temporal IoU
            if pred_step.time is not None:
                temp_iou = compute_temporal_iou(
                    pred_step.time, pred_step.time + 1.0,  # Assume 1s duration
                    best_gt_step["t_s"], best_gt_step["t_e"]
                )
                temporal_ious.append(temp_iou)
            
            # 3. Think→Pred refinement (reward bbox refinement)
            if len(pred_step.think_bboxes) > 0:
                think_bbox = pred_step.think_bboxes[0]
                think_iou = compute_bbox_iou(think_bbox, gt_bbox)
                pred_iou = spatial_iou
                refinement = max(0, pred_iou - think_iou)  # Reward improvement
                refinement_scores.append(refinement)
            
            # 4. Motion text similarity
            motion_sim = compute_text_similarity(
                pred_step.motion_text,
                best_gt_step.get("motion_desc", "")
            )
            motion_sims.append(motion_sim)
    
    # Aggregate rewards
    reward_breakdown = {
        "spatial_iou": np.mean(spatial_ious) if spatial_ious else 0.0,
        "temporal_iou": np.mean(temporal_ious) if temporal_ious else 0.0,
        "refinement": np.mean(refinement_scores) if refinement_scores else 0.0,
        "motion_sim": np.mean(motion_sims) if motion_sims else 0.0,
        "format": format_reward,
    }
    
    # Weighted sum
    total_reward = sum(reward_breakdown[k] * weights[k] for k in weights.keys())
    
    return total_reward, reward_breakdown
```

---

## 🚀 Recommended Training Configuration

### Model: Qwen2.5-VL-7B-Instruct
- Better spatial grounding than 2B
- Generates varying bboxes (2B was copying)
- Fits on 4x V100-32GB easily

### Hardware Setup:
```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4  # Use V100s, skip GTX 745
```

### Training Args:
```python
training_args = GRPOConfig(
    output_dir="./outputs/grpo_7b_think_predict",
    num_train_epochs=3,
    per_device_train_batch_size=1,      # Large images + 8 frames
    gradient_accumulation_steps=8,      # Effective batch size = 8
    learning_rate=5e-6,                 # Conservative for LoRA
    bf16=True,
    logging_steps=10,
    save_steps=100,
    max_prompt_length=2048,             # Long prompts with 8 frames
    max_completion_length=1536,         # Long chains with Think+Predict
    temperature=0.7,
    num_generations=4,                  # GRPO samples per prompt
    
    # Multi-GPU settings
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,        # Save memory
)
```

---

## 📋 Implementation Checklist

- [ ] Update prompt format in `src/motion_dataset.py` to Think-Predict
- [ ] Add `parse_think_predict_chain()` to `src/evidence_parser.py`
- [ ] Fix model loading in `scripts/train_motion_grpo.py` (remove AutoModelForVision2Seq)
- [ ] Update `compute_geometric_reward()` to handle Think+Predict bboxes
- [ ] Change default CUDA_VISIBLE_DEVICES to "1,2,3,4"
- [ ] Test updated pipeline on 1 example before full training
- [ ] Run full GRPO training with 7B model

---

## 🎯 Expected Results After Training

After GRPO training with this format, the model should:

1. ✅ Generate spatially varying bboxes (already does!)
2. ✅ Ground bboxes accurately to real object positions (GRPO will teach this)
3. ✅ Produce smooth, realistic motion trajectories (GRPO will enforce)
4. ✅ Refine Think→Predict (intermediate reasoning will improve)
5. ✅ Generate quantifiable motion descriptors (velocity, displacement)

The key insight: **The model can already generate the format and varying bboxes**. GRPO just needs to teach it to ground those bboxes in real visual evidence rather than plausible hallucinations.
