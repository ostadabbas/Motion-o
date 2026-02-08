# Open-o3 Video: Infrastructure Analysis & Integration Plan

## Executive Summary

Open-o3 Video provides **excellent training infrastructure** that we should leverage. Their code is well-organized with:
- ✅ Multi-GPU distributed training (DeepSpeed ZeRO-2/3)
- ✅ Modular reward system (easy to extend)
- ✅ STGR dataset with spatial-temporal annotations
- ✅ Qwen2.5-VL / Qwen3-VL support
- ✅ Professional GRPO trainer with Group Sequence Policy Optimization

**Key Finding:** We can integrate our **motion-aware trajectory reward** by simply adding a new reward function to their modular reward system, keeping 95% of their infrastructure.

---

## 1. Open-o3 Video Architecture

### Directory Structure
```
Open-o3-Video/
├── src/
│   ├── r1-v/
│   │   ├── configs/
│   │   │   ├── data_root.py          # Dataset paths
│   │   │   ├── ddp.yaml              # Distributed training config
│   │   │   ├── zero2.yaml / zero3.yaml
│   │   ├── src/open_r1/
│   │   │   ├── data_loader.py        # Dataset loading
│   │   │   ├── reward_func.py        # ⭐ REWARD FUNCTIONS
│   │   │   ├── grpo.py               # Main training script
│   │   │   ├── sft_multi_task.py     # SFT cold-start
│   │   │   ├── trainer/
│   │   │   │   └── grpo_trainer.py   # ⭐ GRPO TRAINER (808 lines)
│   │   │   └── vision_process.py     # Video processing
│   │   └── local_scripts/
│   │       ├── zero2.json / zero3.json
│   └── scripts/
│       ├── run_sft_video.sh          # SFT training launcher
│       └── run_grpo_video.sh         # RL training launcher
├── eval/                             # Evaluation suite
│   ├── config/                       # Benchmark configs
│   ├── dataloader/                   # Benchmark loaders
│   ├── test/                         # Evaluation scripts
│   └── scripts/eval_all.sh
└── assets/                           # Demo videos
```

---

## 2. Their Reward System (reward_func.py)

### Current Reward Functions

Open-o3 has **7 reward functions** that are **summed** together:

```python
reward_funcs_registry = {
    "ans_acc": ans_acc_reward,           # Answer accuracy (ROUGE / exact match)
    "ans_tiou": ans_tiou_reward,         # Answer temporal IoU
    "ans_viou": ans_viou_reward,         # Answer visual IoU (bbox)
    "thk_temporal_point": thk_temporal_point_reward,    # Think temporal proximity
    "thk_temporal_segment": thk_temporal_segment_reward, # Think temporal segment
    "thk_spatial": thk_spatial_reward,   # Think spatial IoU (bbox)
    "format": format_reward              # Format validation
}

# In grpo.py:
reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
trainer = Qwen2VLGRPOTrainer(
    model=model_args.model_name_or_path,
    reward_funcs=reward_funcs,  # List of functions
    ...
)
```

### How Rewards Are Computed (grpo_trainer.py:645-658)

```python
# Compute the rewards
prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

for i, (reward_func, reward_processing_class) in enumerate(
    zip(self.reward_funcs, self.reward_processing_classes)
):
    # Call each reward function
    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

# Sum all rewards equally
rewards = rewards_per_func.sum(dim=1)
```

### Key Observations

1. **Modular Design**: Each reward function is independent
2. **Equal Weighting**: All rewards are summed without weights (unlike our λ system)
3. **Signature**: `reward_func(prompts, completions, **kwargs) -> list[float]`
4. **Static Spatial**: `thk_spatial_reward` only checks **per-frame bbox IoU**, NOT trajectory motion

---

## 3. What They Do Well (Keep)

### ✅ GRPO Trainer (808 lines, highly optimized)
- **Group Sequence Policy Optimization** with clipping
- **KL divergence penalty** for stability
- **DeepSpeed ZeRO-2/3** integration
- **Multi-GPU** distributed training
- **WandB logging**
- **Gradient checkpointing** for memory efficiency
- **Flash Attention 2** support

### ✅ Data Pipeline
- **STGR Dataset**: 30k SFT + 36k RL samples with spatial-temporal annotations
- **Multi-task support**: visual QA, temporal QA, temporal-spatial QA, MCQ, free-form
- **Video processing**: Efficient frame extraction with `process_vision_info`

### ✅ Format & Prompting
- **XML-like tags**: `<think>`, `<answer>`, `<obj>`, `<box>`, `<t>`
- **System prompts** tailored per task type
- **Format validation** reward

### ✅ Evaluation Suite
- V-STAR, VideoMME, VideoMMMU, WorldSense benchmarks
- Automated evaluation scripts

---

## 4. What They Miss (Our Contribution)

### ❌ Motion-Aware Trajectory Reward

**Their `thk_spatial_reward` (line 475-606):**
```python
def thk_spatial_reward(completions, **kwargs):
    # ...
    for claim in parsed_claims:
        pred_time = claim['timestamp']
        # Find closest GT frame
        # Compute bbox IoU for that single frame
        max_iou = calculate_iou(gt_box, pred_box)
        total_iou_score += max_iou
    
    return total_iou_score / len(parsed_claims)
```

**Problems:**
1. **Frame-independent**: Each bbox is evaluated in isolation
2. **No motion**: Doesn't check if object moved correctly between frames
3. **Static spatial matching only**: Just checks "is the bbox in the right place?"

**Our Solution:**
- **Trajectory-level reward**: Evaluate bbox sequences across time
- **Direction matching**: Cosine similarity of displacement vectors
- **Speed fidelity**: Velocity magnitude matching
- **Smoothness penalty**: Physically plausible motion constraints

### ❌ Think-Predict Bbox Refinement

**Their format:** Single bbox per evidence step
```xml
<obj>person</obj><box>[100,200,300,400]</box>at<t>2.5</t>s
```

**Our format:** Dual-phase bbox refinement
```
Step 1: [0.0s–2.5s] Description
  Think: (300,400),(500,700)    # Initial rough estimate
  Predict: (320,420),(480,680)  # Refined prediction
  Motion: velocity: 150px/s, direction: northeast
```

---

## 5. Integration Strategy

### Option A: Add Motion Reward to Their System (RECOMMENDED)

**Pros:**
- Minimal code changes (~200 lines)
- Leverage their infrastructure (trainer, data, eval)
- Direct comparison (with/without motion reward)

**Cons:**
- Must adapt to their format

**Steps:**
1. Add `motion_reward()` to `reward_func.py`
2. Import our `motion_metrics.py` functions
3. Register in `reward_funcs_registry`
4. Add to `script_args.reward_funcs` list

### Option B: Fork and Replace Trainer

**Pros:**
- Full control over training logic
- Can implement weighted rewards (λ system)

**Cons:**
- Lose their optimizations
- More maintenance burden

**Not recommended** - their trainer is excellent.

---

## 6. Detailed Integration Plan

### Step 1: Add Motion Reward Function

**File:** `Open-o3-Video/src/r1-v/src/open_r1/reward_func.py`

Add at the end:

```python
def motion_trajectory_reward(completions, **kwargs):
    """
    Motion-aware trajectory reward: Evaluates how objects moved, not just where they are.
    
    Computes:
    - Direction similarity (cosine of displacement vectors)
    - Speed fidelity (velocity magnitude matching)
    - Trajectory smoothness (acceleration penalty)
    """
    from motion_metrics import (
        compute_trajectory_iou,
        compute_direction_similarity,
        compute_speed_fidelity,
        compute_trajectory_smoothness
    )
    
    motion_rewards = []
    idx = 0
    
    for completion in completions:
        think_match = re.search(r"<think>(.*?)</think>", completion[0]["content"], re.DOTALL)
        
        if not think_match:
            motion_rewards.append(0.0)
            idx += 1
            continue
        
        think_content = think_match.group(1)
        
        # Parse predicted trajectory (bbox sequence across time)
        parsed_claims = parse_temporal_spatial_reasoning_process(think_content)
        
        if not parsed_claims or len(parsed_claims) < 2:
            # Need at least 2 frames for motion
            motion_rewards.append(0.0)
            idx += 1
            continue
        
        # Get GT trajectory
        gt_items = kwargs["key_items"][idx]
        gt_frames = kwargs["key_frames"][idx]
        fps = kwargs.get("fps", [30])[0]
        
        # Extract predicted bbox sequence
        pred_bboxes = []
        pred_times = []
        for claim in sorted(parsed_claims, key=lambda x: x['timestamp']):
            if claim['bboxes']:
                pred_bboxes.append(claim['bboxes'][0])  # Take first bbox
                pred_times.append(claim['timestamp'])
        
        if len(pred_bboxes) < 2:
            motion_rewards.append(0.0)
            idx += 1
            continue
        
        # Extract GT bbox sequence
        gt_bboxes = []
        gt_times = []
        for frame in sorted(gt_frames, key=lambda x: x['time']):
            frame_idx = str(frame["idx"])
            if frame_idx in gt_items and gt_items[frame_idx]:
                obj_key = list(gt_items[frame_idx].keys())[0]
                gt_bbox = gt_items[frame_idx][obj_key][0]
                gt_bbox = convert_coord_format(gt_bbox, kwargs["image_size"][idx])
                gt_bboxes.append(gt_bbox)
                gt_times.append(frame['time'])
        
        if len(gt_bboxes) < 2:
            motion_rewards.append(0.0)
            idx += 1
            continue
        
        # Compute motion metrics
        try:
            # 1. Direction similarity (displacement vectors)
            direction_score = compute_direction_similarity(pred_bboxes, gt_bboxes, fps)
            
            # 2. Speed fidelity (velocity matching)
            speed_score = compute_speed_fidelity(pred_bboxes, gt_bboxes, fps)
            
            # 3. Trajectory smoothness
            smoothness_score = compute_trajectory_smoothness(pred_bboxes, fps)
            
            # Combine motion components
            motion_reward = (
                0.4 * direction_score +
                0.4 * speed_score +
                0.2 * smoothness_score
            )
        except Exception as e:
            print(f"Error computing motion reward: {e}")
            motion_reward = 0.0
        
        motion_rewards.append(motion_reward)
        idx += 1
    
    return motion_rewards
```

### Step 2: Register Motion Reward

**File:** `Open-o3-Video/src/r1-v/src/open_r1/reward_func.py`

```python
# At the bottom, update registry:
reward_funcs_registry = {
    "ans_acc": ans_acc_reward,
    "ans_tiou": ans_tiou_reward,
    "ans_viou": ans_viou_reward,
    "thk_temporal_point": thk_temporal_point_reward,
    "thk_temporal_segment": thk_temporal_segment_reward,
    "thk_spatial": thk_spatial_reward,
    "motion_trajectory": motion_trajectory_reward,  # ⭐ NEW
    "format": format_reward
}
```

### Step 3: Copy Motion Metrics Module

```bash
cp ../vlmm-mcot/src/motion_metrics.py ./src/open_r1/
```

### Step 4: Enable Motion Reward in Training

**File:** `Open-o3-Video/src/r1-v/src/open_r1/grpo.py`

```python
@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: [
            "ans_acc", 
            "ans_tiou", 
            "ans_viou", 
            "thk_temporal_point", 
            "thk_temporal_segment", 
            "thk_spatial", 
            "motion_trajectory",  # ⭐ ADD THIS
            "format"
        ],
        metadata={"help": "List of reward functions"},
    )
```

### Step 5: Update Config

**File:** `Open-o3-Video/src/r1-v/configs/data_root.py`

```python
DATA_ROOT = "/mnt/data/stgr"  # Your actual path
```

### Step 6: Train

```bash
cd Open-o3-Video/src/r1-v

# SFT cold-start (use their original)
bash ../scripts/run_sft_video.sh

# RL with motion-aware rewards
bash ../scripts/run_grpo_video.sh
```

---

## 7. Comparison: Their Code vs Ours

| Component | Open-o3 | Ours | Best Choice |
|-----------|---------|------|-------------|
| **GRPO Trainer** | 808 lines, DeepSpeed, multi-GPU | 400 lines, basic | **Use theirs** ✅ |
| **Data Loader** | STGR dataset, multi-task | Generic HF | **Use theirs** ✅ |
| **Reward System** | Modular, 7 functions | Weighted (λ), 4 components | **Hybrid**: Use their system, add our motion reward ✅ |
| **Motion Metrics** | ❌ None | ✅ Direction, speed, smoothness | **Use ours** ✅ |
| **Bbox Format** | Single bbox | Think→Predict | **Optional**: Can add later |
| **Evaluation** | V-STAR, VideoMME, etc. | None | **Use theirs** ✅ |
| **Distributed Training** | DeepSpeed ZeRO-2/3 | None | **Use theirs** ✅ |

---

## 8. Expected Workflow

### Phase 1: Baseline (1 day)
1. ✅ Clone Open-o3 Video
2. ✅ Download STGR dataset
3. ✅ Run their SFT training (baseline)
4. ✅ Run their RL training (baseline)
5. ✅ Evaluate on V-STAR

### Phase 2: Add Motion Reward (2 days)
1. ✅ Copy `motion_metrics.py` to their codebase
2. ✅ Add `motion_trajectory_reward()` to `reward_func.py`
3. ✅ Register in `reward_funcs_registry`
4. ✅ Enable in `grpo.py`
5. ✅ Train with motion reward
6. ✅ Evaluate and compare

### Phase 3: Ablation Studies (2 days)
1. ✅ Train with only direction matching
2. ✅ Train with direction + speed
3. ✅ Train with full motion reward
4. ✅ Compare to baseline

### Phase 4: Think-Predict Format (optional, 3 days)
1. Update prompt template
2. Modify parser
3. Train and evaluate

---

## 9. Advantages of Using Their Infrastructure

### ✅ Production-Ready Training
- **Multi-GPU**: 8x A100 support out of the box
- **Memory efficient**: DeepSpeed ZeRO-3, gradient checkpointing
- **Stable**: KL penalty, gradient clipping, warmup
- **Fast**: Flash Attention 2, optimized data loading

### ✅ High-Quality Dataset
- **30k SFT samples**: Diverse tasks with spatial-temporal annotations
- **36k RL samples**: Motion-focused with keyframes and bboxes
- **Multiple domains**: GQA, TimeRFT, TVG, VideoEspresso, PLM

### ✅ Comprehensive Evaluation
- **V-STAR**: Spatio-temporal reasoning benchmark
- **VideoMME**: General video understanding
- **VideoMMMU**: Multi-modal understanding
- **WorldSense**: Temporal reasoning

### ✅ Professional Codebase
- **Clean**: Modular, well-documented
- **Maintained**: Active GitHub repo
- **Reproducible**: Pre-trained checkpoints available

---

## 10. Final Recommendation

**USE OPEN-O3 INFRASTRUCTURE + ADD OUR MOTION REWARD**

**Rationale:**
1. Their training code is **production-quality** (808-line GRPO trainer)
2. Their dataset (STGR) is **exactly what we need**
3. Their reward system is **modular** - easy to extend
4. We only need to add **~200 lines** for motion reward
5. We can **directly compare** with/without motion reward

**Time Savings:**
- Implementing their trainer from scratch: **2-3 weeks**
- Curating STGR-quality dataset: **4-6 weeks**
- Setting up evaluation: **1 week**
- **Total saved: ~8 weeks** by using their infrastructure

**Our Unique Contribution:**
- Motion-aware trajectory reward (direction, speed, smoothness)
- Trajectory-level metrics (not just per-frame IoU)
- Empirical validation on motion-heavy tasks

This is a **win-win**: We build on solid infrastructure while adding a novel, well-defined contribution.

---

## 11. Next Steps

1. ✅ Clone Open-o3 Video (done)
2. ⏳ Download STGR dataset
3. ⏳ Copy our `motion_metrics.py` to their codebase
4. ⏳ Implement `motion_trajectory_reward()` in `reward_func.py`
5. ⏳ Run baseline training (their rewards only)
6. ⏳ Run motion-aware training (+ our reward)
7. ⏳ Evaluate and compare results
8. ⏳ Write paper
