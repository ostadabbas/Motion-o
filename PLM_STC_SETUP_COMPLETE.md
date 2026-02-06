# PLM-STC Dataset Setup - Status Report

## What We Accomplished ✅

### 1. Downloaded Annotations from HuggingFace
```
Source: facebook/PLM-Video-Human (rdcap subset)
Samples: 117,248 (from 42,068 unique videos)
Location: /mnt/data/plm_stc/raw/rdcap/
Status: ✅ COMPLETE
```

### 2. Created All Scripts
- `scripts/download_plm_stc_annotations.py` - ✅ Works, ran successfully
- `scripts/convert_plm_stc_to_format.py` - ✅ Ready to convert SA-V → training format
- `scripts/preprocess_plm_stc.py` - ✅ Already exists, tested
- `scripts/train_motion_grpo.py` - ✅ Updated with Think-Predict format
- `scripts/identify_needed_tar_files.py` - ✅ Identifies which videos to download

### 3. Identified Download Requirements
For 100-video test, you only need:
- **1 tar file**: `sav_000.tar` (~8GB)
- **Not 442GB**: No need to download full SA-V dataset
- **Analysis complete**: script shows exactly which file to get

### 4. Updated Training Pipeline
- ✅ Think-Predict format implemented
- ✅ Multi-GPU support (4x V100)
- ✅ Evidence parser handles new format
- ✅ Reward computation updated
- ✅ Verified with test videos

---

## What's Missing ⏸️

### SA-V Videos (Manual Download Required)

**Why manual?** Meta requires:
- Login with Meta/Facebook account
- Accept CC BY 4.0 license agreement
- Cannot be auto-downloaded (403 Forbidden without auth)

**What you need**:
- File: `sav_000.tar` (~8GB)
- Contains: 1000 videos (sav_000000 - sav_000999)
- For testing: We only use 100 of these videos

---

## How to Download SA-V

### Step 1: Go to Official Page
```
URL: https://ai.meta.com/datasets/segment-anything-video-downloads/
```

### Step 2: Sign In & Accept License
- Use Meta/Facebook account
- Read and accept CC BY 4.0 terms

### Step 3: Download File
Look for: **SA-V Training Videos** section
Download: `sav_000.tar` (first tar file, ~8GB)

Or if they provide a download script, run it:
```bash
# Example (adjust based on their interface)
python meta_download.py --file sav_000.tar --output /mnt/data/plm_stc/raw/
```

### Step 4: Extract
```bash
cd /mnt/data/plm_stc/raw/
tar -xvf sav_000.tar -C sa-v/
```

### Step 5: Verify
```bash
ls sa-v/*.mp4 | wc -l      # Should be ~1000
ls sa-v/*_manual.json | wc -l  # Should be ~1000
```

---

## After SA-V Download - Run Pipeline

Once you have videos in `/mnt/data/plm_stc/raw/sa-v/`, run these 3 commands:

### Command 1: Convert Format (~2 min)
```bash
python scripts/convert_plm_stc_to_format.py \
    --input-annotations /mnt/data/plm_stc/raw/rdcap \
    --input-videos /mnt/data/plm_stc/raw/sa-v \
    --output-dir /mnt/data/plm_stc/formatted_test \
    --limit 100
```
Creates: Symlinks to videos + decoded masklets + annotations

### Command 2: Preprocess (~5 min)
```bash
python scripts/preprocess_plm_stc.py \
    /mnt/data/plm_stc/formatted_test \
    /mnt/data/plm_stc/preprocessed_test \
    --split train \
    --max-frames 8
```
Creates: HuggingFace dataset with frames + motion descriptors

### Command 3: Train Test (~10 min)
```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4

python scripts/train_motion_grpo.py \
    /mnt/data/plm_stc/preprocessed_test/train \
    --output-dir ./outputs/plm_stc_test \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --use-lora \
    --max-steps 10 \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --num-generations 2 \
    --max-frames 8 \
    --save-steps 5 \
    --debug-reward
```
Trains: 10 steps to verify pipeline works

---

## Expected Results

After running all 3 commands, you should see:

✅ **Training completes without errors**
- Model loads on 4x V100s
- Generates Think-Predict format
- No OOM errors

✅ **Rewards are positive**
- Spatial IoU > 0
- Temporal IoU > 0  
- Motion similarity > 0
- Format is valid

✅ **Loss decreases**
- Even slight decrease proves learning
- Checkpoints saved at step 5 and 10

✅ **Inference works**
```bash
python scripts/test_think_bbox_inference.py \
    test_videos/Ball_Animation_Video_Generation.mp4 \
    "Describe the motion" \
    --model-id outputs/plm_stc_test/checkpoint-10 \
    --strategies explicit_binding \
    --num-frames 8
```
Model generates varying bboxes with motion descriptions

---

## File Locations

### Current State
```
/mnt/data/plm_stc/
├── raw/
│   ├── rdcap/                           ✅ Downloaded
│   ├── video_ids_needed.txt             ✅ Created (42,068 IDs)
│   ├── video_ids_test_100.txt           ✅ Created (100 IDs)
│   └── sa-v/                            ⏸️ Needs manual download
├── DOWNLOAD_INSTRUCTIONS.md             ✅ Created
├── HOW_TO_DOWNLOAD_SAV.md              ✅ Created
├── NEXT_STEPS.md                        ✅ Created
└── SETUP_STATUS.md                      ✅ Created

/home/bi.ga/Workspace/vlmm-mcot/
├── scripts/
│   ├── download_plm_stc_annotations.py  ✅ Created & ran
│   ├── identify_needed_tar_files.py     ✅ Created & ran
│   ├── download_sav_videos.sh           ✅ Created (403 error expected)
│   ├── convert_plm_stc_to_format.py     ✅ Ready
│   ├── preprocess_plm_stc.py            ✅ Ready
│   └── train_motion_grpo.py             ✅ Updated
└── PLM_STC_SETUP_COMPLETE.md            ✅ This file
```

### After SA-V Download
```
/mnt/data/plm_stc/
├── formatted_test/                      ⏳ After Command 1
│   ├── videos/ (100 symlinks)
│   ├── masklets/ (100+ .npy)
│   └── annotations/train.json
└── preprocessed_test/                   ⏳ After Command 2
    └── train/ (HF dataset)
```

---

## Timeline Estimate

| Step | Time | Status |
|------|------|--------|
| Annotations download | 1 min | ✅ Done |
| Script creation | 5 min | ✅ Done |
| SA-V download (manual) | 10-30 min | ⏸️ **YOU ARE HERE** |
| Conversion | 2 min | Waiting |
| Preprocessing | 5 min | Waiting |
| Training test | 10 min | Waiting |
| **TOTAL** | **~1 hour** | **50% done** |

---

## Key Insight

**The dataset is split across 2 sources** (not just HuggingFace):

1. **HuggingFace** = Annotations only (no videos)
2. **Meta SA-V** = Actual videos + masklets

This is common for large video datasets to avoid storing 442GB on HuggingFace!

---

## Documentation Created

All info you need is in:
- `/mnt/data/plm_stc/NEXT_STEPS.md` - What to do next
- `/mnt/data/plm_stc/HOW_TO_DOWNLOAD_SAV.md` - Detailed download guide
- `/mnt/data/plm_stc/DOWNLOAD_INSTRUCTIONS.md` - Quick reference
- `/mnt/data/plm_stc/SETUP_STATUS.md` - Current status

---

## Questions?

**Q: Can I test without downloading SA-V?**
A: Yes, use the ball video. But you'll need SA-V for real training.

**Q: Do I need all 442GB?**
A: No! Just 8GB (sav_000.tar) for 100-video test.

**Q: Can I download it programmatically?**
A: No, Meta requires manual login and license acceptance.

**Q: What if download fails?**
A: Check the official Meta download page for their current method.

---

## Bottom Line

✅ **What's ready**: Scripts, annotations, analysis (50% done)
⏸️ **What's needed**: Download `sav_000.tar` from Meta (~8GB, 10-30 min)
⏭️ **What's next**: 3 commands, ~15 minutes total

You're very close to having a working PLM-STC training pipeline!
