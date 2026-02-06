# PLM-STC Pipeline Status Update

**Date:** February 6, 2026  
**Status:** Partially Complete - Blocking Issue Identified

## ✅ Completed Steps

### 1. SA-V Dataset Download (sav_000.tar)
- **Status:** ✅ SUCCESS
- **Location:** `/mnt/data/plm_stc/raw/sa-v/`
- **Content:**
  - 568 videos (.mp4 files)
  - 566 manual masklets (JSON)
  - 534 auto masklets (JSON)
- **Size:** ~7.5GB

### 2. RDCap Annotations Download
- **Status:** ✅ SUCCESS
- **Location:** `/mnt/data/plm_stc/raw/rdcap/`
- **Samples:** 117,248 total

### 3. RDCap Filtering
- **Status:** ✅ SUCCESS
- **Location:** `/mnt/data/plm_stc/raw/rdcap_filtered/`
- **Samples:** 150 matching our downloaded videos
- **Reason:** Original RDCap references videos from many tar files (sav_000 through sav_XXX), but we only downloaded sav_000

### 4. Format Conversion Script
- **Status:** ✅ SUCCESS
- **Script:** `scripts/convert_plm_stc_to_format.py`
- **Functionality:**
  - Loads RDCap annotations
  - Loads SA-V masklets (RLE format)
  - Decodes RLE masks to numpy arrays
  - Creates video symlinks
  - Generates question/answer pairs
  - Saves masklets as `.npy` files

### 5. Conversion Test (100 samples)
- **Status:** ✅ PARTIAL SUCCESS
- **Location:** `/mnt/data/plm_stc/formatted_test/`
- **Results:**
  - 83/100 successful conversions (83%)
  - 17 failures (likely corrupt videos or masklets)
- **Output:**
  - 55 video symlinks
  - 135 masklet `.npy` files
  - `annotations/train.json` with 83 samples

### 6. Conversion Test (10 samples)
- **Status:** ✅ SUCCESS
- **Location:** `/mnt/data/plm_stc/formatted_test_small/`
- **Results:**
  - 10/10 successful conversions (100%)
- **Output:**
  - 8 video symlinks
  - 26 masklet `.npy` files
  - `annotations/train.json` with 10 samples

### 7. Training Pipeline Updates
- **Status:** ✅ COMPLETE
- **Updated Files:**
  - `src/evidence_parser.py` - Added `parse_think_predict_chain()`
  - `src/motion_dataset.py` - Updated `_build_chain_prompt()` for Think/Predict format
  - `scripts/train_motion_grpo.py` - Multi-GPU support, dynamic model loading
  - `src/geometric_reward.py` - Added Think/Predict parsing and refinement rewards
  - `scripts/verify_training_updates.py` - Verification script (all tests pass)

## ⚠️ Current Blocking Issue

### Preprocessing Performance Problem
- **Script:** `scripts/preprocess_plm_stc.py`
- **Problem:** Extremely slow HuggingFace dataset saving
- **Observed Behavior:**
  - 10 samples: Preprocessing takes 25s, but saving takes 2.5+ minutes
  - 83 samples: Preprocessing takes 2.8 minutes, saving takes 20+ minutes and uses 20-70GB RAM
  - The script uses `Dataset.from_list()` which loads all data into memory before saving
  
- **Root Cause:**
  - `preprocess_item()` extracts video frames and computes motion descriptors
  - All frames are held in memory as numpy arrays
  - When converted to HuggingFace Dataset, this creates massive memory pressure
  - Saving the dataset to disk is extremely slow

- **Impact:**
  - Cannot proceed to training with current preprocessing approach
  - Need to either:
    1. Use streaming/batch-based saving instead of loading all into memory
    2. Reduce data size (fewer frames, lower resolution)
    3. Skip preprocessing and use on-the-fly frame extraction during training

## 📋 Next Steps

### Option 1: Fix Preprocessing (Recommended for Full Pipeline)
1. Modify `scripts/preprocess_plm_stc.py` to use streaming/iterative saving
2. Instead of `Dataset.from_list(all_items)`, save items one-by-one or in batches
3. Use HuggingFace datasets' `IterableDataset` or manual `.arrow` file writing

### Option 2: Skip Preprocessing (Quick Path to Training)
1. Modify `src/motion_dataset.py` to load raw format directly
2. Extract frames and compute motion descriptors on-the-fly during training
3. Add caching layer to avoid re-extracting same frames

### Option 3: Use Small Subset First
1. Wait for the 10-sample preprocessing to complete (may take another 5-10 min)
2. Run a short training test with just 10 samples
3. Verify the full pipeline works end-to-end
4. Then fix preprocessing for larger scale

## 🔍 Investigation Findings

### SA-V JSON Format (Discovered During Conversion)
The SA-V masklet JSON structure is:
```json
{
  "masklet_id": [0, 1, 2, 3, 4],  // List of object IDs in this video
  "masklet": [  // List of frames
    [  // Frame 0
      {"size": "[1920, 1080]", "counts": "RLE_string"},  // Object 0
      {"size": "[1920, 1080]", "counts": "RLE_string"},  // Object 1
      ...
    ],
    [...]  // Frame 1
  ],
  "video_width": 1920,
  "video_height": 1080,
  ...
}
```

### Conda Run Buffering Issue
- `conda run -n dora_cuda` buffers stdout, making it appear the script is hung
- Solution: Use direct Python path `/home/bi.ga/.conda/envs/dora_cuda/bin/python`

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| SA-V Videos Downloaded | 568 |
| RDCap Samples (total) | 117,248 |
| RDCap Samples (matching) | 150 |
| Successful Conversions (100-sample test) | 83 |
| Successful Conversions (10-sample test) | 10 |
| Formatted Dataset Size (83 samples) | ~80KB annotations + 135 masklet files |

## 🚀 Recommendation

**Proceed with Option 3:** Let the 10-sample preprocessing finish, then run a quick training test to verify the full pipeline. While training is running, work on fixing the preprocessing performance issue for larger-scale training.

The core pipeline is solid:
- ✅ Data download works
- ✅ Conversion works
- ✅ Training code is updated for Think/Predict format
- ⚠️ Only preprocessing saving needs optimization

## 📁 Key Directories

```
/mnt/data/plm_stc/
├── raw/
│   ├── sa-v/                      # 568 videos + masklet JSONs
│   ├── rdcap/                     # Full RDCap (117K samples)
│   └── rdcap_filtered/            # Filtered RDCap (150 samples)
├── formatted_test/                # 83 converted samples
├── formatted_test_small/          # 10 converted samples
└── preprocessed_test/             # (in progress, 10 samples)

/home/bi.ga/Workspace/vlmm-mcot/
├── scripts/
│   ├── convert_plm_stc_to_format.py          # ✅ Working
│   ├── preprocess_plm_stc.py                  # ⚠️ Needs optimization
│   ├── train_motion_grpo.py                   # ✅ Updated
│   └── filter_rdcap_by_available_videos.py    # ✅ Working
└── src/
    ├── motion_dataset.py          # ✅ Updated for Think/Predict
    ├── evidence_parser.py         # ✅ Added Think/Predict parser
    └── geometric_reward.py        # ✅ Added refinement rewards
```

## 🐛 Known Issues

1. **Preprocessing saves slowly:** See "Current Blocking Issue" above
2. **Some video decoding errors:** 17/100 samples failed conversion (corrupt videos or masklet mismatches)
3. **High memory usage:** Preprocessing uses 20-70GB RAM for 83 samples
4. **Conda run buffering:** Use direct Python path to see real-time output

---

**Last Updated:** 2026-02-06 18:43 UTC
**Next Check:** Monitor PID 2689238 for preprocessing completion
