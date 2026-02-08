# Quick Dataset Download Instructions

## TL;DR - Fastest Path to Get Started

### Option 1: Contact Open-o3 Video Authors (Recommended)

The STGR dataset is curated by the Open-o3 Video team. The fastest way to get it:

1. **Visit their GitHub:**
   ```
   https://github.com/marinero4972/Open-o3-Video
   ```

2. **Check the README** for data download links (look for HuggingFace or Google Drive links)

3. **If links are not public**, open an issue:
   ```
   https://github.com/marinero4972/Open-o3-Video/issues
   ```
   
   Template:
   ```
   Title: Request access to STGR dataset
   
   Hi, I'm working on motion-aware trajectory reasoning for video understanding
   and would like to train on the STGR dataset mentioned in your paper.
   
   Could you please share:
   - Download links for STGR-SFT.json and STGR-RL.json
   - Instructions for downloading the video files
   
   Thank you!
   ```

### Option 2: Use HuggingFace Dataset (If Available)

Check if STGR is on HuggingFace:

```bash
# Try loading from HuggingFace
python -c "
from datasets import load_dataset
try:
    ds = load_dataset('marinero4972/STGR', split='train')  # Might not exist yet
    print('✅ Found on HuggingFace!')
except:
    print('❌ Not on HuggingFace - use Option 1')
"
```

### Option 3: Build from Source Datasets

If STGR is not directly available, build it from source datasets:

1. **Download base datasets:**
   - GQA: https://cs.stanford.edu/people/dorarad/gqa/download.html
   - ActivityNet: http://activity-net.org/download.html
   - SA-V (PLM): https://ai.meta.com/datasets/segment-anything-video/

2. **Use Open-o3's annotation format** (see their paper appendix)

3. **Create STGR-style JSON** following their schema

## Practical Steps (What to Do Right Now)

### Step 1: Try to Find the Dataset

```bash
# Check Open-o3 Video README
curl -s https://raw.githubusercontent.com/marinero4972/Open-o3-Video/main/README.md | grep -i "data\|download\|dataset" -A 5

# Check their HuggingFace profile
curl -s https://huggingface.co/marinero4972 | grep -i "dataset"
```

### Step 2: If Found, Download

```bash
# Create directory
mkdir -p /mnt/data/stgr/{json_data,videos}

# Download JSON files (adjust URL)
wget <STGR-SFT-URL> -O /mnt/data/stgr/json_data/STGR-SFT.json
wget <STGR-RL-URL> -O /mnt/data/stgr/json_data/STGR-RL.json

# Download videos (follow their instructions)
# This will likely be multiple tar/zip files
```

### Step 3: Verify Download

```bash
# Check what you have
python scripts/download_stgr.py --check-only --output-dir /mnt/data/stgr

# Test loading
python -c "
import json
with open('/mnt/data/stgr/json_data/STGR-SFT.json') as f:
    data = json.load(f)
    print(f'✅ Loaded {len(data)} SFT samples')
"
```

### Step 4: Update Config

```bash
# Update configs/data_root.py
echo 'DATA_ROOT = "/mnt/data/stgr"' > configs/data_root.py
```

## Interim Solution: Use Existing Datasets

While waiting for STGR access, you can test the training pipeline with:

### Option A: Use ActivityNet Captions + COCO

```python
# scripts/create_interim_dataset.py
from datasets import load_dataset

# Load ActivityNet Captions
activitynet = load_dataset('activity_net', 'captions')

# Convert to STGR format
# (You'll need to add spatial annotations manually or use pre-trained detector)
```

### Option B: Use Video-ChatGPT Dataset

```bash
# Download Video-ChatGPT
git clone https://github.com/mbzuai-oryx/Video-ChatGPT
cd Video-ChatGPT/data

# Convert to STGR format
python ../../scripts/convert_videochatgpt_to_stgr.py
```

### Option C: Create Synthetic Dataset (Testing Only)

```python
# For pipeline testing only (not for real training)
python scripts/create_toy_dataset.py \
    --num-samples 1000 \
    --output-dir /tmp/stgr_toy
```

## What Each Dataset Component Looks Like

### STGR-SFT.json Format (Expected)

```json
{
  "video_id": "gqa_012345",
  "video_path": "videos/gqa/012345.mp4",
  "question": "What happens in this video?",
  "answer": "A person walks across the room and picks up a ball.",
  "task": "temporal-spatial free-form QA",
  "key_frames": [
    {
      "idx": 0,
      "time": 0.5,
      "frame_path": "videos/gqa/012345_frame0.jpg"
    }
  ],
  "key_items": {
    "0": {
      "person": [[0.2, 0.3, 0.5, 0.8]],  # normalized bbox
      "ball": [[0.6, 0.7, 0.7, 0.8]]
    }
  },
  "image_size": [1920, 1080]
}
```

## Current Status Check

Run this to see what you need:

```bash
# Check if you have any data
ls -lh /mnt/data/stgr/json_data/ 2>/dev/null || echo "❌ No JSON data yet"
ls -lh /mnt/data/stgr/videos/ 2>/dev/null || echo "❌ No video data yet"

# Check disk space
df -h /mnt/data

# Check network speed
speedtest-cli  # or: curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python -
```

## Estimated Timeline

| Task | Time | Notes |
|------|------|-------|
| Contact authors | 1-7 days | Wait for response |
| Download JSON | 10 mins | ~100MB |
| Download videos | 2-4 hours | ~80GB @ 100Mbps |
| Verify & setup | 30 mins | Run checks |
| **Total (fast path)** | **3-5 hours** | If links available |
| **Total (slow path)** | **1-2 weeks** | If need to build from scratch |

## Priority Actions (Do This Now)

1. ✅ **Open an issue** on Open-o3 Video GitHub (do this first!)
2. ✅ **Check their README** for any update on data release
3. ✅ **Prepare storage**: Ensure 200GB free on `/mnt/data`
4. ✅ **Test pipeline**: Use toy dataset to verify training works
5. ⏳ **Wait for dataset**: Meanwhile, prepare your training setup

## Fallback Plan

If STGR is not accessible:

1. **Use their baseline model** (if they release checkpoints)
2. **Train on ActivityNet + COCO** with your motion reward
3. **Collect your own data** (small scale, specific domain)
4. **Synthesize data** using GPT-4V to generate motion descriptions

The key insight is: **Your motion reward is the novel contribution**, not the dataset. You can demonstrate it on any dataset with temporal-spatial annotations.

## Need Help?

- Our project: See `DATASET_DOWNLOAD.md` for full details
- Open-o3 Video: https://github.com/marinero4972/Open-o3-Video
- Contact: Check paper for author emails
