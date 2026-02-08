Here are the exact steps. The dataset is 48.7 GB on HuggingFace and contains both the JSON annotations and most of the videos.

## Step 1: Clone the code repo

```bash
git clone https://github.com/marinero4972/Open-o3-Video
cd Open-o3-Video
conda create -n open-o3-video python=3.11
conda activate open-o3-video
bash setup.sh
```

## Step 2: Download the dataset from HuggingFace

The JSON annotations + videos are all at [`huggingface.co/datasets/marinero4972/Open-o3-Video`](https://huggingface.co/datasets/marinero4972/Open-o3-Video). Total is ~48.7 GB.

```bash
# Install huggingface CLI if you haven't
pip install huggingface_hub[cli]

# Login (you need a free HF account)
huggingface-cli login

# Download the entire dataset (~48.7 GB)
# This gets both json_data/ and videos/
huggingface-cli download marinero4972/Open-o3-Video \
    --repo-type dataset \
    --local-dir ./data/Open-o3-Video

# OR if you only want the JSON annotations first (much smaller, to inspect)
huggingface-cli download marinero4972/Open-o3-Video \
    --repo-type dataset \
    --include "json_data/*" \
    --local-dir ./data/Open-o3-Video
```

Alternatively using Python:

```python
from huggingface_hub import snapshot_download

# Full dataset
snapshot_download(
    repo_id="marinero4972/Open-o3-Video",
    repo_type="dataset",
    local_dir="./data/Open-o3-Video",
)

# OR just the JSONs first
snapshot_download(
    repo_id="marinero4972/Open-o3-Video",
    repo_type="dataset",
    local_dir="./data/Open-o3-Video",
    allow_patterns=["json_data/*"],
)
```

## Step 3: Understand what you downloaded

The structure will be:

```
data/Open-o3-Video/
├── json_data/
│   ├── STGR-RL.json      ← 36K samples, THIS IS WHAT YOU WANT for RL
│   └── STGR-SFT.json     ← 30K samples, for SFT cold-start
└── videos/
    ├── gqa/               ← GQA dataset frames/videos
    ├── stgr/
    │   ├── plm/           ← PLM-RDCap source videos
    │   └── temporal_grounding/  ← ActivityNet, COIN, etc.
    ├── timerft/           ← Time-R1 videos
    ├── treevgr/           ← TreeVGR videos
    ├── tvg_r1/            ← TVG-R1 videos
    ├── videoespresso/     ← VideoEspresso videos
    └── videor1/           ← Video-R1 videos
```

## Step 4: Inspect the data format

```python
import json

# Load the RL dataset (this is your primary interest)
with open("data/Open-o3-Video/json_data/STGR-RL.json") as f:
    rl_data = json.load(f)

print(f"Total RL samples: {len(rl_data)}")
print(f"Keys: {rl_data[0].keys()}")
print(json.dumps(rl_data[0], indent=2, default=str)[:2000])

# Check what fields are available
# Expected fields per sample:
#   - id, source, video_path, question, answer
#   - task (temporal / spatial / spatio-temporal / qa)
#   - key_frames: [{idx, path, time}, ...]  ← timestamped keyframes
#   - key_items: {object_name: [[bbox_coords], ...]}  ← spatial grounding
#   - image_path, image_size

# Count by task type
from collections import Counter
tasks = Counter(item['task'] for item in rl_data)
print(f"\nTask distribution: {tasks}")
# Expected: ~14% temporal, ~14% spatial, ~30% spatio-temporal, ~42% QA
```

## Step 5: Set the data root path

```bash
# Edit this file to point to your download location
nano src/r1-v/configs/data_root.py
```

Change the `DATA_ROOT` to wherever you downloaded:
```python
DATA_ROOT = "/path/to/data/Open-o3-Video"
```

## Step 6: Verify videos are accessible

```python
import json
import os

with open("data/Open-o3-Video/json_data/STGR-RL.json") as f:
    rl_data = json.load(f)

# Check how many video files actually exist
found = 0
missing = 0
missing_paths = set()

for item in rl_data[:500]:  # Check first 500
    vpath = os.path.join("data/Open-o3-Video", item.get("video_path", ""))
    if os.path.exists(vpath):
        found += 1
    else:
        missing += 1
        # Track which source folders are missing
        parts = item.get("video_path", "").split("/")
        if len(parts) > 1:
            missing_paths.add(parts[0] + "/" + parts[1] if len(parts) > 1 else parts[0])

print(f"Found: {found}, Missing: {missing}")
if missing_paths:
    print(f"Missing from: {missing_paths}")
```

If some videos are missing, they may need to be downloaded from original source datasets. The HuggingFace repo contains most of them (48.7 GB), but some larger source datasets (like ActivityNet videos) might only have keyframes included rather than full videos. The verification script above will tell you exactly what's missing.

## Quick start: just get the RL JSON and inspect

If you want to look at the data format immediately before committing to the full 48.7 GB download:

```bash
# Download just the two JSON files (~few hundred MB)
huggingface-cli download marinero4972/Open-o3-Video \
    --repo-type dataset \
    --include "json_data/*" \
    --local-dir ./data/Open-o3-Video

# Then inspect
python -c "
import json
with open('data/Open-o3-Video/json_data/STGR-RL.json') as f:
    data = json.load(f)
print(f'Samples: {len(data)}')
print(json.dumps(data[0], indent=2, default=str)[:3000])
"
```

This way you can verify the data format matches your pipeline needs before downloading all 48.7 GB of videos.