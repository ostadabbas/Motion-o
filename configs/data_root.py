"""
Data root configuration for STGR dataset.
"""

DATA_ROOT = "/mnt/data/stgr"

# Dataset structure:
# ${DATA_ROOT}/
# ├── json_data/
# │   ├── STGR-SFT.json     (31,166 samples)
# │   └── STGR-RL.json      (37,231 samples)
# └── videos/
#     ├── stgr/             (extracted from stgr.zip)
#     │   ├── plm/
#     │   └── temporal_grounding/
#     └── videoespresso/    (keyframes)
