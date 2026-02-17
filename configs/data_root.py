"""
Data root configuration for Open-o3-Video / STGR dataset.

Training code expects:
  - video_root = DATA_ROOT/videos/  (videos and images live under here)
  - JSON paths: video_path / image_path are relative to video_root
"""

DATA_ROOT = "/scratch/bai.xiang/Open-o3-Video"

# Directory layout:
# ${DATA_ROOT}/
# ├── json_data/
# │   ├── STGR-RL.json          # 37,231 samples (RL training)
# │   └── STGR-SFT.json         # ~30k samples (SFT cold start)
# └── videos/
#     ├── gqa/                   # GQA images (*.jpg) - image_path has no prefix
#     ├── stgr/
#     │   ├── plm/videos/        # PLM videos (sav_*.mp4)
#     │   └── temporal_grounding/videos/
#     │       ├── activitynet/videos/*.mp4
#     │       ├── coin/videos/*.mp4
#     │       ├── DiDeMo/videos/*.mp4
#     │       ├── querYD/videos/*.mp4
#     │       └── qvhighlights/videos/*.mp4
#     ├── timerft/               # TimeR1 *.mp4
#     ├── treevgr/               # TreeVGR images
#     ├── videoespresso/kfs/     # Keyframes (images)
#     ├── Moviechat/, Youcook2/, CUVA/, XD-Violence/  # VideoEspresso source videos
#     ├── GroundedVLLM/activitynet/videos/, qvhighlights/videos/
#     ├── CLEVRER/, LLaVA-Video-178K/, NeXT-QA/, PerceptionTest/, STAR/  # symlinks
#     └── activitynet/, coin/, DiDeMo/, querYD/, qvhighlights/  # symlinks to stgr/temporal_grounding
