#!/bin/bash
# Complete PLM-STC Pipeline: Convert → Preprocess → Train
# Run this after sav_000.tar download completes

set -e

echo "========================================================================"
echo "PLM-STC COMPLETE PIPELINE"
echo "========================================================================"
echo

# Check if SA-V videos exist
if [ ! -d "/mnt/data/plm_stc/raw/sa-v" ] || [ -z "$(ls -A /mnt/data/plm_stc/raw/sa-v/*.mp4 2>/dev/null)" ]; then
    echo "ERROR: SA-V videos not found!"
    echo
    echo "Expected: /mnt/data/plm_stc/raw/sa-v/*.mp4"
    echo
    echo "Make sure sav_000.tar download and extraction completed."
    echo "Check: ls /mnt/data/plm_stc/raw/sa-v/*.mp4 | head"
    exit 1
fi

VIDEO_COUNT=$(ls /mnt/data/plm_stc/raw/sa-v/*.mp4 2>/dev/null | wc -l)
echo "✓ Found $VIDEO_COUNT videos in SA-V directory"
echo

# ============================================================================
# STEP 1: Convert Format
# ============================================================================
echo "========================================================================"
echo "STEP 1/3: Converting PLM-STC + SA-V to Training Format"
echo "========================================================================"
echo "Creating symlinks, decoding masklets, generating annotations..."
echo

python scripts/convert_plm_stc_to_format.py \
    --input-annotations /mnt/data/plm_stc/raw/rdcap \
    --input-videos /mnt/data/plm_stc/raw/sa-v \
    --output-dir /mnt/data/plm_stc/formatted_test \
    --limit 100

if [ $? -ne 0 ]; then
    echo "ERROR: Conversion failed!"
    exit 1
fi

echo
echo "✓ Step 1 complete!"
echo

# ============================================================================
# STEP 2: Preprocess for Training
# ============================================================================
echo "========================================================================"
echo "STEP 2/3: Preprocessing for Training"
echo "========================================================================"
echo "Extracting frames, computing motion descriptors..."
echo

python scripts/preprocess_plm_stc.py \
    /mnt/data/plm_stc/formatted_test \
    /mnt/data/plm_stc/preprocessed_test \
    --split train \
    --max-frames 8

if [ $? -ne 0 ]; then
    echo "ERROR: Preprocessing failed!"
    exit 1
fi

echo
echo "✓ Step 2 complete!"
echo

# ============================================================================
# STEP 3: Training Test (10 steps)
# ============================================================================
echo "========================================================================"
echo "STEP 3/3: Running Training Test"
echo "========================================================================"
echo "Training Qwen2.5-VL-7B with GRPO for 10 steps..."
echo "Using 4x V100 GPUs (1,2,3,4)"
echo

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

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

echo
echo "✓ Step 3 complete!"
echo

# ============================================================================
# Verification
# ============================================================================
echo "========================================================================"
echo "VERIFICATION"
echo "========================================================================"
echo

# Check if checkpoints exist
if [ -d "./outputs/plm_stc_test/checkpoint-10" ]; then
    echo "✓ Checkpoint saved at ./outputs/plm_stc_test/checkpoint-10"
else
    echo "⚠ Warning: Checkpoint not found"
fi

# Check logs
if [ -f "./outputs/plm_stc_test/trainer_log.txt" ]; then
    echo "✓ Training log available"
    echo
    echo "Last 20 lines of training log:"
    tail -20 ./outputs/plm_stc_test/trainer_log.txt
else
    echo "⚠ Warning: Training log not found"
fi

echo
echo "========================================================================"
echo "SUCCESS! PIPELINE COMPLETE"
echo "========================================================================"
echo
echo "What was done:"
echo "  ✓ Converted 100 PLM-STC samples to training format"
echo "  ✓ Preprocessed with 8 frames per video"
echo "  ✓ Trained Qwen2.5-VL-7B for 10 steps with GRPO"
echo
echo "Next steps:"
echo "  1. Test inference with trained model:"
echo "     python scripts/test_think_bbox_inference.py \\"
echo "         test_videos/Ball_Animation_Video_Generation.mp4 \\"
echo "         \"Describe the motion\" \\"
echo "         --model-id outputs/plm_stc_test/checkpoint-10 \\"
echo "         --strategies explicit_binding --num-frames 8"
echo
echo "  2. Scale up training:"
echo "     - Download more tar files for more videos"
echo "     - Increase max-steps to 1000+"
echo "     - Add validation split"
echo
echo "  3. Compare checkpoints:"
echo "     - Test checkpoint-5 vs checkpoint-10"
echo "     - Check if rewards increased"
echo "     - Verify bbox accuracy improved"
echo
