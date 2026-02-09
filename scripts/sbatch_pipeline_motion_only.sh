#!/bin/bash
# MotionR1 Training Pipeline (SFT + GRPO Motion Only)
# Skips baseline - just trains the motion-aware model

echo "=========================================="
echo "MotionR1 Training Pipeline (Motion Only)"
echo "=========================================="
echo "This will submit 2 jobs with dependencies:"
echo "1. SFT Training (5,544 samples WITH motion tags, ~6-8 hours)"
echo "2. GRPO Motion (5,819 samples, ~8-12 hours) - depends on SFT"
echo ""
echo "Dataset: Using motion-augmented data"
echo "Time limit: 7 days per job (167:59:00)"
echo "Total estimated time: ~14-20 hours"
echo "=========================================="
echo ""

# Navigate to project directory
cd /home/bi.ga/Workspace/vlmm-mcot

# Create logs directory
mkdir -p logs

# Submit SFT job
echo "Submitting SFT training job..."
SFT_JOB=$(sbatch --parsable scripts/sbatch_sft_full.sh)
echo "✓ SFT Job ID: $SFT_JOB"
echo ""

# Submit GRPO motion job (depends on SFT)
echo "Submitting GRPO Motion job (depends on SFT)..."
MOTION_JOB=$(sbatch --parsable --dependency=afterok:$SFT_JOB scripts/sbatch_grpo_motion_full.sh)
echo "✓ GRPO Motion Job ID: $MOTION_JOB"
echo ""

echo "=========================================="
echo "All jobs submitted successfully!"
echo "=========================================="
echo "Job Chain:"
echo "  SFT:         $SFT_JOB (running first)"
echo "  GRPO Motion: $MOTION_JOB (waits for SFT)"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $SFT_JOB,$MOTION_JOB"
echo ""
echo "View logs in real-time:"
echo "  tail -f logs/sft_full_${SFT_JOB}.out"
echo "  tail -f logs/grpo_motion_full_${MOTION_JOB}.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel $SFT_JOB $MOTION_JOB"
echo "=========================================="
