#!/bin/bash
# Complete MotionR1 Training Pipeline with SLURM Job Dependencies
# This script submits all three training jobs with proper dependencies

echo "=========================================="
echo "MotionR1 Complete Training Pipeline"
echo "=========================================="
echo "This will submit 3 jobs with dependencies:"
echo "1. SFT Training (5,696 samples, ~6-8 hours)"
echo "2. GRPO Baseline (5,819 samples, ~8-12 hours) - depends on SFT"
echo "3. GRPO Motion (5,819 samples, ~8-12 hours) - depends on SFT"
echo ""
echo "Total estimated time: ~24-32 hours"
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

# Submit GRPO baseline job (depends on SFT)
echo "Submitting GRPO baseline job (depends on SFT)..."
BASELINE_JOB=$(sbatch --parsable --dependency=afterok:$SFT_JOB scripts/sbatch_grpo_baseline_full.sh)
echo "✓ GRPO Baseline Job ID: $BASELINE_JOB"
echo ""

# Submit GRPO motion job (depends on SFT)
echo "Submitting GRPO motion job (depends on SFT)..."
MOTION_JOB=$(sbatch --parsable --dependency=afterok:$SFT_JOB scripts/sbatch_grpo_motion_full.sh)
echo "✓ GRPO Motion Job ID: $MOTION_JOB"
echo ""

echo "=========================================="
echo "All jobs submitted successfully!"
echo "=========================================="
echo "Job Chain:"
echo "  SFT:           $SFT_JOB (running first)"
echo "  GRPO Baseline: $BASELINE_JOB (waits for SFT)"
echo "  GRPO Motion:   $MOTION_JOB (waits for SFT)"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $SFT_JOB,$BASELINE_JOB,$MOTION_JOB"
echo ""
echo "View logs in real-time:"
echo "  tail -f logs/sft_full_${SFT_JOB}.out"
echo "  tail -f logs/grpo_baseline_full_${BASELINE_JOB}.out"
echo "  tail -f logs/grpo_motion_full_${MOTION_JOB}.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel $SFT_JOB $BASELINE_JOB $MOTION_JOB"
echo "=========================================="
