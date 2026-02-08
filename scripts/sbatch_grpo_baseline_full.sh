#!/bin/bash
#SBATCH --job-name=motionr1_grpo_baseline
#SBATCH --output=logs/grpo_baseline_full_%j.out
#SBATCH --error=logs/grpo_baseline_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --partition=journey_gpu

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Navigate to project directory
cd /home/bi.ga/Workspace/vlmm-mcot

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dora_cuda

# Add workspace to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export WANDB_MODE="online"

# FULL TRAINING MODE
export QUICK_TEST="false"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Configuration
# Find the most recent SFT checkpoint
MODEL_PATH=$(ls -td outputs/sft_full_* | head -1)
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: No SFT checkpoint found! Please run SFT training first."
    exit 1
fi

EXP_NAME="rl_baseline_full_slurm_${SLURM_JOB_ID}"
OUT_DIR="outputs/${EXP_NAME}"
DATA_ROOT="/mnt/data/stgr"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-RL-subset.json"

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON (5,819 samples)"
echo "Output: $OUT_DIR"
echo "Rewards: Standard (NO motion_trajectory)"
echo "GPUs: 4 (DeepSpeed ZeRO-2 + LoRA)"
echo "Expected Duration: ~8-12 hours"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Baseline GRPO training WITHOUT motion_trajectory reward
srun torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12321 \
    training/train_grpo.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "$DATASET_JSON" \
    --deepspeed configs/zero2.json \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --bf16 true \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --attn_implementation eager \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name $EXP_NAME \
    --save_steps 500 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model true \
    --num_generations 2 \
    --reward_funcs ans_acc ans_tiou ans_viou thk_temporal_point thk_temporal_segment thk_spatial format

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job Complete!"
echo "=========================================="
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "Model saved to: $OUT_DIR"
echo "=========================================="

exit $EXIT_CODE
