#!/bin/bash
#SBATCH --job-name=motion_debug
#SBATCH --output=logs/grpo_motion_debug_%j.out
#SBATCH --error=logs/grpo_motion_debug_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=0:45:00
#SBATCH --partition=journey_gpu

echo "=========================================="
echo "SLURM DEBUG JOB (QUICK_TEST)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=========================================="

cd /home/bi.ga/Workspace/vlmm-mcot

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dora_cuda

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_MODE="offline"

# Reuse existing QUICK_TEST - train_grpo.py will .select(max_samples) from dataset
export QUICK_TEST="true"
export MAX_SAMPLES="10"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

MODEL_PATH=$(ls -td outputs/sft_full_* 2>/dev/null | head -1)
[ -z "$MODEL_PATH" ] && MODEL_PATH=$(ls -td outputs/sft_* 2>/dev/null | head -1)
[ -z "$MODEL_PATH" ] && { echo "ERROR: No SFT checkpoint"; exit 1; }

EXP_NAME="rl_motion_debug_${SLURM_JOB_ID}"
OUT_DIR="outputs/${EXP_NAME}"
DATA_ROOT="/mnt/data/stgr"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-RL-subset.json"
mkdir -p logs

echo "=========================================="
echo "QUICK_TEST: 10 samples, 1 GPU"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON"
echo "Output: $OUT_DIR"
echo "=========================================="

srun torchrun --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    training/train_grpo.py \
    --output_dir "$OUT_DIR" \
    --model_name_or_path "$MODEL_PATH" \
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
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --bf16 true \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --attn_implementation eager \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --max_steps 20 \
    --run_name "$EXP_NAME" \
    --save_steps 10 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model true \
    --num_generations 2 \
    --reward_funcs ans_acc ans_tiou ans_viou thk_temporal_point thk_temporal_segment thk_spatial motion_trajectory format

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "DEBUG JOB END"
echo "=========================================="
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
