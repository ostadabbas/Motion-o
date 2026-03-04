#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --time=23:59:59
#SBATCH --job-name=grpo_dense_t07
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/grpo_dense_t07_%j.out
#SBATCH --error=logs/grpo_dense_t07_%j.err

set -euo pipefail

cd /projects/zura-storage/Workspace/vlmm-mcot
mkdir -p logs

source /projects/zura-storage/Workspace/dora/env_grpo/bin/activate
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DECORD_EOF_RETRY_MAX=20480
export WANDB_MODE="online"
export DEBUG_MOTION_REWARD=1

MODEL_PATH="outputs/motiono_sft_dense_4553555/merged"
EXP_NAME="grpo_dense_t07_${SLURM_JOB_ID}"
OUT_DIR="outputs/${EXP_NAME}"
DATA_ROOT="/scratch/bai.xiang/Open-o3-Video"
DATASET_JSON="${DATA_ROOT}/json_data/STGR-RL-dense-5k-mixed.json"

RESUME_ARG=""
if [ -d "$OUT_DIR" ]; then
    LATEST_CKPT=$(ls -d ${OUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
        echo "Resuming from: $LATEST_CKPT"
    fi
fi

echo "=========================================="
echo "GRPO — Dense SFT + temp=0.7"
echo "2x H200 141GB"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON"
echo "Output: $OUT_DIR"
echo "Resume: ${RESUME_ARG:-fresh start}"
echo "Start: $(date)"
echo "=========================================="

torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=12321 \
    training/train_grpo_v2.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "$DATASET_JSON" \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_generations 4 \
    --generation_batch_size 4 \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --max_pixels 401408 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation eager \
    --num_train_epochs 1 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --logging_steps 25 \
    --save_steps 200 \
    --save_only_model true \
    --report_to wandb \
    --run_name $EXP_NAME \
    --gen_temperature 0.7 \
    $RESUME_ARG \
    --reward_funcs ans_acc ans_tiou ans_viou thk_temporal_point thk_temporal_segment thk_spatial motion_trajectory format

echo ""
echo "Done! Model at: $OUT_DIR"
echo "End: $(date)"