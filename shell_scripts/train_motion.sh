#!/bin/bash

# GRPO Training Script for Motion Reasoning
# Trains Qwen2.5-VL-7B on PLM-STC dataset with geometric rewards

# Dataset path (update this to your preprocessed PLM-STC dataset)
DATASET_PATH="${1:-/path/to/preprocessed/plm_stc/train}"

# Output directory
OUTPUT_DIR="./outputs/motion_grpo"

# Model configuration
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
USE_4BIT="--use-4bit"
USE_LORA="--use-lora"

# Training hyperparameters
NUM_GENERATIONS=8
MAX_STEPS=1000
SAVE_STEPS=100
BATCH_SIZE=4
GRADIENT_ACCUM_STEPS=4
LEARNING_RATE=1e-5

# Sequence lengths
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=512  # Evidence chains need more tokens
MAX_FRAMES=16

# RL parameters
KL_BETA=0.01
REWARD_WEIGHTS=1.0

# Motion-specific reward weights
LAMBDA_SPATIAL=0.25   # Bbox IoU
LAMBDA_TEMPORAL=0.15  # Interval IoU
LAMBDA_MOTION=0.35    # Trajectory match
LAMBDA_CAPTION=0.20   # Text similarity
LAMBDA_FORMAT=0.05    # Parseability gate

# FPS for motion computation
FPS=30.0

# System
DATALOADER_WORKERS=8
GENERATION_BATCH_SIZE=32  # batch_size * num_generations = 4 * 8

python scripts/train_motion_grpo.py "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model-id "$MODEL_ID" \
    $USE_4BIT \
    $USE_LORA \
    --num-generations $NUM_GENERATIONS \
    --max-steps $MAX_STEPS \
    --save-steps $SAVE_STEPS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUM_STEPS \
    --learning-rate $LEARNING_RATE \
    --max-prompt-length $MAX_PROMPT_LENGTH \
    --max-response-length $MAX_RESPONSE_LENGTH \
    --max-frames $MAX_FRAMES \
    --dataloader-num-workers $DATALOADER_WORKERS \
    --generation-batch-size $GENERATION_BATCH_SIZE \
    --kl-beta $KL_BETA \
    --reward-weights $REWARD_WEIGHTS \
    --fps $FPS \
    --lambda-spatial $LAMBDA_SPATIAL \
    --lambda-temporal $LAMBDA_TEMPORAL \
    --lambda-motion $LAMBDA_MOTION \
    --lambda-caption $LAMBDA_CAPTION \
    --lambda-format $LAMBDA_FORMAT
