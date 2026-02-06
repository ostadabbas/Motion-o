#!/bin/bash
# Run think_bbox test with larger model on multi-GPU setup

# Set which GPUs to use (1,2,3,4 for the V100s - skip GTX 745 on GPU 0)
export CUDA_VISIBLE_DEVICES=1,2,3,4

# Model options:
# - Qwen/Qwen2.5-VL-7B-Instruct (recommended - newer, better grounding)
# - Qwen/Qwen2-VL-7B-Instruct (older version)
# - Qwen/Qwen3-VL-8B-Instruct (newest, largest - may need 4bit)

MODEL_ID="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
VIDEO="${2:-test_videos/Ball_Animation_Video_Generation.mp4}"
QUESTION="${3:-Describe the motion trajectory of the red ball}"

echo "================================================================================"
echo "MULTI-GPU TEST WITH LARGER MODEL"
echo "================================================================================"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_ID"
echo "Video: $VIDEO"
echo "================================================================================"
echo ""

conda run -n dora_cuda python scripts/test_think_bbox_inference.py \
    "$VIDEO" \
    "$QUESTION" \
    --model-id "$MODEL_ID" \
    --num-frames 8 \
    --device auto \
    --use-4bit \
    --multi-gpu \
    --strategies explicit_binding freeform_chain \
    --max-tokens 1536 \
    --output-dir "outputs/multi_gpu_test_$(basename $MODEL_ID)"

echo ""
echo "================================================================================"
echo "Test complete! Check outputs/multi_gpu_test_*/"
echo "================================================================================"
