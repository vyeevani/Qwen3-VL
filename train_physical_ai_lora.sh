#!/bin/bash
# LoRA training script for Physical AI autonomous driving dataset
# Memory-efficient fine-tuning using LoRA adapters

set -e

# ======================
# Configuration
# ======================

# Distributed training
export NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}

# Model configuration
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-2B-Instruct"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/physical_ai_driving_lora"}
CACHE_DIR=${CACHE_DIR:-"./cache"}

# Dataset
DATASET="physical_ai_driving"

# Training hyperparameters
LEARNING_RATE=2e-4  # Higher LR for LoRA
BATCH_SIZE=4  # Can use larger batch with LoRA
GRAD_ACCUM_STEPS=2
EPOCHS=5  # More epochs for LoRA

# LoRA configuration
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

# Video processing parameters
VIDEO_FPS=30
VIDEO_MAX_FRAMES=32
VIDEO_MIN_FRAMES=16
VIDEO_MAX_PIXELS=$((1024 * 28 * 28))
VIDEO_MIN_PIXELS=$((256 * 28 * 28))

# Image processing
MAX_PIXELS=$((576 * 28 * 28))
MIN_PIXELS=$((16 * 28 * 28))

# DeepSpeed configuration (optional for LoRA, can train without it)
DEEPSPEED_CONFIG="qwen-vl-finetune/scripts/zero2.json"

# Training entry
ENTRY_FILE="qwen-vl-finetune/qwenvl/train/train_qwen.py"

echo "========================================="
echo "Physical AI Driving Fine-tuning (LoRA)"
echo "========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE"
echo "LoRA config: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "Batch size: $BATCH_SIZE (grad_accum: $GRAD_ACCUM_STEPS)"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))"
echo "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# Launch training (using venv's torchrun)
.venv/bin/torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $ENTRY_FILE \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_use "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    \
    --deepspeed "$DEEPSPEED_CONFIG" \
    \
    --lora_enable True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    \
    --data_flatten True \
    --data_packing False \
    \
    --video_fps $VIDEO_FPS \
    --video_max_frames $VIDEO_MAX_FRAMES \
    --video_min_frames $VIDEO_MIN_FRAMES \
    --video_max_pixels $VIDEO_MAX_PIXELS \
    --video_min_pixels $VIDEO_MIN_PIXELS \
    \
    --max_pixels $MAX_PIXELS \
    --min_pixels $MIN_PIXELS \
    \
    --bf16 \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --num_train_epochs $EPOCHS \
    \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --optim "adamw_torch" \
    \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 25 \
    --save_total_limit 2 \
    \
    --dataloader_num_workers 2 \
    --report_to "none"

echo "========================================="
echo "LoRA training complete!"
echo "LoRA adapters saved to: $OUTPUT_DIR"
echo ""
echo "To merge LoRA weights back into the base model, use:"
echo "python merge_lora.py --base-model $MODEL_PATH --lora-path $OUTPUT_DIR --output merged_model/"
echo "========================================="
