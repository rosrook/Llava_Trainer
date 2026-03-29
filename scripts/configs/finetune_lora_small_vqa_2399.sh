#!/usr/bin/env bash
# Small VQA dataset preset (~2399 samples) for LLaVA-1.5-7B LoRA SFT.
# This file is sourced by:
#   scripts/finetune_lora_llava_v1_5_7b_mydata.sh
#
# Training entry: train_mem.py uses FlashAttention2 (requires flash_attn). Override if needed:
#   TRAIN_ENTRY=llava/train/train.py

# Training script (default: FlashAttention)
TRAIN_ENTRY=llava/train/train_mem.py

# Required data paths
DATA_PATH="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/first_data4llava7B/vqa_data.json"
IMAGE_FOLDER="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/first_data4llava7B"

# Hardware / runtime
NUM_GPUS=8
# ZeRO-3 + more frequent DeepSpeed console timing (loss / wall clock)
DEEPSPEED_CONFIG="./scripts/zero3_logging.json"

# Model setup
MODEL_NAME_OR_PATH="liuhaotian/llava-v1.5-7b"
PROMPT_VERSION="v1"
VISION_TOWER="openai/clip-vit-large-patch14-336"
MM_PROJECTOR_TYPE="mlp2x_gelu"
MM_VISION_SELECT_LAYER=-2
MM_USE_IM_START_END=False
MM_USE_IM_PATCH_TOKEN=False
IMAGE_ASPECT_RATIO="pad"
GROUP_BY_MODALITY_LENGTH=True
BF16=True

# LoRA setup
LORA_ENABLE=True
LORA_R=128
LORA_ALPHA=256
MM_PROJECTOR_LR=2e-5

# Output
OUTPUT_DIR="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/checkpoints/llava-v1.5-7b-mydata-lora-small2399"

# Logging & experiment tracking (requires: pip install tensorboard; for wandb: pip install wandb && wandb login)
# TensorBoard: tensorboard --logdir "${OUTPUT_DIR}/tb" --port 6006 --bind_all
# W&B offline: export WANDB_MODE=offline
REPORT_TO="tensorboard wandb"
LOGGING_DIR="${OUTPUT_DIR}/tb"
RUN_NAME="llava-v1.5-7b-lora-small2399"
LOGGING_FIRST_STEP=True
LOG_LEVEL=info
LOGGING_NAN_INF_FILTER=True
SEED=42
SAVE_SAFETENSORS=True

# Training hyperparameters for small data
NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"

# Logging / checkpoint
EVALUATION_STRATEGY="no"
SAVE_STRATEGY="steps"
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=1

# Other runtime args
TF32=True
MODEL_MAX_LENGTH=2048
GRADIENT_CHECKPOINTING=True
DATALOADER_NUM_WORKERS=4
LAZY_PREPROCESS=True

