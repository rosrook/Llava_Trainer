#!/usr/bin/env bash
# LLaVA-1.5-7B LoRA SFT on second_data4llava7B_ready_full.
# Sourced by: scripts/finetune_lora_llava_v1_5_7b_mydata.sh
#
# Run (from LLaVA repo root):
#   CONFIG_FILE=./scripts/configs/finetune_lora_second_data4llava7B_ready_full.sh \
#     bash scripts/common_scripts/run_finetune_lora_llava_v1_5_7b_mydata.sh

TRAIN_ENTRY=llava/train/train_mem.py

DATA_PATH="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/second_data4llava7B_ready_full/vqa_data.json"
IMAGE_FOLDER="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/second_data4llava7B_ready_full/images"

NUM_GPUS=8
DEEPSPEED_CONFIG="./scripts/zero3_logging.json"

# Base model starting point.
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

LORA_ENABLE=True
LORA_R=128
LORA_ALPHA=256
MM_PROJECTOR_LR=2e-5

OUTPUT_DIR="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/checkpoints/second_data4llava7B_ready_full"

REPORT_TO="tensorboard wandb"
LOGGING_DIR="${OUTPUT_DIR}/tb"
RUN_NAME="lora_second_data4llava7B_ready_full"
LOGGING_FIRST_STEP=True
LOG_LEVEL=info
LOGGING_NAN_INF_FILTER=True
SEED=42
SAVE_SAFETENSORS=True

NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"

EVALUATION_STRATEGY="no"
SAVE_STRATEGY="steps"
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=1

TF32=True
MODEL_MAX_LENGTH=2048
GRADIENT_CHECKPOINTING=True
DATALOADER_NUM_WORKERS=4
LAZY_PREPROCESS=True
