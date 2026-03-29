#!/usr/bin/env bash
set -euo pipefail

# Launcher for finetuning (loads scripts/configs/finetune_lora_small_vqa_2399.sh by default).
#
# Uses train_mem.py + flash-attn when TRAIN_ENTRY is unchanged in config. Fallback:
#   TRAIN_ENTRY=llava/train/train.py bash scripts/run_finetune_lora_llava_v1_5_7b_mydata.sh
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_finetune_lora_llava_v1_5_7b_mydata.sh
#   CONFIG_FILE=./scripts/configs/your_config.sh bash scripts/run_finetune_lora_llava_v1_5_7b_mydata.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export CONFIG_FILE="${CONFIG_FILE:-./scripts/configs/finetune_lora_small_vqa_2399.sh}"
export WANDB_PROJECT="${WANDB_PROJECT:-llava-finetune}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_WATCH="${WANDB_WATCH:-false}"
# W&B enabled by default; requires `wandb login`. To skip W&B on machines without API key: export WANDB_DISABLED=true
export WANDB_DISABLED="${WANDB_DISABLED:-false}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"

timestamp="$(date +"%Y%m%d_%H%M%S")"
log_file="${LOG_DIR}/finetune_lora_llava_v1_5_7b_mydata_${timestamp}.log"

echo "Working dir: ${ROOT_DIR}"
echo "Logging to:  ${log_file}"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES} (NUM_GPUS=${NUM_GPUS})"
echo "Config:      ${CONFIG_FILE}"
echo "WANDB:       disabled=${WANDB_DISABLED} project=${WANDB_PROJECT} mode=${WANDB_MODE}"

bash scripts/finetune_lora_llava_v1_5_7b_mydata.sh 2>&1 | tee "${log_file}"
