#!/usr/bin/env bash
set -euo pipefail

# Inference launcher for LoRA finetuned LLaVA-1.5-7B.
# It reuses project-native entrypoint: python3 -m llava.serve.cli
#
# Usage:
#   bash scripts/run_infer_lora_llava_v1_5_7b_mydata.sh /path/to/image.jpg
#   MODEL_PATH=/path/to/lora_ckpt IMAGE_FILE=/path/to/image.jpg bash scripts/run_infer_lora_llava_v1_5_7b_mydata.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Optional: set these on server for stable cache location.
# export HF_HOME=/path/to/hf_home
# export TRANSFORMERS_CACHE=/path/to/hf_home/hub

MODEL_BASE="${MODEL_BASE:-liuhaotian/llava-v1.5-7b}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/llava-v1.5-7b-mydata-lora}"
IMAGE_FILE="${IMAGE_FILE:-${1:-}}"
DEVICE="${DEVICE:-cuda}"
TEMPERATURE="${TEMPERATURE:-0.2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

if [[ -z "${IMAGE_FILE}" ]]; then
  echo "Error: IMAGE_FILE is required."
  echo "Example: bash scripts/run_infer_lora_llava_v1_5_7b_mydata.sh /path/to/test.jpg"
  exit 1
fi

echo "MODEL_BASE=${MODEL_BASE}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "IMAGE_FILE=${IMAGE_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python3 -m llava.serve.cli \
  --model-path "${MODEL_PATH}" \
  --model-base "${MODEL_BASE}" \
  --image-file "${IMAGE_FILE}" \
  --device "${DEVICE}" \
  --temperature "${TEMPERATURE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

