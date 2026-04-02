#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Export the original base model into a local standalone HF directory.
BASE_MODEL="liuhaotian/llava-v1.5-7b"
BASE_LOCAL_HF_DIR="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/Llava-v1.5-7b_hf/base_model"

echo "ROOT_DIR:          ${ROOT_DIR}"
echo "BASE_MODEL:        ${BASE_MODEL}"
echo "BASE_LOCAL_HF_DIR: ${BASE_LOCAL_HF_DIR}"

python3 "${ROOT_DIR}/scripts/merge_lora_weights.py" \
  --model-path "${BASE_MODEL}" \
  --save-model-path "${BASE_LOCAL_HF_DIR}"

echo "Done. Base model exported to: ${BASE_LOCAL_HF_DIR}"
