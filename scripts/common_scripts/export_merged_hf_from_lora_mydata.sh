#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Hard-coded paths for your setup.
# LoRA adapter directory produced by your finetune config:
ADAPTER_DIR="${ROOT_DIR}/checkpoints/llava-v1.5-7b-mydata-lora-small2399"
# Base model (HF repo id or local path):
BASE_MODEL="liuhaotian/llava-v1.5-7b"
# Output: a standalone HuggingFace model directory loadable via from_pretrained().
MERGED_DIR="${ROOT_DIR}/checkpoints/llava-v1.5-7b-mydata-lora-small2399-merged-hf"

echo "ROOT_DIR:    ${ROOT_DIR}"
echo "ADAPTER_DIR: ${ADAPTER_DIR}"
echo "BASE_MODEL:  ${BASE_MODEL}"
echo "MERGED_DIR:  ${MERGED_DIR}"

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "Error: ADAPTER_DIR not found: ${ADAPTER_DIR}"
  exit 1
fi

python3 "${ROOT_DIR}/scripts/merge_lora_weights.py" \
  --model-path "${ADAPTER_DIR}" \
  --model-base "${BASE_MODEL}" \
  --save-model-path "${MERGED_DIR}"

echo "Done. You can now load: ${MERGED_DIR}"
