#!/usr/bin/env bash
# MMBench (EN) — LoRA fine-tuned checkpoint (merge base + adapter at load time).
#
# Prereq: same tsv as eval_mmbench_v15_7b_base.sh
#
# Set LORA_PATH to your training output_dir (the folder with adapter weights + config).
# Example:
#   export LORA_PATH=/path/to/checkpoints/llava-v1.5-7b-mydata-lora-small2399
#   bash scripts/eval_mmbench_v15_7b_lora.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LORA_PATH="${LORA_PATH:-}"
if [[ -z "${LORA_PATH}" ]]; then
  echo "Set LORA_PATH to your LoRA output directory, e.g.:"
  echo "  export LORA_PATH=/mnt/.../checkpoints/llava-v1.5-7b-mydata-lora-small2399"
  exit 1
fi

SPLIT="mmbench_dev_20230712"
QUESTION="${ROOT}/playground/data/eval/mmbench/${SPLIT}.tsv"
EXPERIMENT="llava-v1.5-7b-lora-mydata"
ANS_DIR="${ROOT}/playground/data/eval/mmbench/answers/${SPLIT}"
ANS_FILE="${ANS_DIR}/${EXPERIMENT}.jsonl"
UPLOAD_DIR="${ROOT}/playground/data/eval/mmbench/answers_upload/${SPLIT}"

if [[ ! -f "${QUESTION}" ]]; then
  echo "Missing: ${QUESTION}"
  echo "Download: https://download.openmmlab.com/mmclassification/datasets/mmbench/${SPLIT}.tsv"
  exit 1
fi

mkdir -p "${ANS_DIR}" "${UPLOAD_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m llava.eval.model_vqa_mmbench \
  --model-path "${LORA_PATH}" \
  --model-base liuhaotian/llava-v1.5-7b \
  --question-file "${QUESTION}" \
  --answers-file "${ANS_FILE}" \
  --single-pred-prompt \
  --temperature 0 \
  --conv-mode vicuna_v1

python "${ROOT}/scripts/convert_mmbench_for_submission.py" \
  --annotation-file "${QUESTION}" \
  --result-dir "${ANS_DIR}" \
  --upload-dir "${UPLOAD_DIR}" \
  --experiment "${EXPERIMENT}"

echo "Done. Submit ${UPLOAD_DIR}/${EXPERIMENT}.xlsx to OpenCompass MMBench leaderboard (see docs/Evaluation.md)."
