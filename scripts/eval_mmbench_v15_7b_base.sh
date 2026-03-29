#!/usr/bin/env bash
# MMBench (EN) — original LLaVA-1.5-7B checkpoint (no LoRA).
#
# Prereq:
#   1) cd to LLaVA repo root
#   2) Download tsv to: ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv
#      https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv
#   3) pip install openpyxl pandas  (for convert step)
#
# Official eval uses greedy decoding (--temperature 0), same as README Evaluation section.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SPLIT="mmbench_dev_20230712"
QUESTION="${ROOT}/playground/data/eval/mmbench/${SPLIT}.tsv"
EXPERIMENT="llava-v1.5-7b-base"
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
  --model-path liuhaotian/llava-v1.5-7b \
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
