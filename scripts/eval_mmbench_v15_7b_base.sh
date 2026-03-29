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
#
# Multi-GPU: one process per GPU, data split via model_vqa_mmbench --num-chunks/--chunk-idx.
# Default chunk count = number of GPUs (`nvidia-smi -L`). Override: MMBENCH_NUM_CHUNKS=1 (single)
# or MMBENCH_NUM_CHUNKS=4. Optional MMBENCH_GPU_LIST=0,1,2,3 (physical ids, length >= chunks).

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

# shellcheck source=scripts/eval_mmbench_v15_7b_multigpu.sh
source "${ROOT}/scripts/eval_mmbench_v15_7b_multigpu.sh"
MMBENCH_VQA_EXTRA_ARGS=( --model-path liuhaotian/llava-v1.5-7b )
eval_mmbench_v15_run_inference

python "${ROOT}/scripts/convert_mmbench_for_submission.py" \
  --annotation-file "${QUESTION}" \
  --result-dir "${ANS_DIR}" \
  --upload-dir "${UPLOAD_DIR}" \
  --experiment "${EXPERIMENT}"

echo "Done. Submit ${UPLOAD_DIR}/${EXPERIMENT}.xlsx to OpenCompass MMBench leaderboard (see docs/Evaluation.md)."
