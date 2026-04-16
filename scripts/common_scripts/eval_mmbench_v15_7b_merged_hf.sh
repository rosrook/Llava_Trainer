#!/usr/bin/env bash
# MMBench (EN) - merged HF LLaVA checkpoint.
#
# Usage:
#   export MODEL_PATH=/mnt/.../train_full_data_hf_600
#   bash scripts/common_scripts/eval_mmbench_v15_7b_merged_hf.sh
#
# Optional:
#   export EXPERIMENT=llava-v1.5-7b-merged-step600
#   export MMBENCH_NUM_CHUNKS=1

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "Set MODEL_PATH to your merged HF model directory, e.g.:"
  echo "  export MODEL_PATH=/mnt/.../train_full_data_hf_600"
  exit 1
fi

SPLIT="mmbench_dev_20230712"
QUESTION="${ROOT}/playground/data/eval/mmbench/${SPLIT}.tsv"
EXPERIMENT="${EXPERIMENT:-llava-v1.5-7b-merged-hf}"
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
MMBENCH_VQA_EXTRA_ARGS=( --model-path "${MODEL_PATH}" )
eval_mmbench_v15_run_inference

python "${ROOT}/scripts/convert_mmbench_for_submission.py" \
  --annotation-file "${QUESTION}" \
  --result-dir "${ANS_DIR}" \
  --upload-dir "${UPLOAD_DIR}" \
  --experiment "${EXPERIMENT}"

echo "Done. Submit ${UPLOAD_DIR}/${EXPERIMENT}.xlsx to OpenCompass MMBench leaderboard (see docs/Evaluation.md)."
