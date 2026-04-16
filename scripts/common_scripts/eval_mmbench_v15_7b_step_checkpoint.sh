#!/usr/bin/env bash
# MMBench (EN) - evaluate a specific training checkpoint directory by
# materializing a temporary LoRA-style adapter folder and then using the
# same loading path as eval_mmbench_v15_7b_lora.sh.
#
# Usage:
#   export CHECKPOINT_PATH=/mnt/.../second_data4llava7B_ready_full/checkpoint-600
#   export SHARED_DIR=/mnt/.../second_data4llava7B_ready_full
#   export EXPERIMENT=llava-v1.5-7b-step600
#   bash scripts/common_scripts/eval_mmbench_v15_7b_step_checkpoint.sh
#
# Optional:
#   export MODEL_BASE=liuhaotian/llava-v1.5-7b
#   export STEP_EVAL_WORK_DIR=/mnt/.../tmp_eval_step600
#   export MMBENCH_NUM_CHUNKS=1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
SHARED_DIR="${SHARED_DIR:-}"
MODEL_BASE="${MODEL_BASE:-liuhaotian/llava-v1.5-7b}"
STEP_EVAL_WORK_DIR="${STEP_EVAL_WORK_DIR:-}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH, e.g.:"
  echo "  export CHECKPOINT_PATH=/mnt/.../second_data4llava7B_ready_full/checkpoint-600"
  exit 1
fi
if [[ -z "${SHARED_DIR}" ]]; then
  echo "Set SHARED_DIR, e.g.:"
  echo "  export SHARED_DIR=/mnt/.../second_data4llava7B_ready_full"
  exit 1
fi
if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Missing CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
  exit 1
fi
if [[ ! -d "${SHARED_DIR}" ]]; then
  echo "Missing SHARED_DIR: ${SHARED_DIR}"
  exit 1
fi

checkpoint_name="$(basename "${CHECKPOINT_PATH}")"
if [[ -z "${STEP_EVAL_WORK_DIR}" ]]; then
  STEP_EVAL_WORK_DIR="${SHARED_DIR}/.eval_${checkpoint_name}"
fi
MATERIALIZED_DIR="${STEP_EVAL_WORK_DIR}/llava-v1.5-7b-lora-${checkpoint_name}"

echo "ROOT:             ${ROOT}"
echo "CHECKPOINT_PATH:  ${CHECKPOINT_PATH}"
echo "SHARED_DIR:       ${SHARED_DIR}"
echo "MODEL_BASE:       ${MODEL_BASE}"
echo "WORK_DIR:         ${STEP_EVAL_WORK_DIR}"
echo "MATERIALIZED_DIR: ${MATERIALIZED_DIR}"

rm -rf "${STEP_EVAL_WORK_DIR}"
mkdir -p "${MATERIALIZED_DIR}"

# Copy checkpoint-local adapter files
cp -a "${CHECKPOINT_PATH}/." "${MATERIALIZED_DIR}/"

# Copy shared files expected by the LLaVA LoRA loading path
for f in config.json non_lora_trainables.bin; do
  if [[ -f "${SHARED_DIR}/${f}" ]]; then
    cp -a "${SHARED_DIR}/${f}" "${MATERIALIZED_DIR}/"
  fi
done

echo "Materialized files:"
ls -lah "${MATERIALIZED_DIR}"

SPLIT="mmbench_dev_20230712"
QUESTION="${ROOT}/playground/data/eval/mmbench/${SPLIT}.tsv"
EXPERIMENT="${EXPERIMENT:-llava-v1.5-7b-${checkpoint_name}}"
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
MMBENCH_VQA_EXTRA_ARGS=( --model-path "${MATERIALIZED_DIR}" --model-base "${MODEL_BASE}" )
eval_mmbench_v15_run_inference

python "${ROOT}/scripts/convert_mmbench_for_submission.py" \
  --annotation-file "${QUESTION}" \
  --result-dir "${ANS_DIR}" \
  --upload-dir "${UPLOAD_DIR}" \
  --experiment "${EXPERIMENT}"

echo "Done. Materialized checkpoint eval results:"
echo "  answers : ${ANS_FILE}"
echo "  upload  : ${UPLOAD_DIR}/${EXPERIMENT}.xlsx"
echo "Temporary materialized dir kept at: ${MATERIALIZED_DIR}"
