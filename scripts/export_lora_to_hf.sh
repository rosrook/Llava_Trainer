#!/usr/bin/env bash
# Step 1 of the LoRA -> MMBench eval pipeline.
#
# Export a LoRA training checkpoint to a standalone Hugging Face model dir.
# This script must run in the *training* conda env (llava_train: transformers
# 4.37, peft, deepspeed, etc.) because it imports llava.model.
# Step 2 (eval_my_checkpoint_mmbench.sh) runs in the *eval* env (llava_vlmevalkit)
# and consumes the HF dir produced here.
#
# Required env:
#   CHECKPOINT_PATH   abs path to a LoRA checkpoint dir (has adapter_config.json)
#
# Optional env:
#   BASE_MODEL        default: liuhaotian/llava-v1.5-7b
#   OUTPUT_DIR        default: <CKPT_PARENT>/_exported_hf/<CKPT_BASENAME>_hf
#                     This matches the path Step 2 auto-detects, so by default
#                     you can reuse the same CHECKPOINT_PATH between the two
#                     steps without needing to remember the HF dir.
#   FORCE_REEXPORT    set to 1 to wipe and re-export
#   BF16              set to 1 to load base model in bf16 (default: fp16)
#
# Examples:
#   conda activate llava_train
#   cd /home/zhuxuzhou/Llava_Trainer
#   CHECKPOINT_PATH=/mnt/tidal-alsh01/.../direct_5k/checkpoint-600 \
#     bash scripts/export_lora_to_hf.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAVA_REPO_ROOT="$(cd "${HERE}/.." && pwd)"
EXPORT_PY="${HERE}/export_lora_checkpoint_to_hf.py"

if [[ ! -f "${EXPORT_PY}" ]]; then
  echo "[export] ERROR: cannot find ${EXPORT_PY}" >&2
  echo "         Are you running this from the LLaVA training repo?" >&2
  exit 2
fi

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  echo "[export] ERROR: CHECKPOINT_PATH is required." >&2
  echo "         Example: CHECKPOINT_PATH=/path/to/checkpoint-600 bash $0" >&2
  exit 2
fi
CHECKPOINT_PATH="$(cd "${CHECKPOINT_PATH}" && pwd)"
if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "[export] ERROR: CHECKPOINT_PATH not a directory: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT_PATH}/adapter_config.json" ]]; then
  echo "[export] ERROR: ${CHECKPOINT_PATH} is not a LoRA checkpoint (missing adapter_config.json)." >&2
  echo "         If it is already a standalone HF dir, skip Step 1 and pass it directly to Step 2." >&2
  exit 3
fi

BASE_MODEL="${BASE_MODEL:-liuhaotian/llava-v1.5-7b}"
CKPT_PARENT="$(dirname "${CHECKPOINT_PATH}")"
CKPT_BASENAME="$(basename "${CHECKPOINT_PATH}")"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPT_PARENT}/_exported_hf/${CKPT_BASENAME}_hf}"
FORCE_REEXPORT="${FORCE_REEXPORT:-0}"

if [[ "${FORCE_REEXPORT}" == "1" && -d "${OUTPUT_DIR}" ]]; then
  echo "[export] FORCE_REEXPORT=1, removing ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi
if [[ -f "${OUTPUT_DIR}/config.json" ]]; then
  echo "[export] HF dir already exists, skipping export: ${OUTPUT_DIR}"
  echo "[export] (set FORCE_REEXPORT=1 to redo)"
else
  mkdir -p "$(dirname "${OUTPUT_DIR}")"
  echo "[export] base_model       = ${BASE_MODEL}"
  echo "[export] checkpoint       = ${CHECKPOINT_PATH}"
  echo "[export] output_dir       = ${OUTPUT_DIR}"
  echo "[export] precision        = $([[ "${BF16:-0}" == "1" ]] && echo bf16 || echo fp16)"

  EXTRA=()
  [[ "${BF16:-0}" == "1" ]] && EXTRA+=(--bf16)

  cd "${LLAVA_REPO_ROOT}"
  python3 "${EXPORT_PY}" \
    --base_model "${BASE_MODEL}" \
    --checkpoint_dir "${CHECKPOINT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    "${EXTRA[@]}"
fi

echo
echo "[export] DONE. HF model is at:"
echo "         ${OUTPUT_DIR}"
echo
echo "[export] Next: switch to the eval env and run Step 2."
echo "         conda activate llava_vlmevalkit"
echo "         cd <your xuzhou_vlmeval clone>"
echo "         CHECKPOINT_PATH=${CHECKPOINT_PATH} \\"
echo "           bash eval_my_checkpoint_mmbench.sh"
echo "         (Step 2 auto-detects the HF dir above; you can keep passing the"
echo "          original LoRA path as CHECKPOINT_PATH.)"
