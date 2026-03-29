# shellcheck shell=bash
# Sourced by eval_mmbench_v15_7b_base.sh / eval_mmbench_v15_7b_lora.sh
#
# Before sourcing, set:
#   ROOT, QUESTION, ANS_DIR, ANS_FILE, EXPERIMENT
#   MMBENCH_VQA_EXTRA_ARGS=( ... )   # args for model_vqa_mmbench (--model-path, optional --model-base, etc.)
#
# Optional env:
#   MMBENCH_NUM_CHUNKS   if unset, uses visible GPU count from `nvidia-smi -L` (min 1); override e.g. 8 or 1
#   MMBENCH_GPU_LIST     comma-separated physical GPU ids, length >= NUM_CHUNKS (default 0,1,...,N-1)

eval_mmbench_v15_run_inference() {
  local NUM_CHUNKS
  if [[ -n "${MMBENCH_NUM_CHUNKS:-}" ]]; then
    NUM_CHUNKS="${MMBENCH_NUM_CHUNKS}"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NUM_CHUNKS="$(nvidia-smi -L | wc -l | tr -d '[:space:]')"
    [[ -z "${NUM_CHUNKS}" || "${NUM_CHUNKS}" -lt 1 ]] && NUM_CHUNKS=1
  else
    NUM_CHUNKS=1
  fi
  local -a _GPU_ARR

  if [[ -n "${MMBENCH_GPU_LIST:-}" ]]; then
    IFS=',' read -ra _GPU_ARR <<< "${MMBENCH_GPU_LIST// /}"
  else
    local _i
    for ((_i = 0; _i < NUM_CHUNKS; _i++)); do
      _GPU_ARR+=("${_i}")
    done
  fi

  if [[ "${#_GPU_ARR[@]}" -lt "${NUM_CHUNKS}" ]]; then
    echo "MMBENCH_NUM_CHUNKS=${NUM_CHUNKS} needs at least that many ids in MMBENCH_GPU_LIST (comma-separated); got ${#_GPU_ARR[@]}"
    exit 1
  fi

  if [[ "${NUM_CHUNKS}" -eq 1 ]]; then
    local g="${_GPU_ARR[0]:-0}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$g}" python -m llava.eval.model_vqa_mmbench \
      --question-file "${QUESTION}" \
      --answers-file "${ANS_FILE}" \
      --single-pred-prompt \
      --temperature 0 \
      --conv-mode vicuna_v1 \
      "${MMBENCH_VQA_EXTRA_ARGS[@]}"
    return
  fi

  # Parallel: one Python per chunk; do not rely on parent CUDA_VISIBLE_DEVICES.
  local -a pids=()
  local _i
  for ((_i = 0; _i < NUM_CHUNKS; _i++)); do
    local g="${_GPU_ARR[$_i]}"
    local part="${ANS_DIR}/${EXPERIMENT}_chunk${_i}.jsonl"
    CUDA_VISIBLE_DEVICES="${g}" python -m llava.eval.model_vqa_mmbench \
      --num-chunks "${NUM_CHUNKS}" \
      --chunk-idx "${_i}" \
      --question-file "${QUESTION}" \
      --answers-file "${part}" \
      --single-pred-prompt \
      --temperature 0 \
      --conv-mode vicuna_v1 \
      "${MMBENCH_VQA_EXTRA_ARGS[@]}" &
    pids+=("$!")
  done

  local _err=0 _pid
  for _pid in "${pids[@]}"; do
    wait "${_pid}" || _err=1
  done
  if [[ "${_err}" -ne 0 ]]; then
    echo "One or more MMBench chunk workers failed."
    exit 1
  fi

  : > "${ANS_FILE}"
  for ((_i = 0; _i < NUM_CHUNKS; _i++)); do
    cat "${ANS_DIR}/${EXPERIMENT}_chunk${_i}.jsonl" >> "${ANS_FILE}"
  done
}
