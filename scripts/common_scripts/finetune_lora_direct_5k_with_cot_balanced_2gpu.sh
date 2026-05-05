#!/usr/bin/env bash
# ==============================================================================
# LLaVA-1.5-7B LoRA SFT on direct_5k + Chain-of-Thought, "balanced" recipe.
#
# Baseline pair (clean ablation, only the GPT-side answer differs):
#   finetune_lora_direct_5k_balanced_2gpu.sh
#       (gpt = single letter, e.g. "B")                  -- LETTER baseline
#   finetune_lora_direct_5k_with_cot_balanced_2gpu.sh
#       (gpt = "<letter>\n<explanation>", e.g. CoT)      -- THIS script
#
# Both use IDENTICAL hyperparameters (the explain_balanced recipe), so any
# downstream-eval delta can be attributed to the answer format alone.
#
# Source data is the ShareGPT-style JSONL produced upstream by the cast_vqa
# pipeline, which (per the upstream "sft_with_cot" mode) takes each direct_5k
# QA and APPENDS an explanation to the GPT response while keeping the human
# turn and image reference unchanged. Image filenames (basenames) still
# resolve into the original direct_5k images folder.
#
# Verification done (2026-05-05):
#   intersection-by-image-filename = 4999 / 5000   (4999 ⊂ direct_5k)
#   spot-check 3 samples: human turn byte-equal, new gpt starts with old gpt
#   so the only variable is the appended CoT explanation.
#
# Quirks of the upstream JSONL (handled below):
#   1. ``image`` field is an ABSOLUTE path into an OpenImages cache, which
#      preflight_lora_dataset.py rejects via path-escape. We pre-normalize it
#      to basename via scripts/normalize_sharegpt_image_paths.py.
#   2. Every row has the literal id "direct_qa" (the upstream wrote the
#      template-id into the id slot). LLaVA's data loader does not use ``id``
#      so this is harmless for training.
#   3. New file has 4999 rows (one less than direct_5k's 5000); negligible.
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# ---- GPUs / launcher ---------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=2
MASTER_PORT="${MASTER_PORT:-29541}"

# ---- Data / output paths -----------------------------------------------------
# Data: the upstream-prepared CoT-augmented JSONL. (Override DATA_PATH /
# IMAGE_FOLDER from the env if you ever rebuild the upstream output to a
# different location.)
DATA_PATH="${DATA_PATH:-/home/zhuxuzhou/cast_vqa/outputs/direct_qa_baseline_5k_v1/sft_with_cot/sft_with_cot_sharegpt.jsonl}"

# Images: reuse the EXACT same folder as the LETTER baseline. The CoT pipeline
# only appends to the GPT turn; image filenames are unchanged.
IMAGE_FOLDER="${IMAGE_FOLDER:-/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/bysj_first_run/ok_for_training/direct_5k/images}"

OUTPUT_DIR="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/bysj/Llava_v1_5_7B/direct_5k_with_cot_balanced"

# ---- Model / LoRA ------------------------------------------------------------
MODEL_NAME_OR_PATH="liuhaotian/llava-v1.5-7b"
PROMPT_VERSION="v1"
VISION_TOWER="openai/clip-vit-large-patch14-336"
MM_PROJECTOR_TYPE="mlp2x_gelu"
MM_VISION_SELECT_LAYER=-2
MM_USE_IM_START_END=False
MM_USE_IM_PATCH_TOKEN=False
IMAGE_ASPECT_RATIO="pad"
GROUP_BY_MODALITY_LENGTH=True
BF16=False
FP16=True

LORA_ENABLE=True
LORA_R=32
LORA_ALPHA=64
MM_PROJECTOR_LR=5e-6

# ---- Optimizer / schedule (5k samples, 2 epochs) -----------------------------
NUM_TRAIN_EPOCHS=2
PER_DEVICE_TRAIN_BATCH_SIZE=4
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2

LEARNING_RATE=5e-5
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.05
LR_SCHEDULER_TYPE="cosine"

# ---- Save / log --------------------------------------------------------------
EVALUATION_STRATEGY="no"
SAVE_STRATEGY="steps"
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=16
LOGGING_STEPS=5

REPORT_TO="tensorboard"
LOGGING_DIR="${OUTPUT_DIR}/tb"
RUN_NAME="llava_v1_5_7b_lora_direct_5k_with_cot_balanced_2gpu"
LOGGING_FIRST_STEP=True
LOG_LEVEL=info
LOGGING_NAN_INF_FILTER=True
SEED=42
SAVE_SAFETENSORS=True

# ---- Runtime ----------------------------------------------------------------
TF32=True
MODEL_MAX_LENGTH=2048
GRADIENT_CHECKPOINTING=True
DATALOADER_NUM_WORKERS=4
LAZY_PREPROCESS=True

# ---- DeepSpeed ---------------------------------------------------------------
DEEPSPEED_CONFIG="${ROOT_DIR}/scripts/zero3.json"
if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "Error: DeepSpeed config not found: ${DEEPSPEED_CONFIG}" >&2
  exit 1
fi

# ---- Wrapper entry -----------------------------------------------------------
TRAIN_ENTRY="${ROOT_DIR}/scripts/_train_with_step_save.py"
if [[ ! -f "${TRAIN_ENTRY}" ]]; then
  echo "Error: training entry not found: ${TRAIN_ENTRY}" >&2
  exit 1
fi

# ---- Pre-create output dirs --------------------------------------------------
mkdir -p "${OUTPUT_DIR}" "${LOGGING_DIR}"

# ---- Safety: refuse to silently auto-resume ---------------------------------
# llava/train/train.py auto-resumes from any pre-existing checkpoint-* dirs in
# OUTPUT_DIR. For a clean ablation that must start from the BASE model, refuse
# to launch if such a dir exists; user can either rm it or set ALLOW_RESUME=1.
if [[ "${ALLOW_RESUME:-0}" != "1" ]]; then
  if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
    echo "[launcher] ERROR: ${OUTPUT_DIR} already contains checkpoint-* dirs." >&2
    echo "[launcher]        llava/train/train.py would AUTO-RESUME from them," >&2
    echo "[launcher]        which makes the run NOT a from-scratch SFT and" >&2
    echo "[launcher]        invalidates the direct_5k_balanced ablation." >&2
    echo "[launcher]        Either remove them, or rerun with ALLOW_RESUME=1." >&2
    exit 1
  fi
fi

# ---- Step 1: normalize image paths (abs path -> basename) -------------------
# The upstream sft_with_cot exporter writes absolute /mnt/.../openimages_cache
# paths into the ``image`` field; LLaVA's loader and preflight expect a
# basename relative to ``image_folder``. We rewrite to a sibling JSONL.
echo "[launcher] normalize image paths..."
NORMALIZE_LOG="${OUTPUT_DIR}/normalize.log"
NORMALIZED_DATA="${OUTPUT_DIR}/data.normalized.jsonl"
python3 "${ROOT_DIR}/scripts/normalize_sharegpt_image_paths.py" \
  --input  "${DATA_PATH}" \
  --output "${NORMALIZED_DATA}" \
  --image-folder "${IMAGE_FOLDER}" 2>&1 | tee "${NORMALIZE_LOG}"
NORMALIZED_DATA_RESOLVED="$(grep -E '^USE_DATA_PATH=' "${NORMALIZE_LOG}" | tail -n1 | sed 's/^USE_DATA_PATH=//')"
if [[ -z "${NORMALIZED_DATA_RESOLVED}" ]]; then
  echo "[launcher] normalize failed: no USE_DATA_PATH= line in ${NORMALIZE_LOG}" >&2
  exit 1
fi
echo "[launcher] normalized data: ${NORMALIZED_DATA_RESOLVED}"

# ---- Step 2: preflight (JSONL->JSON array + row/image validation) ------------
echo "[launcher] preflight..."
PREFLIGHT_LOG="${OUTPUT_DIR}/preflight.log"
python3 "${ROOT_DIR}/scripts/preflight_lora_dataset.py" \
  --data "${NORMALIZED_DATA_RESOLVED}" \
  --images "${IMAGE_FOLDER}" 2>&1 | tee "${PREFLIGHT_LOG}"
DATA_PATH_RESOLVED="$(grep -E '^USE_DATA_PATH=' "${PREFLIGHT_LOG}" | tail -n1 | sed 's/^USE_DATA_PATH=//')"
if [[ -z "${DATA_PATH_RESOLVED}" ]]; then
  echo "[launcher] preflight failed: no USE_DATA_PATH= line in ${PREFLIGHT_LOG}" >&2
  exit 1
fi
echo "[launcher] using data file: ${DATA_PATH_RESOLVED}"

# ---- Banner ------------------------------------------------------------------
echo "===== LLaVA LoRA SFT (direct_5k + CoT, BALANCED hyperparams, 2-GPU H20) ====="
echo "ROOT_DIR        : ${ROOT_DIR}"
echo "GPUs            : ${CUDA_VISIBLE_DEVICES}    (NUM_GPUS=${NUM_GPUS})"
echo "MASTER_PORT     : ${MASTER_PORT}"
echo "DATA_PATH (in)  : ${DATA_PATH}"
echo "DATA_PATH (use) : ${DATA_PATH_RESOLVED}"
echo "IMAGE_FOLDER    : ${IMAGE_FOLDER}    (shared with direct_5k baseline)"
echo "OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "RUN_NAME        : ${RUN_NAME}"
echo "LR / mm_proj_lr : ${LEARNING_RATE} / ${MM_PROJECTOR_LR}"
echo "LoRA r / alpha  : ${LORA_R} / ${LORA_ALPHA}"
echo "epochs / save   : ${NUM_TRAIN_EPOCHS} epoch / save_steps=${SAVE_STEPS} / limit=${SAVE_TOTAL_LIMIT}"
echo "BS / accum      : per_dev=${PER_DEVICE_TRAIN_BATCH_SIZE}  accum=${GRADIENT_ACCUMULATION_STEPS}  effective=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "precision       : bf16=${BF16} fp16=${FP16} tf32=${TF32}"
echo "ablation pair   : direct_5k_balanced (letter) vs direct_5k_with_cot_balanced (letter+explanation)"
echo "==========================================================================="

# ---- Pick deepspeed launcher -------------------------------------------------
PYTHON="${PYTHON:-python3}"
if command -v deepspeed >/dev/null 2>&1; then
  DEEPSPEED_CMD=(deepspeed)
else
  DEEPSPEED_CMD=("${PYTHON}" -m deepspeed)
fi
echo "[launcher] deepspeed cmd: ${DEEPSPEED_CMD[*]}"

# ---- Runtime log -------------------------------------------------------------
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUNTIME_LOG="${OUTPUT_DIR}/runtime_${TIMESTAMP}.log"
echo "[launcher] runtime log: ${RUNTIME_LOG}"

# ---- Train -------------------------------------------------------------------
"${DEEPSPEED_CMD[@]}" --master_port "${MASTER_PORT}" --num_gpus "${NUM_GPUS}" "${TRAIN_ENTRY}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --lora_enable "${LORA_ENABLE}" --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --mm_projector_lr "${MM_PROJECTOR_LR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --version "${PROMPT_VERSION}" \
  --data_path "${DATA_PATH_RESOLVED}" \
  --image_folder "${IMAGE_FOLDER}" \
  --vision_tower "${VISION_TOWER}" \
  --mm_projector_type "${MM_PROJECTOR_TYPE}" \
  --mm_vision_select_layer "${MM_VISION_SELECT_LAYER}" \
  --mm_use_im_start_end "${MM_USE_IM_START_END}" \
  --mm_use_im_patch_token "${MM_USE_IM_PATCH_TOKEN}" \
  --image_aspect_ratio "${IMAGE_ASPECT_RATIO}" \
  --group_by_modality_length "${GROUP_BY_MODALITY_LENGTH}" \
  --bf16 "${BF16}" \
  --fp16 "${FP16}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --evaluation_strategy "${EVALUATION_STRATEGY}" \
  --save_strategy "${SAVE_STRATEGY}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --logging_steps "${LOGGING_STEPS}" \
  --report_to "${REPORT_TO}" \
  --logging_dir "${LOGGING_DIR}" \
  --logging_first_step "${LOGGING_FIRST_STEP}" \
  --log_level "${LOG_LEVEL}" \
  --logging_nan_inf_filter "${LOGGING_NAN_INF_FILTER}" \
  --seed "${SEED}" \
  --run_name "${RUN_NAME}" \
  --save_safetensors "${SAVE_SAFETENSORS}" \
  --tf32 "${TF32}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --lazy_preprocess "${LAZY_PREPROCESS}" 2>&1 | tee "${RUNTIME_LOG}"

echo "[launcher] training finished. Outputs:"
echo "  checkpoints           : ${OUTPUT_DIR}/checkpoint-*"
echo "  final adapter (top)   : ${OUTPUT_DIR}/{adapter_*,non_lora_trainables.bin,config.json}"
echo "  best by train loss    : ${OUTPUT_DIR}/best_by_train_loss/  (heuristic; verify with MMBench)"
echo "  ranking summary       : ${OUTPUT_DIR}/best_step_candidates.json"
echo "  per-step metrics      : ${OUTPUT_DIR}/training_metrics.jsonl"
echo "  runtime log           : ${RUNTIME_LOG}"
