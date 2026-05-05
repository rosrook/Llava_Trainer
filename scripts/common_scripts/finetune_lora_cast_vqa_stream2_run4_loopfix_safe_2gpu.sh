#!/usr/bin/env bash
# ==============================================================================
# LLaVA-1.5-7B LoRA SFT on the CAST_VQA stream2 sft_sharegpt dump
# (cast_vqa/outputs/ut_ol_10k_v1/stream2_run4_loopfix/sft_merged), ~7k MCQ samples.
#
# Hardware: 2x NVIDIA H20-3e (~143 GB each), GPUs 0,1 by default.
#
# Pipeline:
#   1. prepare_cast_vqa_sft.py
#        sft_sharegpt.jsonl  ->  CONVERTED_DIR/{vqa.jsonl, images/}
#        (rewrites absolute image paths -> relative basenames + symlinks
#         the originals into images/ so preflight_lora_dataset.py is happy)
#   2. preflight_lora_dataset.py
#        sanity-check vqa.jsonl + images, emit USE_DATA_PATH=
#   3. deepspeed _train_with_step_save.py
#        SAFE recipe (LR=5e-5, LoRA r=32/alpha=64, 1 epoch, save_steps=50)
#        -- see refined_vqa_1_9_5k_safe_2gpu.sh for the rationale; the same
#        over-fitting problem (1 epoch is enough on a few-thousand-row MCQ
#        dataset) applies here.
#
# Override paths/run name without editing the file:
#   SOURCE_JSONL=/abs/path/to/sft_sharegpt.jsonl \
#   CONVERTED_DIR=/abs/path/to/cast_vqa_xxx \
#   OUTPUT_DIR=/abs/path/to/models/cast_vqa_xxx \
#   bash scripts/common_scripts/finetune_lora_cast_vqa_stream2_run4_loopfix_safe_2gpu.sh
#
# Effective batch: 2 GPU x bs=4 x grad_accum=2 = 16
# At ~7000 rows / 16 = ~438 steps for 1 epoch
# save_steps=50 -> checkpoints at 50/100/150/.../400/438  (~9 candidates)
# Use scripts/plot_training_curves.py + best_step_candidates.json afterwards,
# then verify with MMBench (eval_mmbench_v15_7b_step_checkpoint.sh).
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# ---- GPUs / launcher ---------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=2
MASTER_PORT="${MASTER_PORT:-29535}"

# ---- Source data / converted data / output paths ----------------------------
# Default to where ut_ol_10k_v1 lives on the cluster; override with env vars.
SOURCE_JSONL="${SOURCE_JSONL:-/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/cast_vqa/outputs/ut_ol_10k_v1/stream2_run4_loopfix/sft_merged/sft_sharegpt.jsonl}"
CONVERTED_DIR="${CONVERTED_DIR:-/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/bysj_third_run/ok_for_training/cast_vqa_ut_ol_10k_v1_stream2_run4_loopfix}"
DATA_PATH="${DATA_PATH:-${CONVERTED_DIR}/vqa.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${CONVERTED_DIR}/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/bysj/Llava_v1_5_7B/cast_vqa_ut_ol_10k_v1_stream2_run4_loopfix_safe}"

# ---- Conversion settings -----------------------------------------------------
PREPARE_IMAGE_MODE="${PREPARE_IMAGE_MODE:-symlink}"   # symlink | copy
PREPARE_MAX_SAMPLES="${PREPARE_MAX_SAMPLES:-0}"       # 0 = no cap, otherwise random subsample (uses SAMPLE_SEED)
# If you pass PREPARE_MAX_SAMPLES > 0 and want HEAD order instead of random,
# set PREPARE_SHUFFLE=0 explicitly. Default shuffles because CAST_VQA writes
# curriculum order (easy first) and head-N would bias the training set.
PREPARE_SHUFFLE="${PREPARE_SHUFFLE:-1}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

# ---- Model / LoRA (SAFE recipe) ---------------------------------------------
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
# Override these via env when chasing under/over-fit. Sane sweep around the
# SAFE recipe (letter_text answers, ~5k MCQ): r in {16, 24, 32}, alpha = 2*r,
# LEARNING_RATE in {2e-5, 3e-5, 5e-5}, MM_PROJECTOR_LR ~= LR / 10.
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-5e-6}"

# ---- Optimizer / schedule (~7k samples, 1 epoch by default) -----------------
# Override via env, e.g. `NUM_TRAIN_EPOCHS=2 bash ...` to mirror the
# explain_balanced recipe. Default 1 because the CAST_VQA dump uses
# letter_text answers (short outputs) -- explain_balanced uses 2 epochs only
# because its answer_mode=explanation needs the extra pass to escape the
# format-learning plateau.
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE=4
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2

LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER_TYPE="cosine"

# ---- Save / log --------------------------------------------------------------
EVALUATION_STRATEGY="no"
SAVE_STRATEGY="steps"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-15}"
LOGGING_STEPS=5

REPORT_TO="tensorboard"
LOGGING_DIR="${OUTPUT_DIR}/tb"
RUN_NAME="${RUN_NAME:-llava_v1_5_7b_lora_cast_vqa_ut_ol_10k_v1_stream2_run4_loopfix_safe_2gpu}"
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

PREPARE_ENTRY="${ROOT_DIR}/scripts/prepare_cast_vqa_sft.py"
if [[ ! -f "${PREPARE_ENTRY}" ]]; then
  echo "Error: prepare entry not found: ${PREPARE_ENTRY}" >&2
  exit 1
fi

# ---- 0) Convert sft_sharegpt.jsonl -> vqa.jsonl + images/ -------------------
mkdir -p "${CONVERTED_DIR}" "${OUTPUT_DIR}" "${LOGGING_DIR}"
PREPARE_LOG="${OUTPUT_DIR}/prepare_cast_vqa_sft.log"
if [[ -f "${DATA_PATH}" && -d "${IMAGE_FOLDER}" ]]; then
  echo "[launcher] reuse converted data: ${DATA_PATH}"
else
  echo "[launcher] preparing CAST_VQA sft -> LLaVA layout (image_mode=${PREPARE_IMAGE_MODE})..."
  PREP_ARGS=(
    --input "${SOURCE_JSONL}"
    --output-dir "${CONVERTED_DIR}"
    --image-mode "${PREPARE_IMAGE_MODE}"
    --seed "${SAMPLE_SEED}"
    --overwrite
  )
  if [[ "${PREPARE_MAX_SAMPLES}" -gt 0 ]]; then
    PREP_ARGS+=(--max-samples "${PREPARE_MAX_SAMPLES}")
    if [[ "${PREPARE_SHUFFLE}" != "0" ]]; then
      PREP_ARGS+=(--shuffle)
    fi
  fi
  python3 "${PREPARE_ENTRY}" "${PREP_ARGS[@]}" 2>&1 | tee "${PREPARE_LOG}"
fi

# ---- 1) Preflight ------------------------------------------------------------
echo "[launcher] preflight..."
PREFLIGHT_LOG="${OUTPUT_DIR}/preflight.log"
python3 "${ROOT_DIR}/scripts/preflight_lora_dataset.py" \
  --data "${DATA_PATH}" \
  --images "${IMAGE_FOLDER}" 2>&1 | tee "${PREFLIGHT_LOG}"
DATA_PATH_RESOLVED="$(grep -E '^USE_DATA_PATH=' "${PREFLIGHT_LOG}" | tail -n1 | sed 's/^USE_DATA_PATH=//')"
if [[ -z "${DATA_PATH_RESOLVED}" ]]; then
  echo "[launcher] preflight failed: no USE_DATA_PATH= line in ${PREFLIGHT_LOG}" >&2
  exit 1
fi
echo "[launcher] using data file: ${DATA_PATH_RESOLVED}"

# ---- Banner ------------------------------------------------------------------
echo "===== LLaVA LoRA SFT (CAST_VQA ut_ol_10k_v1 stream2_run4_loopfix, SAFE) ====="
echo "ROOT_DIR        : ${ROOT_DIR}"
echo "GPUs            : ${CUDA_VISIBLE_DEVICES}    (NUM_GPUS=${NUM_GPUS})"
echo "MASTER_PORT     : ${MASTER_PORT}"
echo "SOURCE_JSONL    : ${SOURCE_JSONL}"
echo "CONVERTED_DIR   : ${CONVERTED_DIR}"
echo "DATA_PATH (use) : ${DATA_PATH_RESOLVED}"
echo "IMAGE_FOLDER    : ${IMAGE_FOLDER}"
echo "OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "RUN_NAME        : ${RUN_NAME}"
echo "PREPARE         : max_samples=${PREPARE_MAX_SAMPLES}  shuffle=${PREPARE_SHUFFLE}  seed=${SAMPLE_SEED}  image_mode=${PREPARE_IMAGE_MODE}"
echo "LR / mm_proj_lr : ${LEARNING_RATE} / ${MM_PROJECTOR_LR}  (warmup=${WARMUP_RATIO}, wd=${WEIGHT_DECAY})"
echo "LoRA r / alpha  : ${LORA_R} / ${LORA_ALPHA}"
echo "epochs / save   : ${NUM_TRAIN_EPOCHS} epoch(s) / save_steps=${SAVE_STEPS} / limit=${SAVE_TOTAL_LIMIT}"
echo "BS / accum      : per_dev=${PER_DEVICE_TRAIN_BATCH_SIZE}  accum=${GRADIENT_ACCUMULATION_STEPS}  effective=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "precision       : bf16=${BF16} fp16=${FP16} tf32=${TF32}"
echo "============================================================================="

# ---- 2) Train ----------------------------------------------------------------
deepspeed --master_port "${MASTER_PORT}" "${TRAIN_ENTRY}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --lora_enable "${LORA_ENABLE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --mm_projector_lr "${MM_PROJECTOR_LR}" \
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
  --lazy_preprocess "${LAZY_PREPROCESS}"

echo "[launcher] training finished."
echo "[launcher] output: ${OUTPUT_DIR}"
