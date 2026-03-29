#!/usr/bin/env bash
set -euo pipefail

# LoRA finetune entrypoint for: liuhaotian/llava-v1.5-7b
# Load hyperparameters from a config file so it can be reused.
#
# Override config:
#   CONFIG_FILE=/path/to/your_config.sh bash scripts/finetune_lora_llava_v1_5_7b_mydata.sh
#
# Default entry is train_mem.py (FlashAttention2). Fallback without flash-attn:
#   TRAIN_ENTRY=llava/train/train.py bash scripts/finetune_lora_llava_v1_5_7b_mydata.sh

CONFIG_FILE="${CONFIG_FILE:-./scripts/configs/finetune_lora_small_vqa_2399.sh}"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Error: CONFIG_FILE not found: ${CONFIG_FILE}"
  exit 1
fi
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

NUM_GPUS="${NUM_GPUS:-8}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./scripts/zero3.json}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-liuhaotian/llava-v1.5-7b}"
PROMPT_VERSION="${PROMPT_VERSION:-v1}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
MM_PROJECTOR_TYPE="${MM_PROJECTOR_TYPE:-mlp2x_gelu}"
MM_VISION_SELECT_LAYER="${MM_VISION_SELECT_LAYER:--2}"
MM_USE_IM_START_END="${MM_USE_IM_START_END:-False}"
MM_USE_IM_PATCH_TOKEN="${MM_USE_IM_PATCH_TOKEN:-False}"
IMAGE_ASPECT_RATIO="${IMAGE_ASPECT_RATIO:-pad}"
GROUP_BY_MODALITY_LENGTH="${GROUP_BY_MODALITY_LENGTH:-True}"
BF16="${BF16:-True}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/llava-v1.5-7b-mydata-lora}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
EVALUATION_STRATEGY="${EVALUATION_STRATEGY:-no}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
TF32="${TF32:-True}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LAZY_PREPROCESS="${LAZY_PREPROCESS:-True}"
LORA_ENABLE="${LORA_ENABLE:-True}"
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-256}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-2e-5}"

# Observability (HuggingFace Trainer + optional W&B / TensorBoard)
# REPORT_TO: space-separated list, e.g. "tensorboard" or "tensorboard wandb" or "none"
REPORT_TO="${REPORT_TO:-tensorboard}"
LOGGING_DIR="${LOGGING_DIR:-${OUTPUT_DIR}/tb}"
LOGGING_FIRST_STEP="${LOGGING_FIRST_STEP:-True}"
LOG_LEVEL="${LOG_LEVEL:-info}"
LOGGING_NAN_INF_FILTER="${LOGGING_NAN_INF_FILTER:-True}"
DISABLE_TQDM="${DISABLE_TQDM:-False}"
SEED="${SEED:-42}"
SAVE_SAFETENSORS="${SAVE_SAFETENSORS:-True}"
# RUN_NAME: W&B / TB run name; leave empty for Trainer default
RUN_NAME="${RUN_NAME:-}"

TRAIN_ENTRY="${TRAIN_ENTRY:-llava/train/train_mem.py}"

if [[ -z "${DATA_PATH:-}" ]]; then
  echo "Error: DATA_PATH is required in config file: ${CONFIG_FILE}"
  exit 1
fi
if [[ -z "${IMAGE_FOLDER:-}" ]]; then
  echo "Error: IMAGE_FOLDER is required in config file: ${CONFIG_FILE}"
  exit 1
fi

echo "Using config: ${CONFIG_FILE}"
echo "DATA_PATH: ${DATA_PATH}"
echo "IMAGE_FOLDER: ${IMAGE_FOLDER}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "LOGGING_DIR: ${LOGGING_DIR}"
echo "REPORT_TO: ${REPORT_TO}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "TRAIN_ENTRY: ${TRAIN_ENTRY}"

read -r -a REPORT_ARR <<< "${REPORT_TO}"
if [[ ${#REPORT_ARR[@]} -eq 0 ]]; then
  REPORT_ARR=(tensorboard)
fi
RUN_NAME_ARGS=()
if [[ -n "${RUN_NAME}" ]]; then
  RUN_NAME_ARGS=(--run_name "${RUN_NAME}")
fi

mkdir -p "${LOGGING_DIR}"

DISABLE_TQDM_ARGS=()
if [[ "${DISABLE_TQDM}" == "True" ]]; then
  DISABLE_TQDM_ARGS=(--disable_tqdm True)
fi

PYTHON="${PYTHON:-python3}"
if [[ -n "${DEEPSPEED_LAUNCHER:-}" ]]; then
  read -r -a DEEPSPEED_CMD <<< "${DEEPSPEED_LAUNCHER}"
else
  if command -v deepspeed >/dev/null 2>&1; then
    DEEPSPEED_CMD=(deepspeed)
  else
    DEEPSPEED_CMD=("${PYTHON}" -m deepspeed)
  fi
fi
echo "DeepSpeed launcher: ${DEEPSPEED_CMD[*]}"

"${DEEPSPEED_CMD[@]}" --num_gpus "${NUM_GPUS}" "${TRAIN_ENTRY}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --lora_enable "${LORA_ENABLE}" --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --mm_projector_lr "${MM_PROJECTOR_LR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --version "${PROMPT_VERSION}" \
  --data_path "${DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --vision_tower "${VISION_TOWER}" \
  --mm_projector_type "${MM_PROJECTOR_TYPE}" \
  --mm_vision_select_layer "${MM_VISION_SELECT_LAYER}" \
  --mm_use_im_start_end "${MM_USE_IM_START_END}" \
  --mm_use_im_patch_token "${MM_USE_IM_PATCH_TOKEN}" \
  --image_aspect_ratio "${IMAGE_ASPECT_RATIO}" \
  --group_by_modality_length "${GROUP_BY_MODALITY_LENGTH}" \
  --bf16 "${BF16}" \
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
  --report_to "${REPORT_ARR[@]}" \
  --logging_dir "${LOGGING_DIR}" \
  --logging_first_step "${LOGGING_FIRST_STEP}" \
  --log_level "${LOG_LEVEL}" \
  --logging_nan_inf_filter "${LOGGING_NAN_INF_FILTER}" \
  "${DISABLE_TQDM_ARGS[@]}" \
  --seed "${SEED}" \
  "${RUN_NAME_ARGS[@]}" \
  --save_safetensors "${SAVE_SAFETENSORS}" \
  --tf32 "${TF32}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --lazy_preprocess "${LAZY_PREPROCESS}"
