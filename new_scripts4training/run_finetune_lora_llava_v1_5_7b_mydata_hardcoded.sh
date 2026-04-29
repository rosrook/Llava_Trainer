#!/usr/bin/env bash
set -euo pipefail

# =========================
# Hardcoded training script
# =========================
# This script is fully self-contained:
# - no parameter is read from external environment variables
# - all paths/hyperparameters are hardcoded below
# - keeps step checkpoints for later step->HF export

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# ---------- Fixed runtime ----------
CUDA_VISIBLE_DEVICES_FIXED="0,1,2,3,4,5,6,7"
NUM_GPUS=8
PYTHON_BIN="python3"
DEEPSPEED_CONFIG="./scripts/zero3_logging.json"
TRAIN_ENTRY="llava/train/train_mem.py"

# ---------- Fixed data ----------
DATA_PATH="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/first_data4llava7B/vqa_data.json"
IMAGE_FOLDER="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/first_data4llava7B/images"

# ---------- Fixed model ----------
MODEL_NAME_OR_PATH="liuhaotian/llava-v1.5-7b"
PROMPT_VERSION="v1"
VISION_TOWER="openai/clip-vit-large-patch14-336"
MM_PROJECTOR_TYPE="mlp2x_gelu"
MM_VISION_SELECT_LAYER=-2
MM_USE_IM_START_END=False
MM_USE_IM_PATCH_TOKEN=False
IMAGE_ASPECT_RATIO="pad"
GROUP_BY_MODALITY_LENGTH=True
BF16=True

# ---------- Fixed LoRA ----------
LORA_ENABLE=True
LORA_R=128
LORA_ALPHA=256
MM_PROJECTOR_LR=2e-5

# ---------- Fixed optimization ----------
NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"
TF32=True
MODEL_MAX_LENGTH=2048
GRADIENT_CHECKPOINTING=True
DATALOADER_NUM_WORKERS=4
LAZY_PREPROCESS=True
SEED=42

# ---------- Fixed save/log ----------
SAVE_STRATEGY="steps"
SAVE_STEPS=50
EVALUATION_STRATEGY="no"
LOGGING_STEPS=1
LOG_LEVEL="info"
LOGGING_NAN_INF_FILTER=True
LOGGING_FIRST_STEP=True
REPORT_TO="tensorboard"
SAVE_SAFETENSORS=True

# Keep all checkpoint-* to support step-level export (e.g., checkpoint-600).
# Do NOT set --save_total_limit so Trainer will not prune historical checkpoints.
SAVE_ONLY_MODEL=False

timestamp="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_DIR="./checkpoints/llava-v1.5-7b-mydata-lora-hardcoded-${timestamp}"
LOGGING_DIR="${OUTPUT_DIR}/tb"
DIAG_DIR="${OUTPUT_DIR}/diagnostics"
RUNTIME_LOG="${DIAG_DIR}/runtime_${timestamp}.log"
LAUNCH_LOG="${DIAG_DIR}/launcher_${timestamp}.log"

mkdir -p "${OUTPUT_DIR}" "${LOGGING_DIR}" "${DIAG_DIR}"

if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "Error: deepspeed config not found: ${DEEPSPEED_CONFIG}"
  exit 1
fi
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Error: data file not found: ${DATA_PATH}"
  exit 1
fi
if [[ ! -d "${IMAGE_FOLDER}" ]]; then
  echo "Error: image folder not found: ${IMAGE_FOLDER}"
  exit 1
fi

echo "ROOT_DIR=${ROOT_DIR}" | tee "${LAUNCH_LOG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "${LAUNCH_LOG}"
echo "RUNTIME_LOG=${RUNTIME_LOG}" | tee -a "${LAUNCH_LOG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_FIXED}" | tee -a "${LAUNCH_LOG}"
echo "NUM_GPUS=${NUM_GPUS}" | tee -a "${LAUNCH_LOG}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_FIXED}"

set +e
deepspeed --num_gpus "${NUM_GPUS}" "${TRAIN_ENTRY}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --lora_enable "${LORA_ENABLE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --mm_projector_lr "${MM_PROJECTOR_LR}" \
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
  --save_only_model "${SAVE_ONLY_MODEL}" \
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
  --save_safetensors "${SAVE_SAFETENSORS}" \
  --tf32 "${TF32}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --lazy_preprocess "${LAZY_PREPROCESS}" \
  2>&1 | tee "${RUNTIME_LOG}"
exit_code=${PIPESTATUS[0]}
set -e

if [[ "${exit_code}" -ne 0 ]]; then
  echo "Training failed, see: ${RUNTIME_LOG}"
  exit "${exit_code}"
fi

MANIFEST_PATH="${OUTPUT_DIR}/checkpoint_manifest.txt"
{
  echo "output_dir=${OUTPUT_DIR}"
  echo "generated_at=${timestamp}"
  ls -1d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V
} > "${MANIFEST_PATH}"

echo "Training finished."
echo "Checkpoint manifest: ${MANIFEST_PATH}"
echo "Example export (step600 -> standalone HF):"
echo "python scripts/export_lora_checkpoint_to_hf.py --base_model ${MODEL_NAME_OR_PATH} --checkpoint_dir ${OUTPUT_DIR}/checkpoint-600 --output_dir ${OUTPUT_DIR}/hf_step600 --bf16"

