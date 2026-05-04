#!/usr/bin/env bash
# ==============================================================================
# LLaVA-1.5-7B LoRA SFT on refined_vqa_dataset_1_9 (base64 images), 5000 samples.
#
# This is the same training recipe as direct_5k:
#   - 2x H20 GPUs (0,1 by default)
#   - 3 epochs
#   - save_steps=100
#   - fp16 + ZeRO-3
#   - per-checkpoint exportability via scripts/_train_with_step_save.py +
#     LLaVATrainer's non_lora_trainables.bin save
#
# It adds one preprocessing step before training:
#   JSON array with image_base64 + MCQ QA
#     -> LLaVA SFT vqa.json/vqa.jsonl + decoded images/*.jpg
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# ---- GPUs / launcher ---------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=2
MASTER_PORT="${MASTER_PORT:-29531}"

# ---- Source data / converted data / output paths ----------------------------
SOURCE_JSON="${SOURCE_JSON:-/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/second_refined_and_expanded_vqa_data/refined_vqa_dataset_1_9.json}"
CONVERTED_DIR="${CONVERTED_DIR:-/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/bysj_second_run/ok_for_training/refined_vqa_1_9_direct_5k}"
DATA_PATH="${DATA_PATH:-${CONVERTED_DIR}/vqa.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${CONVERTED_DIR}/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/bysj/Llava_v1_5_7B/refined_vqa_1_9_direct_5k}"

# ---- Conversion settings -----------------------------------------------------
SAMPLE_SIZE="${SAMPLE_SIZE:-5000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
ANSWER_MODE="${ANSWER_MODE:-letter}"   # letter | letter_text | explanation

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
LORA_R=128
LORA_ALPHA=256
MM_PROJECTOR_LR=2e-5

# ---- Optimizer / schedule (5k samples, 3 epochs) -----------------------------
NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=4
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2

LEARNING_RATE=2e-4
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"

# ---- Save / log --------------------------------------------------------------
EVALUATION_STRATEGY="no"
SAVE_STRATEGY="steps"
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=20
LOGGING_STEPS=5

REPORT_TO="tensorboard"
LOGGING_DIR="${OUTPUT_DIR}/tb"
RUN_NAME="${RUN_NAME:-llava_v1_5_7b_lora_refined_vqa_1_9_direct_5k_2gpu}"
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

# ---- Convert source JSON -> LLaVA SFT ---------------------------------------
mkdir -p "${CONVERTED_DIR}" "${OUTPUT_DIR}" "${LOGGING_DIR}"
CONVERT_LOG="${OUTPUT_DIR}/convert_base64_vqa.log"
if [[ -f "${DATA_PATH}" && -d "${IMAGE_FOLDER}" ]]; then
  echo "[launcher] reuse converted data: ${DATA_PATH}"
else
  echo "[launcher] converting base64 VQA -> LLaVA SFT..."
  python3 "${ROOT_DIR}/scripts/convert_base64_vqa_to_llava.py" \
    --input "${SOURCE_JSON}" \
    --output-dir "${CONVERTED_DIR}" \
    --sample-size "${SAMPLE_SIZE}" \
    --seed "${SAMPLE_SEED}" \
    --answer-mode "${ANSWER_MODE}" \
    --image-format jpg \
    --overwrite 2>&1 | tee "${CONVERT_LOG}"
fi

# ---- Preflight: detect/convert JSONL, validate rows + image files ------------
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
echo "============ LLaVA LoRA SFT (refined_vqa_1_9_direct_5k, 2-GPU H20) ============"
echo "ROOT_DIR        : ${ROOT_DIR}"
echo "GPUs            : ${CUDA_VISIBLE_DEVICES}    (NUM_GPUS=${NUM_GPUS})"
echo "MASTER_PORT     : ${MASTER_PORT}"
echo "SOURCE_JSON     : ${SOURCE_JSON}"
echo "CONVERTED_DIR   : ${CONVERTED_DIR}"
echo "DATA_PATH (use) : ${DATA_PATH_RESOLVED}"
echo "IMAGE_FOLDER    : ${IMAGE_FOLDER}"
echo "OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "RUN_NAME        : ${RUN_NAME}"
echo "BS / accum      : per_dev=${PER_DEVICE_TRAIN_BATCH_SIZE}  accum=${GRADIENT_ACCUMULATION_STEPS}  effective=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "epochs/save     : epochs=${NUM_TRAIN_EPOCHS}  save_steps=${SAVE_STEPS}"
echo "precision       : bf16=${BF16} fp16=${FP16} tf32=${TF32}"
echo "============================================================================="

# ---- Launch ------------------------------------------------------------------
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
