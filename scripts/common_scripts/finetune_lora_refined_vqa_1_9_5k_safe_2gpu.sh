#!/usr/bin/env bash
# ==============================================================================
# LLaVA-1.5-7B LoRA SFT on refined_vqa_dataset_1_9, "safe" recipe for 5k MCQ.
#
# This is a sibling of finetune_lora_refined_vqa_1_9_5k_2gpu.sh. The original
# launcher uses LLaVA's official stage-2 hyperparameters (LR=2e-4, LoRA r=128,
# 3 epochs), which are tuned for hundreds of thousands of SFT samples.
# Empirically those values cause severe over-fitting on this 5k single-letter
# MCQ data: train loss collapses to ~0.04 within the first epoch, then spikes,
# and MMBench accuracy drops well below the base model.
#
# This file does NOT modify the original launcher, the old training output, or
# the converted dataset directory used by it. Everything written by this
# script lives under a new "_letter_text_safe" path so the old experiments
# stay reproducible.
#
# Differences vs the original 5k launcher:
#   ANSWER_MODE         letter         -> letter_text  (e.g. "B. center-right")
#   LEARNING_RATE       2e-4           -> 5e-5         (4x lower)
#   MM_PROJECTOR_LR     2e-5           -> 5e-6         (proportional)
#   LORA_R              128            -> 32           (less memorisation)
#   LORA_ALPHA          256            -> 64           (alpha = 2 * r)
#   NUM_TRAIN_EPOCHS    3              -> 1            (single pass)
#   WARMUP_RATIO        0.03           -> 0.05
#   SAVE_STEPS          100            -> 50           (capture early region)
#   SAVE_TOTAL_LIMIT    20             -> 12
#   CONVERTED_DIR       refined_vqa_1_9_direct_5k
#                                      -> refined_vqa_1_9_direct_5k_letter_text_safe
#   OUTPUT_DIR          refined_vqa_1_9_direct_5k
#                                      -> refined_vqa_1_9_direct_5k_letter_text_safe
#   RUN_NAME            ..._2gpu       -> ..._letter_text_safe_2gpu
#   MASTER_PORT         29531          -> 29533       (avoid clash if both run)
#
# Effective batch: 2 GPU x bs=4 x grad_accum=2 = 16
# Total optimisation steps: ~5000 / 16 = ~313
# With save_steps=50 you get checkpoints at 50/100/150/200/250/300, which
# spans both fast-converging and "well-trained" regions. Use TensorBoard or
# scripts/plot_training_curves.py to confirm afterwards.
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# ---- GPUs / launcher ---------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=2
MASTER_PORT="${MASTER_PORT:-29533}"

# ---- Source data / converted data / output paths ----------------------------
SOURCE_JSON="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/second_refined_and_expanded_vqa_data/refined_vqa_dataset_1_9.json"
CONVERTED_DIR="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/bysj_second_run/ok_for_training/refined_vqa_1_9_direct_5k_letter_text_safe"
DATA_PATH="${CONVERTED_DIR}/vqa.jsonl"
IMAGE_FOLDER="${CONVERTED_DIR}/images"
OUTPUT_DIR="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/bysj/Llava_v1_5_7B/refined_vqa_1_9_direct_5k_letter_text_safe"

# ---- Conversion settings -----------------------------------------------------
SAMPLE_SIZE=5000
SAMPLE_SEED=42
ANSWER_MODE="letter_text"   # answer like "B. center-right", richer signal

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

# ---- Optimizer / schedule (5k samples, 1 epoch) -----------------------------
NUM_TRAIN_EPOCHS=1
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
SAVE_TOTAL_LIMIT=12
LOGGING_STEPS=5

REPORT_TO="tensorboard"
LOGGING_DIR="${OUTPUT_DIR}/tb"
RUN_NAME="llava_v1_5_7b_lora_refined_vqa_1_9_direct_5k_letter_text_safe_2gpu"
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

# ---- Convert source JSON -> LLaVA SFT (writes to a NEW directory) -----------
mkdir -p "${CONVERTED_DIR}" "${OUTPUT_DIR}" "${LOGGING_DIR}"
CONVERT_LOG="${OUTPUT_DIR}/convert_base64_vqa.log"
if [[ -f "${DATA_PATH}" && -d "${IMAGE_FOLDER}" ]]; then
  echo "[launcher] reuse converted data: ${DATA_PATH}"
else
  echo "[launcher] converting base64 VQA -> LLaVA SFT (answer_mode=${ANSWER_MODE})..."
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
echo "===== LLaVA LoRA SFT (refined_vqa_1_9 5k, letter_text, SAFE recipe) ====="
echo "ROOT_DIR        : ${ROOT_DIR}"
echo "GPUs            : ${CUDA_VISIBLE_DEVICES}    (NUM_GPUS=${NUM_GPUS})"
echo "MASTER_PORT     : ${MASTER_PORT}"
echo "SOURCE_JSON     : ${SOURCE_JSON}"
echo "CONVERTED_DIR   : ${CONVERTED_DIR}"
echo "DATA_PATH (use) : ${DATA_PATH_RESOLVED}"
echo "IMAGE_FOLDER    : ${IMAGE_FOLDER}"
echo "OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "RUN_NAME        : ${RUN_NAME}"
echo "ANSWER_MODE     : ${ANSWER_MODE}"
echo "LR / mm_proj_lr : ${LEARNING_RATE} / ${MM_PROJECTOR_LR}"
echo "LoRA r / alpha  : ${LORA_R} / ${LORA_ALPHA}"
echo "epochs / save   : ${NUM_TRAIN_EPOCHS} epoch / save_steps=${SAVE_STEPS} / limit=${SAVE_TOTAL_LIMIT}"
echo "BS / accum      : per_dev=${PER_DEVICE_TRAIN_BATCH_SIZE}  accum=${GRADIENT_ACCUMULATION_STEPS}  effective=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "precision       : bf16=${BF16} fp16=${FP16} tf32=${TF32}"
echo "========================================================================="

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
