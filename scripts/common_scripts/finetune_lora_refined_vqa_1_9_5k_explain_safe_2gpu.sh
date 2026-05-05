#!/usr/bin/env bash
# ==============================================================================
# LLaVA-1.5-7B LoRA SFT on refined_vqa_dataset_1_9, "explain_safe" recipe.
#
# Sibling of:
#   finetune_lora_refined_vqa_1_9_5k_2gpu.sh        (letter,      LR 2e-4, r=128)
#   finetune_lora_refined_vqa_1_9_5k_safe_2gpu.sh   (letter_text, LR 5e-5, r=32)
#
# Why this third script:
#   In letter / letter_text modes most of the answer tokens are deterministic
#   (formatting + copy from options), so loss can collapse below 0.2 within
#   ~50 steps without the model actually learning visual reasoning. With
#   `answer-mode=explanation` the supervision target becomes:
#
#       B
#       {explanation, ~30-60 tokens describing why}
#
#   so the gradient pressure shifts onto real semantic tokens. Loss starts
#   higher, drops slower, and is a much more honest signal of fitting.
#
#   On evaluation: vlmevalkit's MCQ prefetch only inspects the first
#   character(s) of the model output, so training with explanation does NOT
#   break MMBench scoring; the first token is still the correct option letter.
#
# Differences vs the "safe" launcher:
#   ANSWER_MODE         letter_text    -> explanation
#   LEARNING_RATE       5e-5           -> 3e-5         (further -40%)
#   MM_PROJECTOR_LR     5e-6           -> 3e-6         (proportional)
#   LORA_R              32             -> 16           (smaller capacity)
#   LORA_ALPHA          64             -> 32           (alpha = 2 * r)
#   CONVERTED_DIR       ..._letter_text_safe
#                                      -> ..._explain_safe   (different target text)
#   OUTPUT_DIR          ..._letter_text_safe
#                                      -> ..._explain_safe
#   RUN_NAME            ..._letter_text_safe_2gpu
#                                      -> ..._explain_safe_2gpu
#   MASTER_PORT         29533          -> 29535       (avoid clash if both run)
#
# Same as the safe launcher:
#   - 2x H20 GPUs (CUDA_VISIBLE_DEVICES=0,1 by default)
#   - effective batch = 2 * 4 * 2 = 16
#   - 1 epoch, ~313 optimisation steps
#   - save_steps=50, save_total_limit=12  (checkpoint-50/100/.../300)
#   - fp16 + ZeRO-3
#   - per-checkpoint exportability via _train_with_step_save.py
#
# Note on inference:
#   When evaluating these checkpoints with eval_my_checkpoint_mmbench.sh you
#   may want to bump VLMEVAL_LLAVA_MAX_NEW_TOKENS (e.g. 128 or 256) so the
#   model can finish generating its explanation cleanly, although the prefetch
#   only needs the first token.
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# ---- GPUs / launcher ---------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=2
MASTER_PORT="${MASTER_PORT:-29535}"

# ---- Source data / converted data / output paths ----------------------------
SOURCE_JSON="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/second_refined_and_expanded_vqa_data/refined_vqa_dataset_1_9.json"
CONVERTED_DIR="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/bysj_second_run/ok_for_training/refined_vqa_1_9_direct_5k_explain_safe"
DATA_PATH="${CONVERTED_DIR}/vqa.jsonl"
IMAGE_FOLDER="${CONVERTED_DIR}/images"
OUTPUT_DIR="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/bysj/Llava_v1_5_7B/refined_vqa_1_9_direct_5k_explain_safe"

# ---- Conversion settings -----------------------------------------------------
SAMPLE_SIZE=5000
SAMPLE_SEED=42
ANSWER_MODE="explanation"   # output: "<letter>\n<explanation>"

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
LORA_R=16
LORA_ALPHA=32
MM_PROJECTOR_LR=3e-6

# ---- Optimizer / schedule (5k samples, 1 epoch) -----------------------------
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=4
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2

LEARNING_RATE=3e-5
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
RUN_NAME="llava_v1_5_7b_lora_refined_vqa_1_9_direct_5k_explain_safe_2gpu"
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
echo "===== LLaVA LoRA SFT (refined_vqa_1_9 5k, EXPLANATION + safe recipe) ====="
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
echo "=========================================================================="

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
