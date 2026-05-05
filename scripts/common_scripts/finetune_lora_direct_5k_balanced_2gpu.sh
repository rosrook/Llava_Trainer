#!/usr/bin/env bash
# ==============================================================================
# LLaVA-1.5-7B LoRA SFT on direct_5k (5000 samples), "balanced" recipe.
#
# Sibling of:
#   finetune_lora_direct_5k_2gpu.sh
#       (LR 2e-4,  r=128/alpha=256, 3 ep, warmup 0.03)  -- aggressive default,
#       LLaVA official preset; tends to overfit on small 5k SFT sets.
#   finetune_lora_refined_vqa_1_9_5k_explain_balanced_2gpu.sh
#       (LR 5e-5,  r=32/alpha=64,   2 ep, warmup 0.05)  -- "healthy middle"
#       recipe tuned for 5k samples; loss settles around 0.20-0.35.
#
# This script applies the SAME hyperparameters as the explain_balanced sibling
# but trains on the existing direct_5k data (pre-converted, answers are the
# raw letter-only direct mode, not the explanation mode). Data paths are
# identical to finetune_lora_direct_5k_2gpu.sh so no re-conversion is needed.
#
# Differences vs finetune_lora_direct_5k_2gpu.sh (data unchanged):
#   LEARNING_RATE       2e-4   -> 5e-5            (-75%)
#   MM_PROJECTOR_LR     2e-5   -> 5e-6            (proportional)
#   LORA_R              128    -> 32              (1/4 capacity)
#   LORA_ALPHA          256    -> 64              (alpha = 2 * r)
#   NUM_TRAIN_EPOCHS    3      -> 2               (one less pass)
#   WARMUP_RATIO        0.03   -> 0.05            (slightly longer warmup)
#   SAVE_STEPS          100    -> 50              (twice as many ckpts)
#   SAVE_TOTAL_LIMIT    20     -> 16
#   OUTPUT_DIR          .../direct_5k -> .../direct_5k_balanced
#   RUN_NAME            ..._direct_5k_2gpu -> ..._direct_5k_balanced_2gpu
#   MASTER_PORT         29527  -> 29539
#
# Identical to both siblings:
#   - 2x H20 GPUs (CUDA_VISIBLE_DEVICES=0,1 by default)
#   - per_device_bs=4, grad_accum=2  ->  effective batch = 16
#   - fp16 + ZeRO-3  (bf16+flash-attn deadlocks on this image)
#   - 5000 / 16 ~= 313 step/epoch  ->  2 ep ~= 626 steps  ->  ~12 ckpts at save_steps=50
#   - per-checkpoint exportability via _train_with_step_save.py
#
# Expected end-of-run training loss for direct (letter-only) mode:
#   ~0.10-0.25 = healthy
#   < 0.05     = warning, likely memorising the small label vocabulary
#   > 0.35     = under-fit, raise r / LR / epochs
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# ---- GPUs / launcher ---------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=2
MASTER_PORT="${MASTER_PORT:-29539}"

# ---- Data / output paths -----------------------------------------------------
# direct_5k data is pre-converted on disk; reuse the exact same files as the
# original finetune_lora_direct_5k_2gpu.sh so this is a hyperparam-only sweep.
DATA_PATH="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/bysj_first_run/ok_for_training/direct_5k/vqa.jsonl"
IMAGE_FOLDER="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/bysj_first_run/ok_for_training/direct_5k/images"
OUTPUT_DIR="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/bysj/Llava_v1_5_7B/direct_5k_balanced"

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
RUN_NAME="llava_v1_5_7b_lora_direct_5k_balanced_2gpu"
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
echo "===== LLaVA LoRA SFT (direct_5k, BALANCED hyperparams, 2-GPU H20) ====="
echo "ROOT_DIR        : ${ROOT_DIR}"
echo "GPUs            : ${CUDA_VISIBLE_DEVICES}    (NUM_GPUS=${NUM_GPUS})"
echo "MASTER_PORT     : ${MASTER_PORT}"
echo "DATA_PATH (in)  : ${DATA_PATH}"
echo "DATA_PATH (use) : ${DATA_PATH_RESOLVED}"
echo "IMAGE_FOLDER    : ${IMAGE_FOLDER}"
echo "OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "RUN_NAME        : ${RUN_NAME}"
echo "LR / mm_proj_lr : ${LEARNING_RATE} / ${MM_PROJECTOR_LR}"
echo "LoRA r / alpha  : ${LORA_R} / ${LORA_ALPHA}"
echo "epochs / save   : ${NUM_TRAIN_EPOCHS} epoch / save_steps=${SAVE_STEPS} / limit=${SAVE_TOTAL_LIMIT}"
echo "BS / accum      : per_dev=${PER_DEVICE_TRAIN_BATCH_SIZE}  accum=${GRADIENT_ACCUMULATION_STEPS}  effective=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "precision       : bf16=${BF16} fp16=${FP16} tf32=${TF32}"
echo "========================================================================"

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
