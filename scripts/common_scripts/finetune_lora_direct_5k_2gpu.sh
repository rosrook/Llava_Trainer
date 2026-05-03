#!/usr/bin/env bash
# ==============================================================================
# LLaVA-1.5-7B LoRA SFT on direct_5k (5000 samples), single-file hard-coded.
#
# Hardware: 2x NVIDIA H20-3e (~143 GB each), GPUs 0 and 1.
# Data    : /mnt/.../direct_5k/{vqa.jsonl|vqa.json, images/}
# Output  : /mnt/.../bysj/Llava_v1_5_7B/direct_5k/
#
# Run (single command):
#   bash Llava/LLaVA/scripts/common_scripts/finetune_lora_direct_5k_2gpu.sh
#
# Notes:
#   - All hyperparameters are HARD-CODED below. Edit the file directly to
#     change them; this script intentionally does NOT read CONFIG_FILE.
#   - Each saved checkpoint-XXX/ is self-contained:
#       * LoRA adapter (HF Trainer)
#       * non_lora_trainables.bin    (LLaVATrainer._save_checkpoint)
#       * config.json + step_loss.json (scripts/_train_with_step_save.py)
#     Standalone export of any step:
#       python3 scripts/export_lora_checkpoint_to_hf.py \
#         --base_model    liuhaotian/llava-v1.5-7b \
#         --checkpoint_dir <OUTPUT_DIR>/checkpoint-<STEP> \
#         --output_dir    <ANY_HF_DIR> --bf16
#     (or scripts/export_llava_lora_checkpoint_hf.py with --shared-dir.)
#   - At the end of training, OUTPUT_DIR/best_step_candidates.json ranks
#     checkpoints by smoothed training loss as a HEURISTIC starting point;
#     a copy of the lowest-loss checkpoint is placed at OUTPUT_DIR/
#     best_by_train_loss/. Confirm with a real eval (MMBench) before picking.
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# ---- GPUs / launcher ---------------------------------------------------------
export CUDA_VISIBLE_DEVICES="0,1"
NUM_GPUS=2
MASTER_PORT="29527"

# ---- Data / output paths -----------------------------------------------------
DATA_PATH="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/bysj_first_run/ok_for_training/direct_5k/vqa.jsonl"
IMAGE_FOLDER="/mnt/tidal-alsh01/dataset/perceptionVLMData/zhuxuzhou_test_data/bysj_first_run/ok_for_training/direct_5k/images"
OUTPUT_DIR="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/bysj/Llava_v1_5_7B/direct_5k"

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
# Precision: bf16 + flash-attn + ZeRO-3 has been observed to deadlock /
# misbehave on this machine's image. Falling back to fp16 (with DeepSpeed's
# dynamic loss scaling, see scripts/zero3.json) which is the configuration
# we know works on H20-3e.
BF16=False
FP16=True

LORA_ENABLE=True
LORA_R=128
LORA_ALPHA=256
MM_PROJECTOR_LR=2e-5

# ---- Optimizer / schedule (5k samples, 3 epochs) -----------------------------
# 2 GPU x bs=4 x grad_accum=2 = effective batch 16
# 5000 / 16 = ~313 steps/epoch  ->  3 epochs ~= 940 steps
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
# 940 / 100 ~= 9-10 ckpts. Set 20 to keep all.
SAVE_TOTAL_LIMIT=20
LOGGING_STEPS=5

REPORT_TO="tensorboard"
LOGGING_DIR="${OUTPUT_DIR}/tb"
RUN_NAME="llava_v1_5_7b_lora_direct_5k_2gpu"
LOGGING_FIRST_STEP=True
LOG_LEVEL=info
LOGGING_NAN_INF_FILTER=True
SEED=42
SAVE_SAFETENSORS=True

# ---- Runtime ----------------------------------------------------------------
TF32=True
MODEL_MAX_LENGTH=2048
GRADIENT_CHECKPOINTING=True
# Forcing use_reentrant=False is done inside the wrapper (see
# scripts/_train_with_step_save.py) because transformers 4.37 does not
# accept JSON for --gradient_checkpointing_kwargs from the CLI.
DATALOADER_NUM_WORKERS=4
LAZY_PREPROCESS=True

# ---- DeepSpeed ---------------------------------------------------------------
# H20 has plenty of HBM, no need for optimizer offload. Use plain ZeRO-3.
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
echo "==================== LLaVA LoRA SFT (direct_5k, 2-GPU H20) ===================="
echo "ROOT_DIR        : ${ROOT_DIR}"
echo "GPUs            : ${CUDA_VISIBLE_DEVICES}    (NUM_GPUS=${NUM_GPUS})"
echo "MASTER_PORT     : ${MASTER_PORT}"
echo "DATA_PATH (in)  : ${DATA_PATH}"
echo "DATA_PATH (use) : ${DATA_PATH_RESOLVED}"
echo "IMAGE_FOLDER    : ${IMAGE_FOLDER}"
echo "OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "DEEPSPEED       : ${DEEPSPEED_CONFIG}"
echo "TRAIN_ENTRY     : ${TRAIN_ENTRY}"
echo "BS / accum      : per_dev=${PER_DEVICE_TRAIN_BATCH_SIZE}  accum=${GRADIENT_ACCUMULATION_STEPS}  effective=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Epochs / save   : epochs=${NUM_TRAIN_EPOCHS}  save_steps=${SAVE_STEPS}  save_total_limit=${SAVE_TOTAL_LIMIT}"
echo "LR / scheduler  : lr=${LEARNING_RATE}  warmup=${WARMUP_RATIO}  scheduler=${LR_SCHEDULER_TYPE}"
echo "LoRA            : r=${LORA_R}  alpha=${LORA_ALPHA}  mm_projector_lr=${MM_PROJECTOR_LR}"
echo "================================================================================"

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
