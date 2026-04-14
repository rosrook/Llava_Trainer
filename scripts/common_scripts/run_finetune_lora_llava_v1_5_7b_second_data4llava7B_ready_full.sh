#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for second_data4llava7B_ready_full.
# You can still override CONFIG_FILE/NUM_GPUS/CUDA_VISIBLE_DEVICES/MASTER_PORT from env.
#
# Usage:
#   MASTER_PORT=29527 bash scripts/common_scripts/run_finetune_lora_llava_v1_5_7b_second_data4llava7B_ready_full.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export CONFIG_FILE="${CONFIG_FILE:-./scripts/configs/finetune_lora_second_data4llava7B_ready_full.sh}"

bash "${ROOT_DIR}/scripts/common_scripts/run_finetune_lora_llava_v1_5_7b_mydata.sh"
