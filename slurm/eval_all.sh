#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT_PATH=${1:-llava-qwen-7b-dpo}
DATA_ROOT=${2:-data/interleave_data}

TASKS=(
  "multi_image_in_domain"
  "multi_image_out_domain"
  "multi_view_in_domain"
)

for task in "${TASKS[@]}"; do
  "${SCRIPT_DIR}/eval_interleave_3d.sh" "${CKPT_PATH}" "${DATA_ROOT}" "${task}"
done
