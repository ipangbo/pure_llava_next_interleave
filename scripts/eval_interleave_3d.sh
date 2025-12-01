#!/bin/bash

alias python=python3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <ckpt_path_or_repo> <data_root> <split_name>" >&2
  exit 1
fi

CKPT_PATH=$1
NAME=$(basename "$CKPT_PATH")
DATA_PATH=$2
EVAL_TYPE=$3
JSON_PATH="${DATA_PATH}/${EVAL_TYPE}.json"

RESULT_NAME="${SCRIPT_DIR}/logs/${NAME}/${EVAL_TYPE}"
mkdir -p "${RESULT_NAME}"

DEFAULT_CUDA_VISIBLE_DEVICES=$(python3 - <<'PY'
from constants import DEFAULT_CUDA_DEVICES
print(DEFAULT_CUDA_DEVICES)
PY
)

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES}"
fi

TEMPERATURE=${TEMPERATURE:-0}

ANSWERS_FILE="${RESULT_NAME}/result.jsonl"
if [ -f "${ANSWERS_FILE}" ]; then
  rm "${ANSWERS_FILE}"
fi

python3 -m eval.interleave_vqa \
    --model-path "${CKPT_PATH}" \
    --question-file "${JSON_PATH}" \
    --answers-file "${ANSWERS_FILE}" \
    --image-folder "${DATA_PATH}" \
    --extra-prompt "" \
    --temperature "${TEMPERATURE}"

python3 -m eval.evaluate_interleave --result-dir "${RESULT_NAME}"
