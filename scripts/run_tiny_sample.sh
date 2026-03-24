#!/usr/bin/env bash
# Smallest run: mt-bench `datasets/mt-bench/sample2.jsonl` (2 examples) with `--sampling`.
# Start vLLM first (same port), or set USE_OPENAI=1 for gpt-4o-mini.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

if [[ "${USE_OPENAI:-0}" == "1" ]]; then
  export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
  exec python3 src/run/main.py \
    --dataset_name mt-bench \
    --model_name "${OPENAI_MODEL:-gpt-4o-mini}" \
    --tokenizer "${TOKENIZER:-google/gemma-2-9b-it}" \
    --task_name openended_esl \
    --cefr_level A \
    --l1 Arabic \
    --save_path ./outputs/mt-bench \
    --file_name tiny_sample \
    --batch_size 2 \
    --data_path . \
    --sampling \
    --max_chain_depth 2 \
    "$@"
else
  exec python3 src/run/main.py \
    --dataset_name mt-bench \
    --model_name "${HF_MODEL:-google/gemma-2-27b-it}" \
    --tokenizer "${HF_MODEL:-google/gemma-2-27b-it}" \
    --task_name openended_esl \
    --cefr_level A \
    --l1 Arabic \
    --save_path ./outputs/mt-bench \
    --file_name tiny_sample \
    --batch_size 2 \
    --data_path . \
    --port_num "${PORT_NUM:-6001}" \
    --sampling \
    --max_chain_depth 2 \
    "$@"
fi
