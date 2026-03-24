#!/usr/bin/env bash
# Full AlpacaFarm eval: CEFR A + L1 together (openended_esl), per L1 language.
# Caps stacked transforms at 3; semantic checker ON (default — do not pass --skip_semantic_check).
# Point at your vLLM OpenAI-compatible server (default port 6002).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export DATA_PATH="${DATA_PATH:-$ROOT}"

PORT_NUM="${PORT_NUM:-6002}"
MODEL_NAME="${MODEL_NAME:-google/gemma-2-27b-it}"
BATCH_SIZE="${BATCH_SIZE:-15}"
MAX_CHAIN_DEPTH="${MAX_CHAIN_DEPTH:-3}"

L1_LANGUAGES=(
  Arabic French German Italian Japanese Mandarin Portuguese Russian Spanish Turkish
)

echo "=== openended_esl: CEFR A + L1 (max_chain_depth=${MAX_CHAIN_DEPTH}, semantic check ON) ==="
for l1 in "${L1_LANGUAGES[@]}"; do
  echo "--- L1: $l1 ---"
  python src/run/main.py \
    --dataset_name alpacafarm \
    --task_name openended_esl \
    --cefr_level A \
    --l1 "$l1" \
    --port_num "$PORT_NUM" \
    --model_name "$MODEL_NAME" \
    --tokenizer "$MODEL_NAME" \
    --save_path "./outputs/alpacafarm/openended_esl_A_${l1}" \
    --file_name alpacafarm_full \
    --data_path "$DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_chain_depth "$MAX_CHAIN_DEPTH"
done

echo "Done."
