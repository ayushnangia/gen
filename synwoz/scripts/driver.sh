#!/bin/bash

# Batch generation driver for SynWOZ
# Example usage:
#   UV="uv" ./synwoz/scripts/driver.sh 500
# Defaults to using uv run; set UV="python" to skip uv.

set -euo pipefail

BATCH_SIZE=200
TEMPS=(0.6 0.7 0.8)
TOPPS=(0.95 1.0)
FREQS=(0.4 0.5 0.6 0.7)
PRES=(0.4 0.5 0.6 0.7)

TOTAL=${1:-}
if [[ -z "$TOTAL" ]]; then
  read -rp "Enter the total number of dialogues to generate: " TOTAL
fi

UV_BIN=${UV:-uv}
PY_CMD=($UV_BIN run python -m synwoz gen-parallel --)

full_batches=$(( TOTAL / BATCH_SIZE ))
remainder=$(( TOTAL % BATCH_SIZE ))

run_batch() {
  local batch=$1
  local size=$2
  local t=${TEMPS[$RANDOM % ${#TEMPS[@]}]}
  local p=${TOPPS[$RANDOM % ${#TOPPS[@]}]}
  local f=${FREQS[$RANDOM % ${#FREQS[@]}]}
  local r=${PRES[$RANDOM % ${#PRES[@]}]}
  echo "Batch ${batch}: size=${size} temp=${t} top_p=${p} freq=${f} presence=${r}"
  "${PY_CMD[@]}" \
    --total_generations "$size" \
    --temperature "$t" \
    --top_p "$p" \
    --frequency_penalty "$f" \
    --presence_penalty "$r"
}

echo "Running ${full_batches} batches of ${BATCH_SIZE}, remainder ${remainder}"
for ((i=1; i<=full_batches; i++)); do
  run_batch "$i" "$BATCH_SIZE"
done

if [[ $remainder -gt 0 ]]; then
  run_batch "extra" "$remainder"
fi
