#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/paths.sh" || exit 1
# Reproduce the released ALFWorld expert on IID, OOD, or both splits.
#
# Usage:
#   bash scripts/evaluate/run_alfworld_expert_eval.sh [iid|ood|both] [gpu] [port]
#
# Optional environment variables:
#   ALL_GAMES=0 ROUNDS=128 BATCH_SIZE=8 REQUEST_WORKERS=8 RESUME=1 RESULT_TAG=vllm
#   EVAL_MODEL_ID=expert EVAL_MODEL_PATH=/path/to/model EVAL_SERVED_NAME=alfworld-expert
set -euo pipefail

SPLIT="${1:-both}"
GPU="${2:-$BACAM_GPU_ALFWORLD}"
PORT="${3:-$BACAM_ALFWORLD_EVAL_PORT}"
ROUNDS="${ROUNDS:-128}"
ALL_GAMES="${ALL_GAMES:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
REQUEST_WORKERS="${REQUEST_WORKERS:-8}"
TEMPERATURE="${TEMPERATURE:-0.4}"
RESUME="${RESUME:-1}"
RESULT_TAG="${RESULT_TAG:-vllm}"
EVAL_MODEL_ID="${EVAL_MODEL_ID:-expert}"

case "$SPLIT" in
  iid|ood|both) ;;
  *) echo "split must be iid, ood, or both" >&2; exit 2 ;;
esac
check_gpu "$GPU" || exit 2
case "$RESULT_TAG" in
  *[!A-Za-z0-9._-]*|'') echo "RESULT_TAG contains invalid characters" >&2; exit 2 ;;
esac
case "$EVAL_MODEL_ID" in
  *[!A-Za-z0-9._-]*|'') echo "EVAL_MODEL_ID contains invalid characters" >&2; exit 2 ;;
esac
case "$ALL_GAMES" in
  0|1) ;;
  *) echo "ALL_GAMES must be 0 or 1" >&2; exit 2 ;;
esac

EXPERIMENT="$BACAM_ROOT"
EVALUATOR="bacam.evaluation.eval_alfworld_expert"
MODEL="${EVAL_MODEL_PATH:-${BACAM_MODEL_ROOT}/GiGPO-Qwen2.5-7B-Instruct-ALFWorld}"
VLLM="$BACAM_VLLM"
ALFWORLD_PYTHON="$BACAM_ALFWORLD_PYTHON"
OUTPUT_DIR="$EXPERIMENT/artifacts/eval/alfworld"
LOG_DIR="$EXPERIMENT/logs/eval"
SERVED_NAME="${EVAL_SERVED_NAME:-alfworld-$EVAL_MODEL_ID}"
API_BASE="http://127.0.0.1:$PORT/v1"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
command -v "$VLLM" >/dev/null 2>&1 || { echo "vLLM executable not found: $VLLM" >&2; exit 1; }
command -v "$ALFWORLD_PYTHON" >/dev/null 2>&1 || {
  echo "ALFWorld Python not found: $ALFWORLD_PYTHON" >&2; exit 1;
}
[ -s "$MODEL/model.safetensors.index.json" ] || {
  echo "Model is incomplete: $MODEL" >&2; exit 1;
}

if curl --max-time 2 -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "Port $PORT already has a healthy server; refusing to replace it." >&2
  exit 1
fi

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting ALFWorld $EVAL_MODEL_ID vLLM server on GPU $GPU, port $PORT"
CUDA_VISIBLE_DEVICES="$GPU" "$VLLM" serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --generation-config vllm \
    --host 127.0.0.1 \
    --port "$PORT" \
    >"$LOG_DIR/alfworld-$EVAL_MODEL_ID-vllm.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 120); do
  if curl --max-time 2 -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready; see $LOG_DIR/alfworld-$EVAL_MODEL_ID-vllm.log" >&2
    exit 1
  fi
  sleep 5
done
[ "$ready" = 1 ] || {
  echo "vLLM did not become ready; see $LOG_DIR/alfworld-$EVAL_MODEL_ID-vllm.log" >&2
  exit 1
}

run_split() {
  local current_split="$1"
  local resume_args=()
  local selection_args=()
  if [ "$RESUME" = "1" ]; then
    resume_args+=(--resume)
  fi
  if [ "$ALL_GAMES" = "1" ]; then
    selection_args+=(--all-games)
  fi
  "$ALFWORLD_PYTHON" -m "$EVALUATOR" \
    --split "$current_split" \
    --rounds "$ROUNDS" \
    --batch-size "$BATCH_SIZE" \
    --request-workers "$REQUEST_WORKERS" \
    --temperature "$TEMPERATURE" \
    --api-base "$API_BASE" \
    --served-model-name "$SERVED_NAME" \
    --inference-backend "vllm-0.11.0" \
    --model-path "$MODEL" \
    --output "$OUTPUT_DIR/${EVAL_MODEL_ID}_${current_split}_${RESULT_TAG}.json" \
    "${selection_args[@]}" \
    "${resume_args[@]}"
}

case "$SPLIT" in
  iid) run_split iid ;;
  ood) run_split ood ;;
  both)
    run_split iid
    run_split ood
    ;;
esac

echo "ALFWorld $EVAL_MODEL_ID evaluation complete: $OUTPUT_DIR"
