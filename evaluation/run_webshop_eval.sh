#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/paths.sh" || exit 1
# Usage: bash evaluation/run_webshop_eval.sh [gpu] [port]
set -euo pipefail

GPU="${1:-$BACAM_GPU_WEBSHOP}"
PORT="${2:-$BACAM_WEBSHOP_EVAL_PORT}"
check_gpu "$GPU" || exit 2

EXPERIMENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${EVAL_MODEL_ID:-expert}"
MODEL="${EVAL_MODEL_PATH:-${BACAM_MODEL_ROOT}/GiGPO-Qwen2.5-7B-Instruct-WebShop}"
SERVED_NAME="${EVAL_SERVED_NAME:-webshop-$MODEL_ID}"
RESULT_TAG="${RESULT_TAG:-vllm}"
RESUME="${RESUME:-1}"
GOAL_COUNT="${GOAL_COUNT:-500}"
VLLM="$BACAM_VLLM"
WEBSHOP_PYTHON="$BACAM_WEBSHOP_PYTHON"
OUTPUT_DIR="$EXPERIMENT/artifacts/eval/webshop"
LOG_DIR="$EXPERIMENT/logs/eval"
OUTPUT="$OUTPUT_DIR/${MODEL_ID}_test_${RESULT_TAG}.json"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
command -v "$VLLM" >/dev/null 2>&1 || { echo "missing vLLM: $VLLM" >&2; exit 1; }
command -v "$WEBSHOP_PYTHON" >/dev/null 2>&1 || { echo "missing WebShop Python" >&2; exit 1; }
[[ -s "$MODEL/model.safetensors.index.json" ]] || { echo "incomplete model: $MODEL" >&2; exit 1; }
[[ -s "${BACAM_DATA_ROOT}/webshop/search_engine_1k/index_manifest.json" ]] || {
  echo "missing WebShop 1k index" >&2; exit 1;
}

if [[ -s "$OUTPUT" ]] && "$WEBSHOP_PYTHON" - "$OUTPUT" "$MODEL" "$GOAL_COUNT" <<'PY'
import json, sys
from pathlib import Path
x=json.load(open(sys.argv[1]))
ok=(x.get('status')=='complete' and
    x.get('model',{}).get('path')==str(Path(sys.argv[2]).resolve()) and
    x.get('evaluation',{}).get('goals_requested')==int(sys.argv[3]) and
    x.get('evaluation',{}).get('goals_completed')==int(sys.argv[3]) and
    x.get('evaluation',{}).get('seed')==20260816 and
    x.get('evaluation',{}).get('max_steps')==15 and
    x.get('evaluation',{}).get('history_length')==2 and
    x.get('evaluation',{}).get('generation')=={
        'max_tokens': 512, 'temperature': 0.4, 'top_p': 1.0})
raise SystemExit(0 if ok else 1)
PY
then
  echo "$OUTPUT"
  exit 0
fi

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"
if curl --max-time 2 -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy server" >&2
  exit 1
fi

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$GPU" "$VLLM" serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --generation-config vllm \
  --host 127.0.0.1 \
  --port "$PORT" \
  > "$LOG_DIR/webshop-$MODEL_ID-vllm-server.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 120); do
  if curl --max-time 2 -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "WebShop vLLM exited; see $LOG_DIR/webshop-$MODEL_ID-vllm-server.log" >&2
    exit 1
  fi
  sleep 5
done
[[ "$ready" == 1 ]] || { echo "WebShop vLLM did not become ready" >&2; exit 1; }

resume_args=()
[[ "$RESUME" == 1 ]] && resume_args+=(--resume)
PATH="$(dirname "$WEBSHOP_PYTHON"):$PATH" "$WEBSHOP_PYTHON" \
  "$EXPERIMENT/evaluation/eval_webshop.py" \
  --goal-count "$GOAL_COUNT" \
  --batch-size "${BATCH_SIZE:-8}" \
  --request-workers "${REQUEST_WORKERS:-8}" \
  --seed 20260816 \
  --max-steps 15 \
  --history-length 2 \
  --max-tokens 512 \
  --temperature 0.4 \
  --top-p 1.0 \
  --api-base "http://127.0.0.1:$PORT/v1" \
  --served-model-name "$SERVED_NAME" \
  --model-path "$MODEL" \
  --output "$OUTPUT" \
  "${resume_args[@]}"

echo "$OUTPUT"
