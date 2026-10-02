#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/paths.sh" || exit 1
# Usage: bash scripts/rollout/alfworld.sh <stage> <round> <model> [gpu] [port]
set -euo pipefail

STAGE="${1:?stage required}"
LABEL="${2:?round required}"
MODEL="${3:?model required}"
GPU="${4:-$BACAM_GPU_ALFWORLD}"
PORT="${5:-$BACAM_ALFWORLD_ROLLOUT_PORT}"
check_gpu "$GPU" || exit 2
[[ "$LABEL" != "expert" ]] || {
  echo "BACAM only permits candidate-generated trajectories" >&2
  exit 2
}

EXPERIMENT="$BACAM_ROOT"
VLLM="$BACAM_VLLM"
ALFWORLD_PYTHON="$BACAM_ALFWORLD_PYTHON"
REL="$LABEL/current_raw"
TAG="bacam-wtsa-$STAGE-$LABEL-alfworld"
STATE_DIR="$EXPERIMENT/data/$STAGE/$REL/states"
LOG_DIR="$EXPERIMENT/logs/$STAGE/$REL"
OUTPUT="$STATE_DIR/alfworld_student_states.jsonl"
SUMMARY="$STATE_DIR/alfworld_student_states.summary.json"
SPLIT="$EXPERIMENT/data/splits/alfworld_split.json"

mkdir -p "$STATE_DIR" "$LOG_DIR"
command -v "$VLLM" >/dev/null 2>&1 || { echo "missing vLLM: $VLLM" >&2; exit 1; }
command -v "$ALFWORLD_PYTHON" >/dev/null 2>&1 || { echo "missing ALFWorld Python: $ALFWORLD_PYTHON" >&2; exit 1; }
[ -s "$MODEL/model.safetensors.index.json" ] || { echo "incomplete model: $MODEL" >&2; exit 1; }
[ -s "$SPLIT" ] || { echo "missing split: $SPLIT" >&2; exit 1; }

if [ -s "$SUMMARY" ] && "$ALFWORLD_PYTHON" - "$SUMMARY" "$STAGE" "$LABEL" "$MODEL" <<'PY'
import json, sys
from pathlib import Path
summary = json.load(open(sys.argv[1]))
expected = {
    "stage": sys.argv[2],
    "round": sys.argv[3],
    "model": str(Path(sys.argv[4]).resolve()),
}
matches = all(summary.get(key) == value for key, value in expected.items())
raise SystemExit(
    0 if matches and summary.get("complete") and summary.get("episodes") == 30 else 1
)
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
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$GPU" "$VLLM" serve "$MODEL" \
  --served-model-name "$TAG" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --generation-config vllm \
  --host 127.0.0.1 \
  --port "$PORT" \
  > "$LOG_DIR/alfworld-server.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 120); do
  if curl --max-time 2 -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ALFWorld candidate server exited; see $LOG_DIR/alfworld-server.log" >&2
    exit 1
  fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "ALFWorld candidate server did not become ready" >&2; exit 1; }

"$ALFWORLD_PYTHON" -m bacam.rollout.alfworld \
  --stage "$STAGE" \
  --round "$LABEL" \
  --model "$MODEL" \
  --api-base "http://127.0.0.1:$PORT/v1" \
  --served-model-name "$TAG" \
  --split "$SPLIT" \
  --output "$OUTPUT" \
  --batch-size 8 \
  --request-workers 8 \
  --seed 20260816 \
  --max-steps 50 \
  --history-length 2 \
  --max-tokens 512 \
  --temperature 0.4 \
  --top-p 1.0 \
  --resume \
  > "$LOG_DIR/alfworld-rollout.log" 2>&1

echo "$OUTPUT"
