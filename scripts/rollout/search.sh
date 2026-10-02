#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/paths.sh" || exit 1
# Usage: bash scripts/rollout/search.sh <stage> <round> <model> [gpu] [port]
set -euo pipefail

STAGE="${1:?stage required}"
LABEL="${2:?round required}"
MODEL="${3:?model required}"
GPU="${4:-$BACAM_GPU_SEARCH}"
check_gpu "$GPU" || exit 2
PORT="${5:-$BACAM_SEARCH_ROLLOUT_PORT}"
EXPERIMENT="$BACAM_ROOT"
[[ "$LABEL" != "expert" ]] || { echo "BACAM only permits candidate-generated trajectories" >&2; exit 2; }
REL="$LABEL/current_raw"
TAG="bacam-wtsa-$STAGE-$LABEL"
SAVE_DIR="$EXPERIMENT/artifacts/$STAGE/$REL/search/run"
STATE_DIR="$EXPERIMENT/data/$STAGE/$REL/states"
LOG_DIR="$EXPERIMENT/logs/$STAGE/$REL"
PY="$BACAM_PYTHON"
"$PY" -m bacam.data.prepare_search_pool --half train --overwrite \
  --selection-key "$STAGE:$LABEL"
DATASET=$("$BACAM_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_name"])' "$EXPERIMENT/data/splits/search_pool_dataset.json")
POOL_N=$("$BACAM_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["questions"])' "$EXPERIMENT/data/splits/search_pool_dataset.json")
CACHE="$SAVE_DIR/cache_${DATASET}_dev.jsonl"
mkdir -p "$SAVE_DIR" "$STATE_DIR" "$LOG_DIR"
unset http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"
curl -sf http://127.0.0.1:$BACAM_RETRIEVER_PORT/health >/dev/null || { echo "retriever $BACAM_RETRIEVER_PORT not ready"; exit 1; }

( export MODEL="$MODEL" SERVED_NAME="$TAG" SGL_GPUS="$GPU" SGL_TP=1 SGL_PORT="$PORT"
  exec setsid bash "$EXPERIMENT/scripts/services/start_search.sh"
) > "$LOG_DIR/sglang.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 5; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "sglang not ready"; exit 1; }
for attempt in 1 2 3; do
  ( cd "${BACAM_RESEARCH_ROOT}/scripts/evaluation"
    "$BACAM_SEARCH_PYTHON" run_eval.py \
      --config_path eval_config.yaml --method_name research \
      --data_dir "${BACAM_DATA_ROOT}/flashrag_eval" --dataset_name "$DATASET" --split dev \
      --save_dir "$SAVE_DIR" --save_note "$TAG" \
      --sgl_remote_url "http://127.0.0.1:$PORT" --remote_retriever_url http://127.0.0.1:$BACAM_RETRIEVER_PORT \
      --generator_model "$MODEL" --apply_chat True
  ) > "$LOG_DIR/search-$attempt.log" 2>&1 || true
  count=$([ -f "$CACHE" ] && wc -l < "$CACHE" || echo 0)
  [ "$count" -ge "$POOL_N" ] && break
done
kill "$SERVER_PID" 2>/dev/null || true
trap - EXIT
ROLLOUT=$(find "$SAVE_DIR" -mindepth 2 -maxdepth 2 -name intermediate_data.json -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$ROLLOUT" ] || { echo "search rollout missing"; exit 1; }
"$BACAM_PYTHON" -m bacam.rollout.collect_search_states --stage "$STAGE" \
  --model "$MODEL" --rollout "$ROLLOUT" --output-dir "$STATE_DIR" > "$LOG_DIR/search-collect.log" 2>&1
echo "$STATE_DIR/search_student_states.jsonl"
