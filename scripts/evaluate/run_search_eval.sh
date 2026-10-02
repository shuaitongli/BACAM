#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/paths.sh" || exit 1
# Score one merged model on the preregistered Musique held-out half.
#   bash scripts/evaluate/run_search_eval.sh <model_id> [gpu] [sgl_port]
#
# Serves one model with sglang and evaluates it on experiment 05's Search held-out set.
# The run repeats until the resume cache covers the full held-out set.
set -euo pipefail

MID="${1:?usage: run_search_eval.sh <model_id> [gpu] [sgl_port]}"
GPU="${2:-$BACAM_GPU_SEARCH}"
check_gpu "$GPU" || exit 2
SGL_PORT="${3:-$BACAM_SEARCH_EVAL_PORT}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"

EXPERIMENT="$BACAM_ROOT"
LOGS="$EXPERIMENT/logs/eval"
DONE="$EXPERIMENT/artifacts/eval/.done"
RETRIEVER_PORT="$BACAM_RETRIEVER_PORT"
mkdir -p "$LOGS" "$DONE"

source "$EXPERIMENT/scripts/lib/models.sh"
unset http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"

TAG="$(search_tag "$MID")"
MODEL="$(model_path "$MID")"
SAVE_DIR="$EXPERIMENT/artifacts/eval/search/$MID/run"
mkdir -p "$SAVE_DIR"
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOGS/search.log"; }

[ -d "$MODEL" ] || { say "$MODEL 不存在"; exit 1; }
MODEL_STAMP="$MODEL/model.safetensors.index.json"
[ -s "$MODEL_STAMP" ] || { say "$MODEL_STAMP 不存在"; exit 1; }
STALE_DONE=0
if [ -f "$DONE/search-$MID" ]; then
  [ "$DONE/search-$MID" -nt "$MODEL_STAMP" ] && {
    say "$TAG 已完成且结果比模型新，跳过"; exit 0;
  }
  STALE_DONE=1
  rm -f "$DONE/search-$MID"
fi
curl -sf "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null || {
  say "retriever 未就绪（端口 $RETRIEVER_PORT）"; exit 1; }

DATASET="$("$BACAM_PYTHON" -c "
import json, sys; print(json.load(open(sys.argv[1]))['dataset_name'])" \
  "$EXPERIMENT/data/splits/search_holdout_dataset.json")"
EVAL_N="$("$BACAM_PYTHON" -c "
import json, sys; print(json.load(open(sys.argv[1]))['questions'])" \
  "$EXPERIMENT/data/splits/search_holdout_dataset.json")"
CACHE="$SAVE_DIR/cache_${DATASET}_dev.jsonl"
[ "$STALE_DONE" -eq 0 ] || rm -f "$CACHE"

pkill -f "sglang.launch_server.*--port $SGL_PORT" 2>/dev/null || true
sleep 3
say "$TAG 起 sglang（GPU $GPU，port $SGL_PORT）"
( export MODEL="$MODEL" SERVED_NAME="$TAG" SGL_GPUS="$GPU" SGL_TP=1 SGL_PORT="$SGL_PORT"
  exec setsid bash "${BACAM_RESEARCH_ROOT}/run_search_eval.sh" sglang
) > "$LOGS/sglang-$MID.log" 2>&1 &

ready=0
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$SGL_PORT/health" >/dev/null 2>&1 && { ready=1; break; }
  sleep 5
done
[ "$ready" = 1 ] || { say "sglang 未就绪"; pkill -f "sglang.launch_server.*--port $SGL_PORT" || true; exit 1; }

round=0
while [ "$round" -lt "$MAX_ROUNDS" ]; do
  round=$((round + 1))
  ( cd "${BACAM_RESEARCH_ROOT}/scripts/evaluation"
    "$BACAM_SEARCH_PYTHON" run_eval.py \
      --config_path eval_config.yaml --method_name research \
      --data_dir "${BACAM_DATA_ROOT}/flashrag_eval" \
      --dataset_name "$DATASET" --split dev \
      --save_dir "$SAVE_DIR" --save_note "$TAG" \
      --sgl_remote_url "http://127.0.0.1:$SGL_PORT" \
      --remote_retriever_url "http://127.0.0.1:$RETRIEVER_PORT" \
      --generator_model "$MODEL" --apply_chat True
  ) > "$LOGS/search-$MID-round-$round.log" 2>&1 || say "第 $round 轮非零退出，继续重试"
  count=$([ -f "$CACHE" ] && wc -l < "$CACHE" || echo 0)
  say "$TAG 第 $round 轮：$count/$EVAL_N"
  [ "$count" -ge "$EVAL_N" ] && break
done

pkill -f "sglang.launch_server.*--port $SGL_PORT" 2>/dev/null || true
[ "${count:-0}" -eq "$EVAL_N" ] || {
  say "$TAG 评测未完成：${count:-0}/$EVAL_N，保留结果供下次继续"; exit 1; }
if ! "$BACAM_PYTHON" - "$SAVE_DIR" "$EVAL_N" <<'PY'
import json, sys
from pathlib import Path
paths = sorted(Path(sys.argv[1]).glob("*/intermediate_data.json"),
               key=lambda path: path.stat().st_mtime)
raise SystemExit(0 if paths and len(json.loads(paths[-1].read_text())) == int(sys.argv[2]) else 1)
PY
then
  say "intermediate_data.json 缺失或数量不完整，未写入完成标记"; exit 1
fi
touch "$DONE/search-$MID"
say "$TAG 完成，结果在 $SAVE_DIR"
