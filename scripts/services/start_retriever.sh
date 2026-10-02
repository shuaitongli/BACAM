#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/paths.sh" || exit 1
# Start the resident retriever that the ReSearch pipeline searches against (idempotent).
#   bash scripts/services/start_retriever.sh
#
# Both scripts/evaluate/run_search_eval.sh and scripts/rollout/search.sh refuse to launch
# sglang until the configured retriever port is healthy.
set -u

EXPERIMENT="$BACAM_ROOT"
LOG="$EXPERIMENT/logs/eval/retriever.log"
PORT="$BACAM_RETRIEVER_PORT"
CONFIG="${BACAM_RETRIEVER_CONFIG:-$BACAM_RESEARCH_ROOT/scripts/serving/retriever_config.yaml}"
mkdir -p "$(dirname "$LOG")"

unset http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"

if curl -sf --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "retriever 已经在跑，跳过"; curl -s "http://127.0.0.1:$PORT/health"; echo; exit 0
fi

# setsid: the service blocks in the foreground, so without its own session it dies with
# the parent shell -- exit 137 with oom_kill count 0, which reads like an OOM but isn't.
echo "起 retriever（61G 索引常驻内存，加载 3-6 分钟）"
cd "$BACAM_RESEARCH_ROOT/scripts/serving"
OMP_NUM_THREADS="${OMP_THREADS:-2}" setsid nohup "$BACAM_SEARCH_PYTHON" \
  retriever_serving.py --config "$CONFIG" \
  --max_concurrency "${RETRIEVER_CONCURRENCY:-64}" --port "$PORT" \
  > "$LOG" 2>&1 < /dev/null &
disown

for i in $(seq 1 90); do
  curl -sf --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && {
    echo "就绪，用时约 $((i * 10)) 秒"; curl -s "http://127.0.0.1:$PORT/health"; echo; exit 0
  }
  sleep 10
done
echo "15 分钟还没就绪，看 $LOG" >&2
exit 1
