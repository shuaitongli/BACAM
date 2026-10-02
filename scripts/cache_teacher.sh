#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/paths.sh" || exit 1
# Cache one manifest stream per GPU; teacher assignment is validated in Python
# against the stage's old_domains/new_domains configuration.
set -euo pipefail

STAGE="${1:?stage required}"
ROUND="${2:?round required}"
EXPERIMENT="$BACAM_ROOT"
PY="$BACAM_PYTHON"
CONFIG="$EXPERIMENT/configs/experiment.json"
LOG_DIR="$EXPERIMENT/logs/$STAGE/$ROUND/teacher_cache"

mapfile -t STREAMS < <("$PY" - "$CONFIG" "$STAGE" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
stage = next(item for item in config["stages"] if item["name"] == sys.argv[2])
domains = stage["old_domains"] + stage["new_domains"]
if len(domains) != len(set(domains)):
    raise SystemExit("stage contains duplicate domains")
for domain in domains:
    print(f"{domain}_current")
PY
)
IFS=, read -r -a GPUS <<< "$BACAM_GPUS"

[[ "${#STREAMS[@]}" -gt 0 ]] || { echo "no cache streams configured for $STAGE" >&2; exit 1; }
[[ "${#STREAMS[@]}" -le "${#GPUS[@]}" ]] || {
  echo "$STAGE has ${#STREAMS[@]} streams but only ${#GPUS[@]} cache GPUs" >&2
  exit 1
}
mkdir -p "$LOG_DIR"

pids=()
for index in "${!STREAMS[@]}"; do
  stream="${STREAMS[$index]}"
  gpu="${GPUS[$index]}"
  log="$LOG_DIR/$stream.log"
  echo "cache $stream on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m bacam.data.cache_teacher \
    --stage "$STAGE" --round "$ROUND" --stream "$stream" --device cuda:0 \
    > "$log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "cache ${STREAMS[$index]} failed; see $LOG_DIR/${STREAMS[$index]}.log" >&2
    status=1
  fi
done
exit "$status"
