#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/paths.sh" || exit 1
# Usage: bash scripts/rollout/tool.sh <stage> <round> <model> [gpu]
set -euo pipefail

STAGE="${1:?stage required}"
LABEL="${2:?round required}"
MODEL="${3:?model required}"
GPU="${4:-$BACAM_GPU_TOOL}"
check_gpu "$GPU" || exit 2
EXPERIMENT="$BACAM_ROOT"
BFCL_ROOT="$BACAM_BFCL_ROOT"
[[ "$LABEL" != "expert" ]] || { echo "BACAM only permits candidate-generated trajectories" >&2; exit 2; }
REL="$LABEL/current_raw"
TAG="bacam-wtsa-$STAGE-$LABEL"
RESULT_DIR="$EXPERIMENT/artifacts/$STAGE/$REL/bfcl/result"
STATE_DIR="$EXPERIMENT/data/$STAGE/$REL/states"
LOG_DIR="$EXPERIMENT/logs/$STAGE/$REL"
IDS_FILE=$BFCL_ROOT/test_case_ids_to_generate.json
mkdir -p "$RESULT_DIR" "$STATE_DIR" "$LOG_DIR"

unset http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"
"$BACAM_TOOL_PYTHON" -m bacam.evaluation.register_bfcl_models > "$LOG_DIR/tool-register.log" 2>&1
"$BACAM_TOOL_PYTHON" - "$EXPERIMENT/data/splits/bfcl_split.json" "$EXPERIMENT/configs/experiment.json" \
  "$IDS_FILE" "$STAGE" "$LABEL" <<'PY'
import hashlib, json, sys
split = json.load(open(sys.argv[1]))
config = json.load(open(sys.argv[2]))
ids = split["categories"]["multi_turn_base"]["train"]
seed = f'{config["seed"]}:{sys.argv[4]}:{sys.argv[5]}:tool'
ids = sorted(ids, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest())
ids = ids[:int(config["data"]["rollout_episodes"]["tool"])]
with open(sys.argv[3], "w") as handle:
    json.dump({"multi_turn_base": ids},
              handle, indent=2)
    handle.write("\n")
PY
CUDA_VISIBLE_DEVICES="$GPU" VLLM_PORT="$BACAM_TOOL_PORT" \
  "$BACAM_BFCL" generate --model "$TAG" --local-model-path "$MODEL" \
    --test-category multi_turn_base --run-ids \
    --num-gpus 1 --gpu-memory-utilization 0.85 --result-dir "$RESULT_DIR" --allow-overwrite \
    > "$LOG_DIR/tool-generate.log" 2>&1
"$BACAM_TOOL_PYTHON" -m bacam.rollout.collect_tool_states --stage "$STAGE" --model "$MODEL" \
  --rollout-dir "$RESULT_DIR/$TAG" --candidate-ids "$IDS_FILE" \
  --output-dir "$STATE_DIR" > "$LOG_DIR/tool-collect.log" 2>&1
echo "$STATE_DIR/tool_student_states.jsonl"
