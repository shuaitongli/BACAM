#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/paths.sh" || exit 1
# T4: WebShop + Tool + Search -> ALFWorld.
set -euo pipefail

RESUME=0
if [[ "${1:-}" == "--resume" ]]; then RESUME=1; shift; fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: bash scripts/pipelines/merge_alfworld.sh [--resume]" >&2
  exit 2
fi

EXPERIMENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$BACAM_PYTHON"
TORCHRUN="$BACAM_TORCHRUN"
T3FINAL="${BACAM_MERGE_ROOT}/bacam-wtsa-t3-final"
T4R0="${BACAM_MERGE_ROOT}/bacam-wtsa-t4-r0"
T4R1="${BACAM_MERGE_ROOT}/bacam-wtsa-t4-r1"
T4R2="${BACAM_MERGE_ROOT}/bacam-wtsa-t4-r2"
T4FINAL="${BACAM_MERGE_ROOT}/bacam-wtsa-t4-final"
ALFWORLD_EXPERT="${BACAM_MODEL_ROOT}/GiGPO-Qwen2.5-7B-Instruct-ALFWorld"
PROGRESS_LOG="$EXPERIMENT/logs/t4_pipeline_progress.log"
STAGE=t4_alfworld

cd "$EXPERIMENT"
mkdir -p logs
progress() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$PROGRESS_LOG"; }

cleanup_gate_states() {
  local round path
  for round in r0 r1 r2; do
    path="$EXPERIMENT/artifacts/$STAGE/$round/gate_state"
    if [[ -d "$path" ]]; then
      rm -f -- "$path"/rank*.pt "$path/metadata.json"
      progress "$round gate checkpoint removed; plasticity budget preserved"
    fi
  done
}

checkpoint_ready() {
  local round="$1" export_dir="$2"
  "$PY" - "$EXPERIMENT" "$STAGE" "$round" "$export_dir" <<'PY'
import json, sys
from pathlib import Path
experiment, stage_name, round_name, export_dir = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
config = json.loads((experiment / "config/experiment.json").read_text())
stage = next(item for item in config["stages"] if item["name"] == stage_name)
try:
    summary = json.loads((experiment / "artifacts" / stage_name / round_name / "training_summary.json").read_text())
    index = json.loads((export_dir / "model.safetensors.index.json").read_text())
    weights = sorted(set(index["weight_map"].values()))
    checks = [summary.get("method_version") == config["method_version"],
              summary.get("stage") == stage_name, summary.get("round") == round_name,
              int(summary.get("optimizer_steps", -1)) == int(stage["optimizer_steps"][round_name]),
              (export_dir / "config.json").is_file(), (export_dir / "tokenizer_config.json").is_file(),
              bool(weights), all((export_dir / name).stat().st_size > 0 for name in weights)]
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if all(checks) else 1)
PY
}

prepare_data() {
  [[ -s "$T3FINAL/model.safetensors.index.json" ]] || {
    echo "missing T3 final: $T3FINAL" >&2
    exit 1
  }
  [[ -s "$ALFWORLD_EXPERT/model.safetensors.index.json" ]] || {
    echo "missing ALFWorld expert: $ALFWORLD_EXPERT" >&2
    exit 1
  }
  [[ -s data/splits/webshop_split.json ]] || {
    echo "missing frozen WebShop split" >&2
    exit 1
  }
  [[ -s "${BACAM_DATA_ROOT}/webshop/search_engine_1k/index_manifest.json" ]] || {
    echo "missing WebShop 1k index" >&2
    exit 1
  }
  progress "prepare frozen WebShop, ALFWorld, Tool, and Search pools"
  bash evaluation/start_retriever.sh
}

rollout_round() {
  local round="$1" model="$2" state_dir="data/$STAGE/$1/current_raw/states"
  local alf_count=0 webshop_count=0 search_count=0
  if [[ -s "$state_dir/alfworld_student_states.summary.json" ]]; then
    alf_count="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("episodes", 0))' \
      "$state_dir/alfworld_student_states.summary.json")"
  fi
  if [[ -s "$state_dir/webshop_student_states.summary.json" ]]; then
    webshop_count="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("episodes", 0))' \
      "$state_dir/webshop_student_states.summary.json")"
  fi
  if [[ -s "$state_dir/search_student_states.summary.json" ]]; then
    search_count="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("states", 0))' \
      "$state_dir/search_student_states.summary.json")"
  fi
  if [[ -s "$state_dir/search_student_states.summary.json" && \
        -s "$state_dir/tool_student_states.summary.json" && \
        "$alf_count" -eq 60 && "$webshop_count" -eq 50 && "$search_count" -ge 256 ]]; then
    progress "$round candidate Search/Tool/ALFWorld/WebShop states already present, skipping"
    return
  fi
  progress "$round rollout candidate WebShop, ALFWorld, Tool, and Search"
  bash scripts/rollout/search.sh "$STAGE" "$round" "$model" "$BACAM_GPU_SEARCH" "$BACAM_SEARCH_ROLLOUT_PORT" & search_pid=$!
  bash scripts/rollout/tool.sh "$STAGE" "$round" "$model" "$BACAM_GPU_TOOL" & tool_pid=$!
  bash scripts/rollout/alfworld.sh "$STAGE" "$round" "$model" "$BACAM_GPU_ALFWORLD" "$BACAM_ALFWORLD_ROLLOUT_PORT" & alfworld_pid=$!
  bash scripts/rollout/webshop.sh "$STAGE" "$round" "$model" "$BACAM_GPU_WEBSHOP" "$BACAM_WEBSHOP_ROLLOUT_PORT" & webshop_pid=$!
  wait "$search_pid"
  wait "$tool_pid"
  wait "$alfworld_pid"
  wait "$webshop_pid"
}

run_round() {
  local round="$1" model="$2" export_dir
  case "$round" in
    r0) export_dir="$T4R0" ;;
    r1) export_dir="$T4R1" ;;
    r2) export_dir="$T4R2" ;;
    *) return 2 ;;
  esac
  if [[ "$RESUME" -eq 1 ]] && checkpoint_ready "$round" "$export_dir"; then
    progress "$round resume checkpoint verified, skipping"
    return
  fi
  rollout_round "$round" "$model"
  progress "$round build WebShop/ALFWorld/Tool/Search states"
  "$PY" core/build_states.py --stage "$STAGE" --round "$round"
  progress "$round cache old/new teachers on candidate trajectories"
  bash scripts/cache_teacher.sh "$STAGE" "$round"
  progress "$round train gate on GPUs $BACAM_GPUS"
  CUDA_VISIBLE_DEVICES="$BACAM_GPUS" "$TORCHRUN" --nproc_per_node=4 --master_port="$BACAM_MASTER_PORT" \
    core/train.py --stage "$STAGE" --round "$round"
  progress "$round training complete"
}

prepare_data
run_round r0 "$T3FINAL"
run_round r1 "$T4R0"
run_round r2 "$T4R1"

progress "finalize T4"
if [[ -e "$T4FINAL" && ! -L "$T4FINAL" ]]; then
  echo "refusing to replace $T4FINAL" >&2
  exit 1
fi
[[ -L "$T4FINAL" ]] && rm "$T4FINAL"
ln -s "$T4R2" "$T4FINAL"
"$PY" analysis/summarize_gate_tensors.py --stage "$STAGE"
"$PY" analysis/plot_stage.py --stage "$STAGE"

progress "held-out Search, Tool, ALFWorld, and WebShop evaluation"
"$PY" rollout/prepare_search_pool.py --half eval --overwrite
bash evaluation/run_search_eval.sh t4 "$BACAM_GPU_SEARCH" "$BACAM_SEARCH_EVAL_PORT" & search_pid=$!
bash evaluation/run_bfcl_eval.sh t4 "$BACAM_GPU_TOOL" & tool_pid=$!
EVAL_MODEL_ID=t4 \
EVAL_MODEL_PATH="$T4FINAL" \
EVAL_SERVED_NAME=alfworld-t4 \
ALL_GAMES=1 RESUME=1 RESULT_TAG=vllm \
  bash evaluation/run_alfworld_expert_eval.sh both "$BACAM_GPU_ALFWORLD" "$BACAM_ALFWORLD_EVAL_PORT" & alfworld_pid=$!
EVAL_MODEL_ID=t4 \
EVAL_MODEL_PATH="$T4FINAL" \
EVAL_SERVED_NAME=webshop-t4 \
RESUME=1 RESULT_TAG=vllm \
  bash evaluation/run_webshop_eval.sh "$BACAM_GPU_WEBSHOP" "$BACAM_WEBSHOP_EVAL_PORT" & webshop_pid=$!
wait "$search_pid"
wait "$tool_pid"
wait "$alfworld_pid"
wait "$webshop_pid"
"$PY" evaluation/collect_results.py --model-id t4
"$PY" analysis/plot_stage.py --stage "$STAGE"
"$PY" analysis/plot_capability_trends.py
cleanup_gate_states
progress "T4 ALFWORLD PIPELINE COMPLETE"
