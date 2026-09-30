#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/paths.sh" || exit 1
# T2: WebShop -> Tool.
set -euo pipefail

RESUME=0
if [[ "${1:-}" == "--resume" ]]; then RESUME=1; shift; fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: bash scripts/pipelines/merge_tool.sh [--resume]" >&2
  exit 2
fi

EXPERIMENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$BACAM_PYTHON"
TORCHRUN="$BACAM_TORCHRUN"
WEBSHOP="${BACAM_MODEL_ROOT}/GiGPO-Qwen2.5-7B-Instruct-WebShop"
TOOL="${BACAM_MODEL_ROOT}/Qwen2.5-7B-Instruct-ToolRL-grpo-cold"
T2R0="${BACAM_MERGE_ROOT}/bacam-wtsa-t2-r0"
T2R1="${BACAM_MERGE_ROOT}/bacam-wtsa-t2-r1"
T2R2="${BACAM_MERGE_ROOT}/bacam-wtsa-t2-r2"
T2FINAL="${BACAM_MERGE_ROOT}/bacam-wtsa-t2-final"
PROGRESS_LOG="$EXPERIMENT/logs/t2_pipeline_progress.log"
STAGE=t2_tool

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
  for model in "$WEBSHOP" "$TOOL"; do
    [[ -s "$model/config.json" ]] || { echo "missing model: $model" >&2; exit 1; }
  done
  [[ -s data/splits/webshop_split.json ]] || { echo "missing frozen WebShop split" >&2; exit 1; }
  [[ -s "${BACAM_DATA_ROOT}/webshop/search_engine_1k/index_manifest.json" ]] || {
    echo "missing WebShop 1k index" >&2; exit 1;
  }
  progress "prepare frozen WebShop and Tool optimization pools"
}

rollout_round() {
  local round="$1" model="$2" state_dir="data/$STAGE/$1/current_raw/states"
  local webshop_count=0
  if [[ -s "$state_dir/webshop_student_states.summary.json" ]]; then
    webshop_count="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("episodes", 0))' \
      "$state_dir/webshop_student_states.summary.json")"
  fi
  if [[ -s "$state_dir/tool_student_states.summary.json" && "$webshop_count" -eq 50 ]]; then
    progress "$round candidate WebShop/Tool rollouts already present, skipping"
    return
  fi
  progress "$round rollout candidate WebShop and Tool"
  bash scripts/rollout/webshop.sh "$STAGE" "$round" "$model" "$BACAM_GPU_WEBSHOP" "$BACAM_WEBSHOP_ROLLOUT_PORT" & webshop_pid=$!
  bash scripts/rollout/tool.sh "$STAGE" "$round" "$model" "$BACAM_GPU_TOOL" & tool_pid=$!
  wait "$webshop_pid"
  wait "$tool_pid"
}

run_round() {
  local round="$1" model="$2" export_dir
  case "$round" in r0) export_dir="$T2R0";; r1) export_dir="$T2R1";; r2) export_dir="$T2R2";; *) return 2;; esac
  if [[ "$RESUME" -eq 1 ]] && checkpoint_ready "$round" "$export_dir"; then
    progress "$round resume checkpoint verified, skipping"
    return
  fi
  rollout_round "$round" "$model"
  progress "$round build WebShop/Tool states"
  "$PY" core/build_states.py --stage "$STAGE" --round "$round"
  progress "$round cache old/new teachers on candidate WebShop/Tool trajectories"
  bash scripts/cache_teacher.sh "$STAGE" "$round"
  progress "$round train gate on GPUs $BACAM_GPUS"
  CUDA_VISIBLE_DEVICES="$BACAM_GPUS" "$TORCHRUN" --nproc_per_node=4 --master_port="$BACAM_MASTER_PORT" core/train.py --stage "$STAGE" --round "$round"
  progress "$round training complete"
}

prepare_data
run_round r0 "$WEBSHOP"
run_round r1 "$T2R0"
run_round r2 "$T2R1"

progress "finalize T2"
if [[ -e "$T2FINAL" && ! -L "$T2FINAL" ]]; then
  echo "refusing to replace $T2FINAL" >&2
  exit 1
fi
[[ -L "$T2FINAL" ]] && rm "$T2FINAL"
ln -s "$T2R2" "$T2FINAL"
"$PY" analysis/summarize_gate_tensors.py --stage "$STAGE"
"$PY" analysis/plot_stage.py --stage "$STAGE"

progress "held-out WebShop and Tool evaluation"
bash evaluation/run_bfcl_eval.sh t2 "$BACAM_GPU_TOOL" & tool_pid=$!
EVAL_MODEL_ID=t2 EVAL_MODEL_PATH="$T2FINAL" EVAL_SERVED_NAME=webshop-t2 \
RESUME=1 RESULT_TAG=vllm bash evaluation/run_webshop_eval.sh "$BACAM_GPU_WEBSHOP" "$BACAM_WEBSHOP_EVAL_PORT" & webshop_pid=$!
wait "$tool_pid"
wait "$webshop_pid"
"$PY" evaluation/collect_results.py --model-id t2
"$PY" analysis/plot_stage.py --stage "$STAGE"
cleanup_gate_states
progress "T2 TOOL PIPELINE COMPLETE"
