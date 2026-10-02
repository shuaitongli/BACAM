#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/paths.sh" || exit 1
# Run WTSA stages in order; each pipeline owns its checkpoints.
set -euo pipefail

RESUME=0
if [[ "${1:-}" == "--resume" ]]; then RESUME=1; shift; fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: bash scripts/run_all.sh [--resume]" >&2
  exit 2
fi

EXPERIMENT="$BACAM_ROOT"
PY="$BACAM_PYTHON"
LOG="$EXPERIMENT/logs/run_all_progress.log"
cd "$EXPERIMENT"
mkdir -p logs
progress() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

run_stage() {
  local model_id="$1" script="$2"
  progress "$model_id starting"
  if [[ "$RESUME" -eq 1 ]]; then
    bash "$script" --resume
  else
    bash "$script"
  fi
  progress "$model_id complete"
}

progress "WTSA preflight"
for model in \
  "${BACAM_MODEL_ROOT}/ReSearch-Qwen-7B-Instruct" \
  "${BACAM_MODEL_ROOT}/Qwen2.5-7B-Instruct-ToolRL-grpo-cold" \
  "${BACAM_MODEL_ROOT}/GiGPO-Qwen2.5-7B-Instruct-ALFWorld" \
  "${BACAM_MODEL_ROOT}/GiGPO-Qwen2.5-7B-Instruct-WebShop"; do
  [[ -d "$model" ]] || { echo "missing expert model: $model" >&2; exit 1; }
done

run_stage t2 scripts/pipelines/merge_tool.sh
run_stage t3 scripts/pipelines/merge_search.sh
run_stage t4 scripts/pipelines/merge_alfworld.sh
progress "capability trends plotting"
"$PY" -m bacam.analysis.plot_capability_trends
progress "capability trends plotting complete"
progress "WTSA ALL PIPELINES COMPLETE"
