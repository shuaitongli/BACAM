#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/paths.sh" || exit 1
# Score one merged model on the preregistered BFCL held-out half.
#   bash evaluation/run_bfcl_eval.sh <model_id> [gpu]
#
# Runs official `bfcl generate` on the held-out multi_turn_base IDs and writes a
# completion marker.
set -euo pipefail

MID="${1:?usage: run_bfcl_eval.sh <model_id> [gpu]}"
GPU="${2:-$BACAM_GPU_TOOL}"
check_gpu "$GPU" || exit 2

BFCL_ROOT="$BACAM_BFCL_ROOT"
EXPERIMENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL="$EXPERIMENT/evaluation"
LOGS="$EXPERIMENT/logs/eval"
RESULT_DIR="$EXPERIMENT/artifacts/eval/bfcl/result"
SCORE_DIR="$EXPERIMENT/artifacts/eval/bfcl/score"
DONE="$EXPERIMENT/artifacts/eval/.done"
IDS_FILE="$BFCL_ROOT/test_case_ids_to_generate.json"
CATS=multi_turn_base
mkdir -p "$LOGS" "$RESULT_DIR" "$SCORE_DIR" "$DONE"

source "$EVAL/models.sh"
unset http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"

TAG="$(tool_tag "$MID")"
MODEL="$(model_path "$MID")"
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOGS/bfcl.log"; }

[ -d "$MODEL" ] || { say "$MODEL 不存在"; exit 1; }
MODEL_STAMP="$MODEL/model.safetensors.index.json"
[ -s "$MODEL_STAMP" ] || { say "$MODEL_STAMP 不存在"; exit 1; }
[ -f "$DONE/bfcl-$MID" ] && [ "$DONE/bfcl-$MID" -nt "$MODEL_STAMP" ] && {
  say "$TAG 已完成且结果比模型新，跳过"; exit 0;
}
rm -f "$DONE/bfcl-$MID"

"$BACAM_TOOL_PYTHON" "$EVAL/register_bfcl_models.py" > "$LOGS/register_bfcl.log" 2>&1

# 评测集 = 留出半边。--run-ids 完全替换 --test-category，读这一个全局文件。
"$BACAM_TOOL_PYTHON" -c "
import json, sys
split = json.load(open(sys.argv[1]))
ids = split['categories']['multi_turn_base']['eval']
json.dump({'multi_turn_base': ids}, open(sys.argv[2], 'w'), indent=2)
print('held-out ids:', len(ids))
" "$EXPERIMENT/data/splits/bfcl_split.json" "$IDS_FILE" | tee -a "$LOGS/bfcl.log"

rm -rf "${RESULT_DIR:?}/$TAG"
say "$TAG -> GPU $GPU, model $MODEL"
CUDA_VISIBLE_DEVICES=$GPU VLLM_PORT="$BACAM_TOOL_PORT" \
  "$BACAM_BFCL" generate --model "$TAG" --local-model-path "$MODEL" \
    --test-category "$CATS" --run-ids \
    --num-gpus 1 --gpu-memory-utilization 0.85 \
    --result-dir "$RESULT_DIR" --allow-overwrite \
    > "$LOGS/bfcl-generate-$MID.log" 2>&1

# bfcl evaluate 断言结果覆盖整个类别(200 条),留出集(100 条)过不了长度检查,
# 所以用 score_bfcl_subset.py 调同一个 multi_turn_runner,只把题目集合缩到留出 ID。
"$BACAM_TOOL_PYTHON" "$EVAL/score_bfcl_subset.py" --tag "$TAG" \
  --result-root "$RESULT_DIR" --score-root "$SCORE_DIR" \
  > "$LOGS/bfcl-evaluate-$MID.log" 2>&1

count=$(find "$SCORE_DIR/$TAG" -maxdepth 1 -name 'BFCL_v3_multi_turn_base_score.json' | wc -l)
[ "$count" -eq 1 ] || { say "$TAG 只有 $count/1 类"; exit 1; }
touch "$DONE/bfcl-$MID"
say "$TAG 完成"
