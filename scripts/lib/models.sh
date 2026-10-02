#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh" || exit 1
MODEL_ROOT="${BACAM_MODEL_ROOT}"
MERGE_ROOT="${BACAM_MERGE_ROOT}"

model_path() {
  case "$1" in
    base)          echo "$MODEL_ROOT/Qwen2.5-7B-Instruct" ;;
    search-expert|t1) echo "$MODEL_ROOT/ReSearch-Qwen-7B-Instruct" ;;
    tool-expert)   echo "$MODEL_ROOT/Qwen2.5-7B-Instruct-ToolRL-grpo-cold" ;;
    alfworld-expert) echo "$MODEL_ROOT/GiGPO-Qwen2.5-7B-Instruct-ALFWorld" ;;
    webshop-expert) echo "$MODEL_ROOT/GiGPO-Qwen2.5-7B-Instruct-WebShop" ;;
    t2)            echo "$MERGE_ROOT/bacam-wtsa-t2-final" ;;
    t3)            echo "$MERGE_ROOT/bacam-wtsa-t3-final" ;;
    t4)            echo "$MERGE_ROOT/bacam-wtsa-t4-final" ;;
    *) echo "unknown model id: $1" >&2; return 1 ;;
  esac
}

tool_tag() {
  echo "bacam-wtsa-$1-ours"
}

search_tag() {
  echo "bacam-wtsa-$1-ours"
}
