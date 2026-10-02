#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/paths.sh" || exit 1
set -euo pipefail

MODEL="${MODEL:?MODEL is required}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL")}"
unset http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES="${SGL_GPUS:-$BACAM_GPU_SEARCH}"
if [[ "$BACAM_SGLANG_PYTHON" == */* ]]; then
  sgl_bin="$(dirname "$BACAM_SGLANG_PYTHON")"
  export PATH="$sgl_bin:$PATH"
  if [[ -x "$sgl_bin/nvcc" && -z "${CUDA_HOME:-}" ]]; then
    export CUDA_HOME="$(dirname "$sgl_bin")"
  fi
fi
args=(
  --served-model-name "$SERVED_NAME" --model-path "$MODEL"
  --tp-size "${SGL_TP:-1}" --context-length "${SGL_CTX:-8192}" --dtype bfloat16
  --mem-fraction-static "${MEM_FRAC:-0.80}"
  --host 0.0.0.0 --port "${SGL_PORT:-$BACAM_SEARCH_ROLLOUT_PORT}"
  --enable-metrics --trust-remote-code
)
[[ "${SGL_DISABLE_OVERLAP:-0}" != 1 ]] || args+=(--disable-overlap-schedule)
[[ "${SGL_DISABLE_RADIX:-0}" != 1 ]] || args+=(--disable-radix-cache)
if [[ -n "${SGL_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "$SGL_EXTRA_ARGS"
  args+=("${extra_args[@]}")
fi
exec "$BACAM_SGLANG_PYTHON" -m sglang.launch_server "${args[@]}"
