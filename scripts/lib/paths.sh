#!/usr/bin/env bash
# Source shared defaults; overrides are exported BACAM_* variables.
_bacam_paths_script="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../bacam" && pwd)/paths.py"
_bacam_path_exports="$("${BACAM_PYTHON:-python}" "$_bacam_paths_script" --shell)" || return 1
eval "$_bacam_path_exports"
export PYTHONPATH="$BACAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset _bacam_paths_script _bacam_path_exports

check_gpu() {
  [[ "$1" =~ ^[0-9]+$ && ",$BACAM_GPUS," == *",$1,"* ]] || {
    echo "GPU must be one of $BACAM_GPUS" >&2; return 2;
  }
}
