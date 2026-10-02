#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/paths.sh" || exit 1
set -euo pipefail

PATCH="$BACAM_ROOT/integrations/research/resume.patch"
[[ -f "$BACAM_RESEARCH_ROOT/src/flashrag/pipeline/active_pipeline.py" ]] || {
  echo "Install the ReSearch re-search branch at $BACAM_RESEARCH_ROOT first" >&2; exit 1;
}
[[ -f "$BACAM_BFCL_ROOT/bfcl_eval/constants/model_config.py" ]] || {
  echo "Install the BFCL source distribution at $BACAM_BFCL_ROOT first" >&2; exit 1;
}

if ! git -C "$BACAM_RESEARCH_ROOT" apply --reverse --check "$PATCH" 2>/dev/null; then
  git -C "$BACAM_RESEARCH_ROOT" apply --check "$PATCH"
  git -C "$BACAM_RESEARCH_ROOT" apply "$PATCH"
fi
install -m 644 "$BACAM_ROOT/integrations/research/retriever_serving.py" \
  "$BACAM_RESEARCH_ROOT/scripts/serving/retriever_serving.py"
install -m 644 "$BACAM_ROOT/integrations/tool/rlla.py" \
  "$BACAM_BFCL_ROOT/bfcl_eval/model_handler/local_inference/rlla.py"
"$BACAM_PYTHON" -m bacam.evaluation.register_bfcl_models
echo "ReSearch and Tool adapters installed"
