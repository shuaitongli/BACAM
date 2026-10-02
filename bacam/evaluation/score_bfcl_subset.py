#!/usr/bin/env python3
"""Score a BFCL result directory against the preregistered held-out IDs.

`bfcl evaluate` cannot do this: it asserts the result file covers the whole category
(200 items), so a 400-item held-out run fails the length check.  This calls the same
`multi_turn_runner` with the prompt and answer lists filtered to the held-out IDs, so
the scoring logic is BFCL's own -- only the item set is narrowed.  Every score in
RESULTS.md, references included, comes from this path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from bacam.paths import PATHS

BFCL_ROOT = Path(PATHS["BACAM_BFCL_ROOT"])
sys.path.insert(0, str(BFCL_ROOT))

from bfcl_eval.eval_checker.eval_runner import multi_turn_runner  # noqa: E402
from bfcl_eval.model_handler.local_inference.rlla import RLLAHandler  # noqa: E402
from bfcl_eval.utils import load_file  # noqa: E402


BFCL_CATEGORIES = ("multi_turn_base",)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    experiment = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads((experiment / "data/splits/bfcl_split.json").read_text())
    handler = RLLAHandler(args.tag, temperature=0)
    metrics = {}
    for category in BFCL_CATEGORIES:
        ids = split["categories"][category]
        wanted = set(ids["eval"])
        filename = f"BFCL_v3_{category}_result.json"
        result = sorted(
            (row for row in read_jsonl(args.result_root / args.tag / filename) if row["id"] in wanted),
            key=lambda row: row["id"],
        )
        prompt_all = load_file(BFCL_ROOT / "bfcl_eval/data" / f"BFCL_v3_{category}.json")
        answer_all = load_file(BFCL_ROOT / "bfcl_eval/data/possible_answer" / f"BFCL_v3_{category}.json")
        prompt_map = {row["id"]: row for row in prompt_all}
        answer_map = {row["id"]: row for row in answer_all}
        ordered_ids = [row["id"] for row in result]
        if set(ordered_ids) != wanted:
            raise RuntimeError(f"{args.tag}/{category}: result IDs do not match held-out split")
        prompt = [prompt_map[item] for item in ordered_ids]
        answers = [answer_map[item] for item in ordered_ids]
        accuracy, total = multi_turn_runner(
            handler,
            result,
            prompt,
            answers,
            args.tag,
            category,
            args.score_root,
        )
        metrics[category] = {"accuracy": accuracy, "total": total}

    macro = sum(item["accuracy"] for item in metrics.values()) / len(metrics)
    summary = {"tag": args.tag, "macro_accuracy": macro, "categories": metrics}
    summary_dir = args.score_root / args.tag
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
