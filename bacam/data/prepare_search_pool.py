#!/usr/bin/env python3
"""Materialise a preregistered Musique half as a flashrag dataset directory.

flashrag reads `<data_dir>/<dataset_name>/<split>.jsonl`, so restricting a run to the
preregistered training half means writing that half out as its own dataset.  This adds
a directory, it never edits the source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bacam.paths import PATHS

FLASHRAG_DATA = Path(PATHS["BACAM_DATA_ROOT"]) / "flashrag_eval"


def main() -> None:
    experiment = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=experiment)
    parser.add_argument("--source", type=Path, default=FLASHRAG_DATA / "musique/dev.jsonl")
    parser.add_argument("--half", choices=("train", "eval"), default="train")
    parser.add_argument("--name", default=None)
    parser.add_argument("--selection-key", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split = json.loads((args.experiment / "data/splits/search_split.json").read_text())
    wanted = list(split[args.half])
    if args.half == "train":
        config = json.loads((args.experiment / "configs/experiment.json").read_text())
        wanted.sort(key=lambda item: hashlib.sha256(
            f'{config["seed"]}:{args.selection_key}:search:{item}'.encode()).hexdigest())
        wanted = wanted[:int(config["data"]["rollout_episodes"]["search"])]
    forbidden = set(split["train" if args.half == "eval" else "eval"])
    args.name = args.name or f"musique_bacam_wtsa_{args.half}"

    rows = {json.loads(line)["id"]: line.rstrip("\n")
            for line in args.source.read_text().splitlines() if line.strip()}
    missing = set(wanted) - rows.keys()
    if missing:
        raise RuntimeError(f"{len(missing)} questions of half {args.half} are absent from {args.source}")

    target_dir = FLASHRAG_DATA / args.name
    target = target_dir / "dev.jsonl"
    if target.exists() and not args.overwrite:
        raise RuntimeError(f"{target} already exists; pass --overwrite to replace it")
    target_dir.mkdir(parents=True, exist_ok=True)
    kept = [rows[item_id] for item_id in wanted]
    if any(json.loads(line)["id"] in forbidden for line in kept):
        raise RuntimeError(f"the other half leaked into the {args.half} dataset")
    target.write_text("\n".join(kept) + "\n", encoding="utf-8")

    manifest = {
        "dataset_name": args.name,
        "split": "dev",
        "half": args.half,
        "path": str(target),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "questions": len(kept),
        "source": split["source"],
        "usage": f"run_eval.py --data_dir {FLASHRAG_DATA} --dataset_name {args.name} --split dev",
    }
    manifest_name = f"search_{'pool' if args.half == 'train' else 'holdout'}_dataset.json"
    (args.experiment / "data/splits" / manifest_name).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
