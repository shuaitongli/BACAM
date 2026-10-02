#!/usr/bin/env python3
"""Build one round's 256-state train view and 24-trajectory probe view."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from bacam.data.trajectory_utils import (annotate_trajectory, assert_contiguous,
                              choose_probe_trajectories, choose_states,
                              group_trajectories)  # noqa: E402
from bacam.merging.merge_stage import MergeStage  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_seed(seed: int, stage: str, round_name: str,
                domain: str, source: str) -> int:
    suffix = int(hashlib.sha256(
        f"{stage}:{round_name}:{domain}:{source}".encode()).hexdigest()[:8], 16)
    return seed + suffix


def canonical_optimization_splits(experiment: Path) -> dict[str, dict[str, list[str]]]:
    """Build optimization allowlists from the authoritative frozen split files."""
    split_root = experiment / "data" / "splits"
    search = json.loads((split_root / "search_split.json").read_text())
    bfcl = json.loads((split_root / "bfcl_split.json").read_text())
    alfworld = json.loads((split_root / "alfworld_split.json").read_text())
    webshop = json.loads((split_root / "webshop_split.json").read_text())
    tool_ids = list(bfcl["categories"]["multi_turn_base"]["train"])
    return {
        "search": {"optimization": list(search["train"])},
        "tool": {"optimization": tool_ids},
        "alfworld": {
            "optimization": [
                gamefile
                for task in alfworld["task_types"]
                for gamefile in alfworld["optimization_by_task"][task]
            ]
        },
        "webshop": {
            "optimization": [
                str(item["goal_index"])
                for item in webshop["optimization_goals"]
            ]
        },
    }


def build_stream(stage: MergeStage, domain: str, role: str, source: str,
                 raw_path: Path, output_path: Path, tokenizer,
                 optimization_splits: dict) -> dict:
    if not raw_path.exists():
        raise FileNotFoundError(f"{domain}/{source}: missing {raw_path}")
    rows = read_jsonl(raw_path)
    if domain not in optimization_splits:
        raise RuntimeError(f"{domain}: no frozen optimization split is registered")
    allowed = {str(value) for value in optimization_splits[domain]["optimization"]}
    outside = sorted({str(row.get("episode_id")) for row in rows} - allowed)
    if outside:
        raise RuntimeError(f"{domain}/{source}: states outside optimization split: {outside[:3]}")

    cfg = stage.config["data"]
    max_sequence = int(cfg["max_sequence_tokens"])
    max_response = int(cfg["max_response_tokens"])
    max_states = int(cfg.get("max_states_per_trajectory_by_domain", {}).get(
        domain, cfg["max_states_per_trajectory"]))
    direction = "forward" if role == "new" else "reverse"
    groups = group_trajectories(domain, rows)
    valid: dict[str, list[dict]] = {}
    rejected: list[dict] = []

    for trajectory_id, trajectory in groups.items():
        reason = None
        if len(trajectory) > max_states:
            reason = f"states={len(trajectory)} > {max_states}"
        if domain in {"alfworld", "webshop"} and not bool(
                trajectory[-1].get("trajectory_terminal", False)):
            reason = f"{domain} trajectory is not terminal"
        checked = []
        if reason is None:
            for row in trajectory:
                prompt = str(row.get("prompt", ""))
                response = str(row.get("response", ""))
                prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                response_tokens = len(tokenizer.encode(response, add_special_tokens=False))
                if not prompt.strip() or not response.strip():
                    reason = "empty prompt or response"
                    break
                if response_tokens + 1 > max_response:
                    reason = f"response_tokens_with_eos={response_tokens + 1} > {max_response}"
                    break
                if prompt_tokens + response_tokens + 1 > max_sequence:
                    reason = f"sequence_tokens_with_eos={prompt_tokens + response_tokens + 1} > {max_sequence}"
                    break
                item = dict(row)
                item.update({"episode_id": str(item["episode_id"]),
                             "prompt": prompt, "response": response,
                             "prompt_tokens": prompt_tokens,
                             "response_tokens": response_tokens})
                checked.append(item)
        if reason is not None:
            rejected.append({"trajectory_id": trajectory_id, "reason": reason})
            continue
        annotated = annotate_trajectory(checked, trajectory_id, domain, role, source, direction)
        assert_contiguous(annotated)
        valid[trajectory_id] = annotated

    seed = source_seed(int(stage.config["seed"]), stage.name, stage.round,
                       domain, source)
    state_budget = int(cfg["states_per_stream"])
    probe_budget = int(stage.config["endpoint"]["trajectories_per_source"])
    selected = choose_states(valid, seed, state_budget)
    selected_keys = {(row["trajectory_id"], int(row["trajectory_position"]))
                     for row in selected}
    probe_ids = choose_probe_trajectories(domain, valid, seed, probe_budget)
    probe_set = set(probe_ids)
    output_rows = []
    for trajectory_id in sorted(valid):
        for row in valid[trajectory_id]:
            key = (trajectory_id, int(row["trajectory_position"]))
            if key not in selected_keys and trajectory_id not in probe_set:
                continue
            item = dict(row)
            item["train_selected"] = key in selected_keys
            item["probe_selected"] = trajectory_id in probe_set
            output_rows.append(item)
    write_jsonl(output_path, output_rows)
    return {
        "domain": domain,
        "role": role,
        "source": source,
        "kl_direction": direction,
        "path": str(output_path),
        "raw_path": str(raw_path),
        "states": state_budget,
        "episodes": len({row["episode_id"] for row in selected}),
        "probe_trajectories": probe_budget,
        "probe_trajectory_ids": probe_ids,
        "cached_states": len(output_rows),
        "rejected_trajectories": rejected,
        "max_states_per_trajectory": max((len(value) for value in valid.values()), default=0),
    }


def build_current(stage: MergeStage, tokenizer, optimization_splits: dict) -> dict:
    stage.prepare()
    streams = {}
    raw_root = stage.round_root / "current_raw" / "states"
    for domain in stage.domains:
        role = "old" if domain in stage.old_domains else "new"
        raw = raw_root / f"{domain}_student_states.jsonl"
        output = stage.state_dir / "train" / f"{domain}_current.jsonl"
        streams[f"{domain}_current"] = build_stream(
            stage, domain, role, "current", raw, output, tokenizer,
            optimization_splits)
    manifest = {"stage": stage.name, "round": stage.round,
                "trajectory_policy": "current_candidate",
                "training_unit": "state", "streams": streams}
    target = stage.state_dir / "train_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    experiment = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=experiment)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--round", required=True)
    args = parser.parse_args()
    stage = MergeStage(args.experiment, args.stage, args.round)
    optimization_splits = canonical_optimization_splits(args.experiment)
    tokenizer = AutoTokenizer.from_pretrained(stage.old_model, local_files_only=True,
                                              trust_remote_code=True)
    manifest = build_current(stage, tokenizer, optimization_splits)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
