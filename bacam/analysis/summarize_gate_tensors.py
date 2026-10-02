#!/usr/bin/env python3
"""Summarize per-tensor gate changes and plasticity budgets before checkpoint cleanup."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


ROUNDS = ("r0", "r1", "r2")
CHUNK = 1_000_000


def gate_stats(current: torch.Tensor, previous: torch.Tensor | None) -> dict:
    if previous is not None and current.shape != previous.shape:
        raise ValueError("gate shard shape changed between rounds")
    count = changed = 0
    absolute = signed = gate_sum = 0.0
    for offset in range(0, current.numel(), CHUNK):
        now = current.flatten()[offset:offset + CHUNK]
        before = previous.flatten()[offset:offset + CHUNK] if previous is not None else 0
        delta = now - before
        count += now.numel()
        changed += int(torch.count_nonzero(delta))
        absolute += float(delta.abs().sum(dtype=torch.float64))
        signed += float(delta.sum(dtype=torch.float64))
        gate_sum += float(now.sum(dtype=torch.float64))
    return {"numel": count, "changed": changed, "absolute": absolute,
            "signed": signed, "gate_sum": gate_sum}


def write_budget_only(stage_dir: Path) -> None:
    budgets = {round_name: json.loads((stage_dir / round_name / "gate_state" /
                                      "plasticity_budget.json").read_text())["tensors"]
               for round_name in ROUNDS}
    fields = ["tensor", "min_b_h"]
    for round_name in ROUNDS:
        fields += [f"{round_name}_b_h", f"{round_name}_plasticity_gradient_rms",
                   f"{round_name}_max_old_conflict"]
    rows = []
    for name in sorted(set().union(*(items.keys() for items in budgets.values()))):
        row = {"tensor": name, "min_b_h": min(
            budgets[r].get(name, {}).get("plasticity_budget", 1.0) for r in ROUNDS)}
        for round_name in ROUNDS:
            budget = budgets[round_name].get(name, {})
            row[f"{round_name}_b_h"] = budget.get("plasticity_budget")
            row[f"{round_name}_plasticity_gradient_rms"] = budget.get("plasticity_rms")
            row[f"{round_name}_max_old_conflict"] = max(
                (old["conflict"] for old in budget.get("old_streams", {}).values()), default=None)
        rows.append(row)
    target = stage_dir / "tensor_budget_diagnostics.csv"
    with target.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{target}: {len(rows)} tensors; historical gate checkpoints unavailable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parents[2]
    stage_dir = experiment / "artifacts" / args.stage
    output = stage_dir / "tensor_gate_diagnostics.csv"
    roots = [stage_dir / round_name / "gate_state" for round_name in ROUNDS]
    if output.is_file() and all(not (root / "metadata.json").exists() for root in roots):
        print(f"{output}: existing report retained (gate checkpoints already cleaned)")
        return
    if all(not (root / "metadata.json").exists() for root in roots):
        write_budget_only(stage_dir)
        return
    budgets = {}
    checkpoints = {}
    for round_name in ROUNDS:
        root = stage_dir / round_name / "gate_state"
        metadata = json.loads((root / "metadata.json").read_text())
        if metadata["stage"] != args.stage or metadata["round"] != round_name:
            raise ValueError(f"wrong gate checkpoint: {root}")
        checkpoints[round_name] = [root / f"rank{rank:02d}.pt"
                                   for rank in range(int(metadata["world_size"]))]
        budgets[round_name] = json.loads((root / "plasticity_budget.json").read_text())["tensors"]
    if len({len(paths) for paths in checkpoints.values()}) != 1:
        raise ValueError("gate checkpoint world size changed between rounds")

    moments: dict[str, dict[str, dict]] = {}
    for rank in range(len(checkpoints["r0"])):
        previous = None
        for round_name in ROUNDS:
            current = torch.load(checkpoints[round_name][rank], map_location="cpu",
                                 weights_only=True, mmap=True)
            if previous is not None and current.keys() != previous.keys():
                raise ValueError("gate tensor names changed between rounds")
            for name, gate in current.items():
                values = gate_stats(gate, previous[name] if previous is not None else None)
                total = moments.setdefault(name, {}).setdefault(
                    round_name, {key: 0 for key in values})
                for key, value in values.items():
                    total[key] += value
            previous = current

    fields = ["tensor", "numel", "total_mean_abs_net_delta_g", "final_mean_g",
              "min_plasticity_budget"]
    for round_name in ROUNDS:
        fields += [f"{round_name}_mean_abs_net_delta_g",
                   f"{round_name}_mean_abs_net_delta_g_changed",
                   f"{round_name}_mean_signed_delta_g",
                   f"{round_name}_changed_fraction",
                   f"{round_name}_mean_g",
                   f"{round_name}_plasticity_budget",
                   f"{round_name}_plasticity_gradient_rms",
                   f"{round_name}_max_old_conflict"]
    rows = []
    for name, by_round in moments.items():
        numel = by_round["r2"]["numel"]
        if not numel:
            continue
        row = {"tensor": name, "numel": numel,
               "total_mean_abs_net_delta_g": sum(by_round[r]["absolute"] for r in ROUNDS) / numel,
               "final_mean_g": by_round["r2"]["gate_sum"] / numel,
               "min_plasticity_budget": min(
                   budgets[r].get(name, {}).get("plasticity_budget", 1.0) for r in ROUNDS)}
        for round_name in ROUNDS:
            stats = by_round[round_name]
            budget = budgets[round_name].get(name, {})
            prefix = f"{round_name}_"
            row[prefix + "mean_abs_net_delta_g"] = stats["absolute"] / numel
            row[prefix + "mean_abs_net_delta_g_changed"] = (
                stats["absolute"] / stats["changed"] if stats["changed"] else 0.0)
            row[prefix + "mean_signed_delta_g"] = stats["signed"] / numel
            row[prefix + "changed_fraction"] = stats["changed"] / numel
            row[prefix + "mean_g"] = stats["gate_sum"] / numel
            row[prefix + "plasticity_budget"] = budget.get("plasticity_budget")
            row[prefix + "plasticity_gradient_rms"] = budget.get("plasticity_rms")
            row[prefix + "max_old_conflict"] = max(
                (old["conflict"] for old in budget.get("old_streams", {}).values()), default=None)
        rows.append(row)
    rows.sort(key=lambda row: row["total_mean_abs_net_delta_g"], reverse=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{output}: {len(rows)} tensors")


if __name__ == "__main__":
    main()
