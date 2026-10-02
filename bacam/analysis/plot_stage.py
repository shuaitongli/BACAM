#!/usr/bin/env python3
"""Plot training curves, gate distributions, and held-out results for one stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DOMAIN_COLORS = {
    "search": "tab:blue",
    "tool": "tab:orange",
    "alfworld": "tab:green",
    "webshop": "tab:red",
}
PASS_LINE = 0.9


def domain_color(name: str) -> str:
    """Return one stable color for a domain across all stages and subplots."""
    if name in DOMAIN_COLORS:
        return DOMAIN_COLORS[name]
    domain = name.rsplit("_", 1)[0] if "_" in name else name
    return DOMAIN_COLORS.get(domain, "tab:gray")


def plot_gate_histograms(experiment: Path, stage_name: str, rounds: list[str], target: Path,
                         gate_range: tuple[float, float]) -> None:
    """Plot round distributions together and adjacent-round distribution changes."""
    summaries = []
    for round_name in rounds:
        path = experiment / f"artifacts/{stage_name}/{round_name}/training_summary.json"
        if not path.exists():
            continue
        try:
            summary = json.loads(path.read_text())
            histogram = [int(value) for value in summary["gate_histogram"]]
            if len(histogram) != 10 or not sum(histogram):
                continue
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        summaries.append((round_name, summary, histogram))

    if not summaries:
        return

    lo, hi = gate_range
    width = (hi - lo) / 10.0
    labels = [f"{lo + i * width:.1f}\u2013{lo + (i + 1) * width:.1f}"
              for i in range(10)]
    percentages_by_round = []
    for round_name, summary, histogram in summaries:
        total = sum(histogram)
        percentages = [100.0 * value / total for value in histogram]
        percentages_by_round.append((round_name, summary, total, percentages))

    figure, axis = plt.subplots(figsize=(14, 5.8), constrained_layout=True)
    positions = list(range(10))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    bar_width = 0.78 / len(percentages_by_round)
    for index, (round_name, summary, total, percentages) in enumerate(percentages_by_round):
        offset = (index - (len(percentages_by_round) - 1) / 2) * bar_width
        bars = axis.bar([position + offset for position in positions], percentages,
                        width=bar_width, color=colors[index % len(colors)],
                        label=f"{round_name} (n={total:,})")
        for bar, percentage in zip(bars, percentages):
            if percentage >= 1.0:
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35,
                          f"{percentage:.1f}%", ha="center", va="bottom", fontsize=7)
    axis.set_title(f"BACAM {stage_name}: final gate distribution by round", pad=12)
    axis.set_ylabel("free-coordinate share (%)")
    axis.set_xlabel("final gate interval")
    axis.set_xticks(positions, labels, rotation=45)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8, ncol=len(percentages_by_round))
    figure.savefig(target / "fig_gate_histogram.png", dpi=180)
    figure.savefig(target / "fig_gate_histogram.pdf")
    plt.close(figure)


def main() -> None:
    experiment = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=experiment)
    parser.add_argument("--stage", choices=("t2_tool", "t3_search", "t4_alfworld"), required=True)
    args = parser.parse_args()
    config = json.loads((args.experiment / "configs/experiment.json").read_text())
    stage = next(item for item in config["stages"] if item["name"] == args.stage)
    rounds = config["optimization"]["rounds"]
    target = args.experiment / "artifacts" / args.stage
    target.mkdir(parents=True, exist_ok=True)

    plot_gate_histograms(
        args.experiment,
        args.stage,
        rounds,
        target,
        tuple(float(value) for value in config["optimization"]["gate_range"]),
    )

    rows, offset, boundaries = [], 0, []
    for round_name in rounds:
        path = args.experiment / f"logs/{args.stage}/{round_name}/train.jsonl"
        if not path.exists():
            continue
        current = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in current:
            row["global_step"] = offset + int(row["optimizer_step"])
            rows.append(row)
        offset += len(current)
        boundaries.append(offset)
    if rows:
        steps = [row["global_step"] for row in rows]
        first = rows[0]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for stream in sorted(first["raw_kl"]):
            axes[0, 0].plot(steps, [row["raw_kl"][stream] for row in rows],
                             color=domain_color(stream), label=stream)
        axes[0, 0].set(title="Trajectory KL (raw)", yscale="log", ylabel="KL")
        new_streams = [name for name in first["raw_kl"]
                       if name.rsplit("_", 1)[0] in stage["new_domains"]]
        for stream in new_streams:
            axes[0, 1].plot(steps, [row["raw_kl"][stream] for row in rows],
                             color=domain_color(stream), label=stream)
        axes[0, 1].set(title="New-task behavior KL (lower is better)", ylabel="raw_kl")
        old_streams = [name for name in first["old_drift"]
                       if name.rsplit("_", 1)[0] in stage["old_domains"]]
        for stream in old_streams:
            axes[1, 0].plot(steps, [row["old_drift"][stream] for row in rows],
                             color=domain_color(stream), label=stream)
        budget = first["raw_kl_budget"]
        axes[1, 0].axhline(
            budget, color="black", linestyle="--", label=f"epsilon={budget:g}")
        axes[1, 0].set(title="Old-task raw KL drift", ylabel="raw_kl - old_endpoint")
        for stream in sorted(first["dual"]):
            axes[1, 1].plot(steps, [row["dual"][stream] for row in rows],
                             color=domain_color(stream), label=stream)
        axes[1, 1].set(title="Constraint multipliers", ylabel="dual")
        for axis in axes.flat:
            for boundary in boundaries[:-1]:
                axis.axvline(boundary + 0.5, color="gray", alpha=0.4)
            axis.set_xlabel("global optimizer step")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        fig.suptitle(f"BACAM {args.stage}: raw-KL constrained training")
        fig.savefig(target / "fig_training.png", dpi=180)
        fig.savefig(target / "fig_training.pdf")

    model_id = {"t2_tool": "t2", "t3_search": "t3", "t4_alfworld": "t4"}[args.stage]
    heldout_path = args.experiment / f"artifacts/eval/results_{model_id}.json"
    if heldout_path.exists():
        result = json.loads(heldout_path.read_text())
        names = list(result["gates"])
        references = json.loads((args.experiment / "configs/expert_references.json").read_text())["references"]
        expert = {
            "search": references["search"]["expert"],
            "tool": references["tool"]["expert"],
            "alfworld_iid": references["alfworld"]["expert_iid"],
            "alfworld_ood": references["alfworld"]["expert_ood"],
            "webshop": references["webshop"]["expert"],
        }
        ratios = [result["gates"][name]["measured"] / expert[name]
                  for name in names]
        passed = [ratio >= PASS_LINE for ratio in ratios]
        fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        bars = axis.bar(names, ratios, color=[domain_color(name) for name in names],
                        edgecolor=["black" if ok else "#d62728" for ok in passed],
                        linewidth=[1.0 if ok else 2.0 for ok in passed],
                        hatch=["" if ok else "//" for ok in passed])
        axis.axhline(1.0, color="black", linewidth=1.2, label="expert = 1.0")
        axis.axhline(PASS_LINE, color="black", linestyle="--", label="pass line = 0.9")
        for bar, ratio in zip(bars, ratios):
            axis.text(bar.get_x() + bar.get_width() / 2, ratio + 0.02,
                      f"{ratio:.1%} of expert", ha="center")
        axis.set(title=f"BACAM {args.stage}: held-out decision", ylabel="measured / expert")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
        fig.savefig(target / "fig_heldout.png", dpi=180)
        fig.savefig(target / "fig_heldout.pdf")
    print(target)


if __name__ == "__main__":
    main()
