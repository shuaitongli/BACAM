#!/usr/bin/env python3
"""Plot task performance as a matrix across successive merging stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


METRICS = {
    "webshop": (("webshop", "WebShop TS", "expert"),),
    "tool": (("tool", "Tool SR", "expert"),),
    "search": (("search", "Search Sub-EM", "expert"),),
    "alfworld": (
        ("alfworld_iid", "ALFWorld IID SR", "expert_iid"),
        ("alfworld_ood", "ALFWorld OOD SR", "expert_ood"),
    ),
}
SHORT = {"webshop": "W", "tool": "T", "search": "S", "alfworld": "A"}
DISPLAY = {"webshop": "WebShop", "tool": "Tool",
           "search": "Search", "alfworld": "ALFWorld"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def stage_data(experiment: Path) -> tuple[str, list[str], list[str], np.ndarray, np.ndarray]:
    config = read_json(experiment / "configs/experiment.json")
    stages = config["stages"]
    order = [stages[0]["old_domains"][0]] + [stage["new_domains"][0] for stage in stages]
    if len(order) != 4 or len(set(order)) != 4:
        raise ValueError(f"expected four distinct domains, got {order}")
    references = read_json(experiment / "configs/expert_references.json")["references"]
    results = [read_json(experiment / f"artifacts/eval/results_t{i}.json")
               for i in (2, 3, 4)]

    rows = [(domain, *metric) for domain in order for metric in METRICS[domain]]
    labels = [label for _, _, label, _ in rows]
    expert = np.array([references[domain][reference]
                       for domain, _, _, reference in rows], dtype=float)
    scores = np.full((len(rows), len(order)), np.nan)
    for row, (domain, key, _, _) in enumerate(rows):
        if domain == order[0]:
            scores[row, 0] = expert[row]
        for column, result in enumerate(results, start=1):
            if domain in order[:column + 1]:
                scores[row, column] = float(result["gates"][key]["measured"])
    return "".join(SHORT[domain] for domain in order), order, labels, scores, expert


def main() -> None:
    experiment = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=experiment)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    order_name, order, labels, scores, expert = stage_data(args.experiment)
    output = args.output or (args.experiment / "artifacts/eval" /
                             f"fig_performance_across_merging_stages_{order_name}.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    ratios = scores / expert[:, None]
    cmap = plt.colormaps["Blues"].copy()
    cmap.set_bad("#f2f4f6")
    norm = Normalize(vmin=0.0, vmax=1.3)
    figure, axis = plt.subplots(figsize=(8.8, 4.1))
    figure.subplots_adjust(left=0.22, right=0.88, bottom=0.08, top=0.73)
    image = axis.imshow(np.ma.masked_invalid(ratios), cmap=cmap, norm=norm,
                        aspect="auto")
    for row in range(len(labels)):
        axis.add_patch(Rectangle((3.5, row - 0.5), 1, 1,
                                 facecolor="#e8edf2", edgecolor="none"))
        axis.text(4, row, f"{expert[row] * 100:.2f}", ha="center", va="center",
                  color="#202631", fontsize=11, fontweight="semibold")
        for column in range(4):
            value = scores[row, column]
            if np.isnan(value):
                axis.text(column, row, "—", ha="center", va="center",
                          color="#98a2ad", fontsize=11)
                continue
            red, green, blue, _ = cmap(norm(ratios[row, column]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "white" if luminance < 0.55 else "#202631"
            axis.text(column, row, f"{value * 100:.2f}", ha="center",
                      va="center", color=color, fontsize=11,
                      fontweight="semibold")

    axis.set_xlim(-0.5, 4.5)
    axis.set_ylim(len(labels) - 0.5, -0.5)
    axis.set_xticks(range(5), [f"Stage {i + 1}\n{DISPLAY[domain]}"
                               for i, domain in enumerate(order)] + ["Expert"])
    axis.set_yticks(range(len(labels)), labels)
    axis.xaxis.tick_top()
    axis.tick_params(axis="both", which="major", length=0, labelsize=10, pad=8)
    axis.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.axvline(3.5, color="#9aa6b2", linewidth=1.2)
    for spine in axis.spines.values():
        spine.set_visible(False)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.035,
                               ticks=(0, 0.5, 1.0, 1.3))
    colorbar.set_label("Score / expert", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    figure.suptitle(f"Performance across merging stages ({order_name})",
                    fontsize=14, y=0.97)
    figure.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white",
                   bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
