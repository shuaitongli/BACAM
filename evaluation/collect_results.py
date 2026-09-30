#!/usr/bin/env python3
"""Collect frozen held-out scores and apply the BACAM acceptance lines."""

from __future__ import annotations

import argparse
import json
import re
import string
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from paths import load_experiment


BFCL_CATEGORIES = ("multi_turn_base",)


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def sub_em(prediction: str, answers: list[str]) -> float:
    prediction = normalize(prediction)
    return float(any(normalize(answer) in prediction for answer in answers if answer))


def bfcl_macro(experiment: Path, model_id: str) -> dict:
    path = experiment / f"artifacts/eval/bfcl/score/bacam-wtsa-{model_id}-ours/summary.json"
    summary = json.loads(path.read_text())
    values = {name: float(summary["categories"][name]["accuracy"])
              for name in BFCL_CATEGORIES}
    if any(int(summary["categories"][name]["total"]) != 100 for name in BFCL_CATEGORIES):
        raise RuntimeError("BFCL held-out category is incomplete")
    return {"macro_accuracy": sum(values.values()) / len(values),
            "per_category": values, "artifact": str(path)}


def search_accuracy(experiment: Path, model_id: str) -> dict:
    root = experiment / f"artifacts/eval/search/{model_id}/run"
    paths = sorted(root.glob("*/intermediate_data.json"), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"no search result under {root}")
    rows = json.loads(paths[-1].read_text())
    expected = json.loads((experiment / "data/splits/search_holdout_dataset.json").read_text())["questions"]
    if len(rows) != expected:
        raise RuntimeError(f"search rows {len(rows)} != {expected}")
    scores = [sub_em((row.get("output") or {}).get("pred"), row.get("golden_answers") or [])
              for row in rows]
    return {"accuracy": sum(scores) / len(scores), "n": len(scores), "artifact": str(paths[-1])}


def alfworld_scores(experiment: Path, model_id: str) -> dict:
    results = {}
    expected = {"iid": 140, "ood": 134}
    for split, rounds in expected.items():
        path = experiment / f"artifacts/eval/alfworld/{model_id}_{split}_vllm.json"
        payload = json.loads(path.read_text())
        evaluation = payload["evaluation"]
        episodes = payload["episodes"]
        if payload.get("status") != "complete" or len(episodes) != rounds:
            raise RuntimeError(f"incomplete ALFWorld result: {path}")
        steps = [int(episode["steps"]) for episode in episodes]
        results[split] = {
            "sr": float(evaluation["sr"]),
            "average_round": sum(steps) / len(steps),
            "n": len(steps),
            "artifact": str(path),
        }
    return results


def webshop_scores(experiment: Path, model_id: str) -> dict:
    path = experiment / f"artifacts/eval/webshop/{model_id}_test_vllm.json"
    payload = json.loads(path.read_text())
    evaluation = payload["evaluation"]
    episodes = payload["episodes"]
    if payload.get("status") != "complete" or len(episodes) != 500:
        raise RuntimeError(f"incomplete WebShop result: {path}")
    if [int(item["goal_index"]) for item in episodes] != list(range(500)):
        raise RuntimeError(f"WebShop test goals are not exactly 0..499: {path}")
    scores = [float(item["task_score"]) for item in episodes]
    rounds = [int(item["steps"]) for item in episodes]
    successes = sum(score == 1.0 for score in scores)
    result = {
        "task_score": sum(scores) / len(scores),
        "sr": successes / len(scores),
        "average_round": sum(rounds) / len(rounds),
        "n": len(episodes),
        "artifact": str(path),
    }
    for key in ("task_score", "sr", "average_round"):
        if abs(float(evaluation[key]) - result[key]) > 1e-12:
            raise RuntimeError(f"WebShop {key} summary does not match episodes: {path}")
    return result


def gate(kind: str, measured: float, line: float, **extra) -> dict:
    return {"kind": kind, "measured": measured, "line": line,
            "pass": measured >= line, **extra}


def git_commit(experiment: Path) -> str:
    result = subprocess.run(["git", "-C", str(experiment), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unknown"


def run_metadata(experiment: Path, stage_name: str, model_id: str, config: dict) -> dict:
    stage = next(item for item in config["stages"] if item["name"] == stage_name)
    steps = stage["optimizer_steps"]
    manifests = [str(experiment / "data" / stage_name / round_name / "states" /
                     "train_manifest.json") for round_name in config["optimization"]["rounds"]]
    endpoint_scales, topk_masses, gate_stats = {}, {}, {}
    for round_name in config["optimization"]["rounds"]:
        artifact = experiment / "artifacts" / stage_name / round_name
        endpoint_path = artifact / "endpoints.json"
        if endpoint_path.exists():
            endpoint_scales[round_name] = {
                name: value["scale"] for name, value in json.loads(endpoint_path.read_text()).items()}
        summary_path = artifact / "training_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            gate_stats[round_name] = {key: summary[key] for key in
                                      ("parameters", "free_coordinates", "mean_gate_on_free",
                                       "gate_histogram")
                                      if key in summary}
        root = experiment / "data" / stage_name / round_name / "teacher_cache"
        for manifest in root.glob("*/manifest.json"):
            item = json.loads(manifest.read_text())
            topk_masses[str(manifest)] = item.get("mean_topk_probability_mass")
    return {
        "model_path": stage["final_export_dir"], "commit": git_commit(experiment),
        "seed": config["seed"], "trajectory_manifests": manifests,
        "top32_mass": topk_masses, "endpoint_scale": endpoint_scales,
        "rounds": config["optimization"]["rounds"], "steps": steps,
        "gate_stats": gate_stats,
    }


def main() -> None:
    experiment = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=experiment)
    parser.add_argument("--model-id", choices=("t2", "t3", "t4"), required=True)
    args = parser.parse_args()
    config = load_experiment(args.experiment / "config/experiment.json")
    success = config["success"]
    references = json.loads(
        (args.experiment / "baseline/config.json").read_text())["references"]

    if args.model_id == "t2":
        stage_name = "t2_tool"
        webshop = webshop_scores(args.experiment, "t2")
        tool = bfcl_macro(args.experiment, "t2")
        retention = float(success["t2"]["retention"])
        absorption = float(success["t2"]["absorption"])
        gates = {
            "webshop": gate(
                "WebShop retention", webshop["task_score"],
                retention * float(references["webshop"]["expert"])),
            "tool": gate(
                "Tool absorption", tool["macro_accuracy"],
                absorption * float(references["tool"]["expert"])),
        }
        details = {"webshop": webshop, "tool": tool}
    elif args.model_id == "t3":
        stage_name = "t3_search"
        webshop = webshop_scores(args.experiment, "t3")
        tool = bfcl_macro(args.experiment, "t3")
        search = search_accuracy(args.experiment, "t3")
        retention = float(success["t3"]["retention"])
        absorption = float(success["t3"]["absorption"])
        gates = {
            "webshop": gate(
                "WebShop retention", webshop["task_score"],
                retention * float(references["webshop"]["expert"])),
            "tool": gate(
                "Tool retention", tool["macro_accuracy"],
                retention * float(references["tool"]["expert"])),
            "search": gate(
                "Search absorption", search["accuracy"],
                absorption * float(references["search"]["expert"])),
        }
        details = {"webshop": webshop, "tool": tool, "search": search}
    else:
        stage_name = "t4_alfworld"
        webshop = webshop_scores(args.experiment, "t4")
        alfworld = alfworld_scores(args.experiment, "t4")
        tool = bfcl_macro(args.experiment, "t4")
        search = search_accuracy(args.experiment, "t4")
        retention = float(success["t4"]["retention"])
        absorption = float(success["t4"]["absorption"])
        gates = {
            "webshop": gate(
                "WebShop retention", webshop["task_score"],
                retention * float(references["webshop"]["expert"])),
            "alfworld_iid": gate(
                "ALFWorld IID absorption", alfworld["iid"]["sr"],
                absorption * float(references["alfworld"]["expert_iid"])),
            "alfworld_ood": gate(
                "ALFWorld OOD absorption", alfworld["ood"]["sr"],
                absorption * float(references["alfworld"]["expert_ood"])),
            "tool": gate(
                "Tool retention", tool["macro_accuracy"],
                retention * float(references["tool"]["expert"])),
            "search": gate(
                "Search retention", search["accuracy"],
                retention * float(references["search"]["expert"])),
        }
        details = {"webshop": webshop, "alfworld": alfworld,
                   "tool": tool, "search": search}

    stage = next(item for item in config["stages"] if item["name"] == stage_name)
    model_path = Path(stage["final_export_dir"])
    forbidden = [name for name in ("gate_state", "teacher_cache", "adapter_model.bin")
                 if (model_path / name).exists()]
    deployment = {
        "model": str(model_path), "exists": model_path.exists(),
        "has_config": (model_path / "config.json").exists(),
        "safetensor_files": len(list(model_path.glob("*.safetensors"))) if model_path.exists() else 0,
        "forbidden_artifacts": forbidden,
    }
    deployment["pass"] = (deployment["exists"] and deployment["has_config"] and
                           deployment["safetensor_files"] > 0 and not forbidden)
    result = {
        "method_version": config["method_version"], "model_id": args.model_id,
        "stage": stage_name, "gates": gates, "deployment": deployment,
        "primary_pass": all(item["pass"] for item in gates.values()) and deployment["pass"],
        "details": details, "run_metadata": run_metadata(args.experiment, stage_name,
                                                            args.model_id, config),
    }
    target = args.experiment / f"artifacts/eval/results_{args.model_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
