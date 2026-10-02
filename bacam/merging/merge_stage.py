#!/usr/bin/env python3
"""Small, explicit resolver for one BACAM stage and one training round."""

from __future__ import annotations

from pathlib import Path

from bacam.paths import load_experiment


class MergeStage:
    def __init__(self, experiment: Path, stage_name: str, round_name: str):
        self.experiment = experiment
        self.config_path = experiment / "configs/experiment.json"
        self.config = load_experiment(self.config_path)
        records = {item["name"]: item for item in self.config["stages"]}
        if stage_name not in records:
            raise SystemExit(f"unknown stage {stage_name!r}; choose from {sorted(records)}")
        rounds = self.config["optimization"]["rounds"]
        if round_name not in rounds:
            raise SystemExit(f"unknown round {round_name!r}; choose from {rounds}")
        self.record = records[stage_name]
        self.round = round_name
        self.round_index = rounds.index(round_name)

    @property
    def name(self) -> str:
        return self.record["name"]

    @property
    def old_model(self) -> Path:
        return Path(self.record["old_model"])

    @property
    def new_expert(self) -> Path:
        return Path(self.record["new_expert"])

    @property
    def old_domains(self) -> list[str]:
        return list(self.record["old_domains"])

    @property
    def new_domains(self) -> list[str]:
        return list(self.record["new_domains"])

    @property
    def domains(self) -> list[str]:
        return self.old_domains + self.new_domains

    @property
    def stage_root(self) -> Path:
        return self.experiment / "data" / self.name

    @property
    def round_root(self) -> Path:
        return self.stage_root / self.round

    @property
    def state_dir(self) -> Path:
        return self.round_root / "states"

    @property
    def cache_dir(self) -> Path:
        return self.round_root / "teacher_cache"

    @property
    def artifact_dir(self) -> Path:
        return self.experiment / "artifacts" / self.name / self.round

    @property
    def log_dir(self) -> Path:
        return self.experiment / "logs" / self.name / self.round

    @property
    def export_dir(self) -> Path:
        return Path(self.record["round_export_dirs"][self.round_index])

    @property
    def final_export_dir(self) -> Path:
        return Path(self.record["final_export_dir"])

    @property
    def previous_artifact_dir(self) -> Path | None:
        if self.round_index == 0:
            return None
        previous = self.config["optimization"]["rounds"][self.round_index - 1]
        return self.experiment / "artifacts" / self.name / previous

    @property
    def current_model(self) -> Path:
        if self.round_index == 0:
            return self.old_model
        return Path(self.record["round_export_dirs"][self.round_index - 1])

    @property
    def steps(self) -> int:
        return int(self.record["optimizer_steps"][self.round])

    @property
    def gate_lr(self) -> float:
        return float(self.config["optimization"]["gate_lr"])

    @property
    def round_start_step(self) -> int:
        rounds = self.config["optimization"]["rounds"]
        return sum(int(self.record["optimizer_steps"][name])
                   for name in rounds[:self.round_index])

    @property
    def global_start_step(self) -> int:
        # The cosine schedule restarts at the beginning of each stage.  This
        # is the same schedule used by the gate implementation in the base
        # experiment; the stage step counts are explicit in the config.
        return self.round_start_step

    @property
    def total_steps(self) -> int:
        total = sum(int(self.record["optimizer_steps"][name])
                    for name in self.config["optimization"]["rounds"])
        configured = int(self.record["total_optimizer_steps"])
        if total != configured:
            raise RuntimeError(f"{self.name}: total_optimizer_steps={configured} != stage sum={total}")
        return total

    def prepare(self) -> None:
        for path in (self.state_dir, self.cache_dir, self.artifact_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)
