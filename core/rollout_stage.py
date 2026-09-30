#!/usr/bin/env python3
"""Compatibility resolver used by the domain rollout collectors."""

from __future__ import annotations

from pathlib import Path

from paths import load_experiment


class Stage:
    def __init__(self, experiment: Path, name: str):
        self.experiment = experiment
        self.config = load_experiment(experiment / "config/experiment.json")
        records = {item["name"]: item for item in self.config["stages"]}
        if name not in records:
            raise SystemExit(f"unknown stage {name!r}; choose from {sorted(records)}")
        self.record = records[name]
        self.name = name

    @property
    def student_init(self) -> Path:
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
    def state_dir(self) -> Path:
        return self.experiment / "data" / self.name / "states"

    @property
    def artifact_dir(self) -> Path:
        return self.experiment / "artifacts" / self.name

    @property
    def log_dir(self) -> Path:
        return self.experiment / "logs" / self.name

    def prepare(self) -> None:
        for path in (self.state_dir, self.artifact_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)
