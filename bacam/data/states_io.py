#!/usr/bin/env python3
"""Shared writer for the per-domain student state files."""

from __future__ import annotations

import json
from pathlib import Path


def write_states(path: Path, records: list[dict], summary: dict) -> None:
    """One jsonl of states plus a sibling .summary.json, printed for the run log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
