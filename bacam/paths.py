"""Shared local paths for Python and shell entry points (stdlib only)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
from string import Template
import sys


ROOT = Path(__file__).resolve().parents[1]


def configured_paths() -> dict[str, str]:
    values = {"BACAM_ROOT": str(ROOT)}
    defaults = {
        "BACAM_MODEL_ROOT": ROOT / "models",
        "BACAM_MERGE_ROOT": ROOT / "outputs" / "models",
        "BACAM_DATA_ROOT": ROOT / "datasets",
        "BACAM_EXTERNAL_ROOT": ROOT / "third_party",
    }
    for key, default in defaults.items():
        path = Path(os.environ.get(key, str(default))).expanduser()
        values[key] = str((ROOT / path).resolve())
    children = {
        "BACAM_RESEARCH_ROOT": Path(values["BACAM_EXTERNAL_ROOT"]) / "ReSearch",
        "BACAM_BFCL_ROOT": Path(values["BACAM_EXTERNAL_ROOT"]) / "bfcl-toolrl",
        "BACAM_VERL_AGENT_ROOT": Path(values["BACAM_EXTERNAL_ROOT"]) / "verl-agent",
    }
    for key, default in children.items():
        path = Path(os.environ.get(key, str(default))).expanduser()
        values[key] = str((ROOT / path).resolve())
    python = os.environ.get("BACAM_PYTHON", sys.executable)
    commands = {
        "BACAM_PYTHON": python,
        "BACAM_TORCHRUN": "torchrun",
        "BACAM_TOOL_PYTHON": python,
        "BACAM_SEARCH_PYTHON": python,
        "BACAM_SGLANG_PYTHON": python,
        "BACAM_ALFWORLD_PYTHON": python,
        "BACAM_WEBSHOP_PYTHON": python,
        "BACAM_VLLM": "vllm",
        "BACAM_BFCL": "bfcl",
    }
    for key, default in commands.items():
        command = os.path.expanduser(os.environ.get(key, default))
        values[key] = str((ROOT / command).resolve()) if "/" in command else command
    return values


PATHS = configured_paths()


def configured_execution(config: dict) -> dict[str, str]:
    execution = config.get("execution", {})
    default_gpus = execution.get("allowed_physical_gpus", [3, 4, 5, 6])
    gpu_text = os.environ.get("BACAM_GPUS", ",".join(map(str, default_gpus)))
    parts = gpu_text.split(",")
    if len(parts) != 4 or any(not value.strip().isdigit() for value in parts):
        raise ValueError("BACAM_GPUS must contain four nonnegative GPU IDs")
    gpus = [int(value) for value in parts]
    if len(set(gpus)) != 4:
        raise ValueError("BACAM_GPUS must contain four distinct GPU IDs")
    values = {"BACAM_GPUS": ",".join(map(str, gpus))}
    for domain, gpu in zip(("TOOL", "SEARCH", "ALFWORLD", "WEBSHOP"), gpus):
        values[f"BACAM_GPU_{domain}"] = str(gpu)
    defaults = {
        "retriever": 8100, "tool": 19300,
        "search_rollout": 18300, "search_eval": 18400,
        "alfworld_rollout": 18500, "alfworld_eval": 18600,
        "webshop_rollout": 18700, "webshop_eval": 18800,
        "master": 29500,
    }
    offset = int(os.environ.get("BACAM_PORT_OFFSET", "0"))
    ports = execution.get("ports", {})
    selected = []
    for name, default in defaults.items():
        key = f"BACAM_{name.upper()}_PORT"
        port = int(os.environ.get(key, str(int(ports.get(name, default)) + offset)))
        if not 1 <= port <= 65535:
            raise ValueError(f"{key} must be between 1 and 65535")
        values[key] = str(port)
        selected.append(port)
    if len(set(selected)) != len(selected):
        raise ValueError("BACAM service ports must be distinct")
    return values


def load_experiment(path: Path) -> dict:
    """Expand shared path variables without depending on the current directory."""
    def expand(value):
        if isinstance(value, str):
            return Template(value).substitute(PATHS)
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    config = expand(json.loads(path.read_text()))
    settings = configured_execution(config)
    if "execution" in config:
        config["execution"]["allowed_physical_gpus"] = list(map(int, settings["BACAM_GPUS"].split(",")))
        config["execution"]["ports"] = {
            key.removeprefix("BACAM_").removesuffix("_PORT").lower(): int(value)
            for key, value in settings.items() if key.endswith("_PORT")
        }
    return config


if __name__ == "__main__":
    values = {**PATHS, **configured_execution(json.loads((ROOT / "configs/experiment.json").read_text()))}
    if sys.argv[1:] == ["--shell"]:
        for key, value in values.items():
            print(f"export {key}={shlex.quote(value)}")
    else:
        print(json.dumps(values, indent=2))
