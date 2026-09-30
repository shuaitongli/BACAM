#!/usr/bin/env python3
"""Reproduce the released GiGPO ALFWorld expert evaluation.

The script intentionally keeps model serving separate from the Python 3.9
ALFWorld environment.  It uses ALFWorld's official TextWorld environment and
loads verl-agent's action projection directly from the checked-out source.

The defaults follow the commit linked by the released model card:
35b3da38293993f9bf4f7873dfb3262a361e956c
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from paths import PATHS, load_experiment


MODEL_CARD_COMMIT = "35b3da38293993f9bf4f7873dfb3262a361e956c"
EXPECTED_PROJECTION_SHA256 = (
    "e794d1217d613ef4b550cfe4bfd0b39b4deb10d47493f31f78aef7b5ebf6dd98"
)
EXPECTED_CONFIG_SHA256 = (
    "a33f87ab43253ea602f93c9a0b176771a8ae16a34cd953130e45fc03700f1f49"
)

VERL_AGENT = Path(PATHS["BACAM_VERL_AGENT_ROOT"])
DEFAULT_DATA = Path(PATHS["BACAM_DATA_ROOT"]) / "alfworld"
DEFAULT_MODEL = Path(PATHS["BACAM_MODEL_ROOT"]) / "GiGPO-Qwen2.5-7B-Instruct-ALFWorld"
DEFAULT_CONFIG = (
    VERL_AGENT
    / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
)
DEFAULT_PROJECTION = (
    VERL_AGENT
    / "agent_system/environments/env_package/alfworld/projection.py"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts/eval/alfworld"


# Exact prompt text from the model card and its linked verl-agent commit.
# Keep the original "observaitons" spelling: that is part of the training prompt.
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reproduction_sources(config_path: Path, projection_path: Path) -> None:
    expected = {
        config_path: EXPECTED_CONFIG_SHA256,
        projection_path: EXPECTED_PROJECTION_SHA256,
    }
    for path, expected_hash in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required upstream file does not exist: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Upstream source drift detected for {path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )


def load_projection(path: Path) -> Callable[[list[str], list[list[str]]], Any]:
    spec = importlib.util.spec_from_file_location("_pinned_alfworld_projection", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load verl-agent projection from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.alfworld_projection


def json_request(
    url: str,
    payload: dict[str, Any] | None,
    *,
    api_key: str,
    timeout: float,
    retries: int = 3,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


class ChatClient:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        timeout: float,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    def check(self) -> None:
        response = json_request(
            f"{self.api_base}/models",
            None,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        names = [item.get("id") for item in response.get("data", [])]
        if self.model_name not in names:
            raise RuntimeError(
                f"Served model {self.model_name!r} not found at {self.api_base}; "
                f"available models: {names}"
            )

    def generate(self, prompt: str, seed: int) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": seed,
        }
        response = json_request(
            f"{self.api_base}/chat/completions",
            payload,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat-completion response: {response}") from exc
        if not isinstance(content, str):
            raise RuntimeError(f"Model returned non-text content: {content!r}")
        return content


@dataclass
class Episode:
    round_index: int
    seed: int
    env: Any
    observation: str
    admissible_actions: list[str]
    task: str
    gamefile: str | None
    history: list[tuple[str, str]] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    won: bool = False
    final_info: dict[str, Any] = field(default_factory=dict)


def extract_task(observation: str) -> str:
    marker = "Your task is to: "
    start = observation.find(marker)
    if start == -1:
        raise ValueError(f"Task description not found in observation: {observation!r}")
    return observation[start + len(marker) :].strip()


def scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    try:
        return value.item()
    except AttributeError:
        return value


def build_prompt(episode: Episode, history_length: int) -> str:
    admissible = "\n ".join(
        f"'{action}'" for action in episode.admissible_actions if action != "help"
    )
    if not episode.history or history_length <= 0:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=episode.observation,
            admissible_actions=admissible,
        )

    recent = episode.history[-history_length:]
    start_index = len(episode.history) - len(recent)
    lines = []
    for offset, (observation, action) in enumerate(recent):
        step_number = start_index + offset + 1
        lines.append(
            f"[Observation {step_number}: '{observation}', "
            f"Action {step_number}: '{action}']"
        )
    return ALFWORLD_TEMPLATE.format(
        task_description=episode.task,
        step_count=len(episode.history),
        history_length=len(recent),
        action_history="\n".join(lines),
        current_step=len(episode.history) + 1,
        current_observation=episode.observation,
        admissible_actions=admissible,
    )


def task_type(gamefile: str | None) -> str:
    if not gamefile:
        return "unknown"
    for name in (
        "pick_and_place_simple",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    ):
        if name in gamefile:
            return name
    return "unknown"


def initialize_episode(
    base_env: Any,
    round_index: int,
    seed: int,
    selected_gamefile: str | None = None,
) -> Episode:
    original_game_files = base_env.game_files
    original_num_games = base_env.num_games
    if selected_gamefile is not None:
        base_env.game_files = [selected_gamefile]
        base_env.num_games = 1
    try:
        env = base_env.init_env(batch_size=1)
    finally:
        base_env.game_files = original_game_files
        base_env.num_games = original_num_games
    env.seed(seed)
    observations, infos = env.reset()
    observation = str(observations[0])
    gamefile = scalar(infos.get("extra.gamefile"))
    if selected_gamefile is not None and Path(str(gamefile)) != Path(selected_gamefile):
        env.close()
        raise RuntimeError(
            f"Requested game {selected_gamefile}, but environment loaded {gamefile}"
        )
    return Episode(
        round_index=round_index,
        seed=seed,
        env=env,
        observation=observation,
        admissible_actions=list(scalar(infos["admissible_commands"])),
        task=extract_task(observation),
        gamefile=str(gamefile) if gamefile is not None else None,
    )


def run_batch(
    base_env: Any,
    start_index: int,
    count: int,
    base_seed: int,
    max_steps: int,
    history_length: int,
    workers: int,
    client: ChatClient,
    projection: Callable[[list[str], list[list[str]]], Any],
    gamefiles: list[str] | None,
) -> list[dict[str, Any]]:
    episodes = [
        initialize_episode(
            base_env,
            index,
            base_seed + index,
            None if gamefiles is None else gamefiles[index],
        )
        for index in range(start_index, start_index + count)
    ]

    try:
        with ThreadPoolExecutor(max_workers=min(workers, count)) as executor:
            for step_index in range(max_steps):
                active = [episode for episode in episodes if not episode.done]
                if not active:
                    break

                prompts = [build_prompt(episode, history_length) for episode in active]
                futures = [
                    executor.submit(client.generate, prompt, episode.seed + step_index)
                    for episode, prompt in zip(active, prompts)
                ]
                responses = [future.result() for future in futures]
                action_pools = [episode.admissible_actions for episode in active]
                actions, valids = projection(responses.copy(), action_pools)

                for episode, response, action, valid in zip(
                    active, responses, actions, valids
                ):
                    previous_observation = episode.observation
                    observations, _, dones, infos = episode.env.step([action])
                    info = {key: scalar(value) for key, value in infos.items()}
                    episode.history.append((previous_observation, action))
                    episode.trajectory.append(
                        {
                            "step": len(episode.history),
                            "observation": previous_observation,
                            "admissible_actions": episode.admissible_actions,
                            "response": response,
                            "action": action,
                            "format_valid": bool(valid),
                            "action_admissible": action in episode.admissible_actions,
                        }
                    )
                    episode.observation = str(observations[0])
                    episode.admissible_actions = list(info["admissible_commands"])
                    episode.done = bool(dones[0])
                    episode.won = bool(info.get("won", False))
                    episode.final_info = {
                        "won": episode.won,
                        "extra.gamefile": info.get("extra.gamefile", episode.gamefile),
                    }

        results = []
        for episode in episodes:
            valid_count = sum(item["format_valid"] for item in episode.trajectory)
            admissible_count = sum(
                item["action_admissible"] for item in episode.trajectory
            )
            results.append(
                {
                    "round": episode.round_index + 1,
                    "seed": episode.seed,
                    "gamefile": episode.gamefile,
                    "task_type": task_type(episode.gamefile),
                    "task": episode.task,
                    "won": episode.won,
                    "steps": len(episode.trajectory),
                    "terminated": episode.done,
                    "valid_format_actions": valid_count,
                    "admissible_actions_taken": admissible_count,
                    "trajectory": episode.trajectory,
                }
            )
        return results
    finally:
        for episode in episodes:
            episode.env.close()


def build_summary(
    args: argparse.Namespace,
    distribution: str,
    dataset: str,
    episodes: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    successes = sum(bool(episode["won"]) for episode in episodes)
    breakdown: dict[str, dict[str, int | float]] = {}
    grouped: dict[str, list[bool]] = defaultdict(list)
    for episode in episodes:
        grouped[episode["task_type"]].append(bool(episode["won"]))
    for name, values in sorted(grouped.items()):
        count = len(values)
        won = sum(values)
        breakdown[name] = {"rounds": count, "successes": won, "sr": won / count}

    return {
        "schema_version": 1,
        "status": "complete" if len(episodes) >= args.rounds else "partial",
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "langfengQ/verl-agent",
            "model_card_commit": MODEL_CARD_COMMIT,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "projection_sha256": EXPECTED_PROJECTION_SHA256,
        },
        "model": {
            "path": str(args.model_path),
            "served_name": args.served_model_name,
            "inference_backend": args.inference_backend,
        },
        "evaluation": {
            "distribution": distribution,
            "dataset": dataset,
            "split": args.split,
            "selection": "all_unique_games" if args.all_games else "seeded_rounds",
            "rounds_requested": args.rounds,
            "rounds_completed": len(episodes),
            "successes": successes,
            "sr": successes / len(episodes) if episodes else None,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "history_length": args.history_length,
            "generation": {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            "task_breakdown": breakdown,
        },
        "episodes": episodes,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the released GiGPO Qwen2.5-7B ALFWorld expert."
    )
    parser.add_argument("--split", choices=("iid", "ood"), required=True)
    parser.add_argument("--rounds", type=int, default=128)
    parser.add_argument(
        "--all-games",
        action="store_true",
        help="Evaluate every unique game in the selected split exactly once.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--request-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    port = load_experiment(Path(PATHS["BACAM_ROOT"]) / "config/experiment.json")["execution"]["ports"]["alfworld_eval"]
    parser.add_argument("--api-base", default=f"http://127.0.0.1:{port}/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--served-model-name", default="alfworld-expert")
    parser.add_argument("--inference-backend", default="openai-compatible")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--projection-path", type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partial output file using the next deterministic round seed.",
    )
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.batch_size <= 0 or args.request_workers <= 0:
        parser.error("--batch-size and --request-workers must be positive")
    if args.max_steps <= 0 or args.history_length < 0 or args.max_tokens <= 0:
        parser.error("invalid step/history/token limit")
    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / f"expert_{args.split}.json"
    return args


def validate_paths(args: argparse.Namespace) -> None:
    required_model_files = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    )
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_path}")
    for name in required_model_files:
        path = args.model_path / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required model file is missing or empty: {path}")
    if not (args.data_path / "json_2.1.1/valid_seen").is_dir():
        raise FileNotFoundError(f"ALFWorld valid_seen data not found under {args.data_path}")
    if not (args.data_path / "json_2.1.1/valid_unseen").is_dir():
        raise FileNotFoundError(f"ALFWorld valid_unseen data not found under {args.data_path}")


def load_resume(path: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    if not args.resume or not path.exists():
        return [], datetime.now(timezone.utc).isoformat()
    with path.open(encoding="utf-8") as handle:
        previous = json.load(handle)
    evaluation = previous.get("evaluation", {})
    expected = {
        "split": args.split,
        "selection": "all_unique_games" if args.all_games else "seeded_rounds",
        "seed": args.seed,
        "max_steps": args.max_steps,
        "history_length": args.history_length,
    }
    for key, value in expected.items():
        if evaluation.get(key) != value:
            raise RuntimeError(
                f"Cannot resume {path}: {key} changed from "
                f"{evaluation.get(key)!r} to {value!r}"
            )
    previous_model = previous.get("model", {}).get("path")
    if previous_model != str(args.model_path):
        raise RuntimeError(
            f"Cannot resume {path}: model changed from {previous_model!r} "
            f"to {str(args.model_path)!r}"
        )
    previous_backend = previous.get("model", {}).get("inference_backend")
    if previous_backend not in (None, args.inference_backend):
        raise RuntimeError(
            f"Cannot resume {path}: inference backend changed from "
            f"{previous_backend!r} to {args.inference_backend!r}"
        )
    generation = evaluation.get("generation", {})
    expected_generation = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    for key, value in expected_generation.items():
        if generation.get(key) != value:
            raise RuntimeError(
                f"Cannot resume {path}: generation.{key} changed from "
                f"{generation.get(key)!r} to {value!r}"
            )
    return list(previous.get("episodes", [])), previous.get(
        "started_at", datetime.now(timezone.utc).isoformat()
    )


def main() -> int:
    args = parse_args()
    validate_paths(args)
    verify_reproduction_sources(args.config_path, args.projection_path)
    projection = load_projection(args.projection_path)

    os.environ["ALFWORLD_DATA"] = str(args.data_path)
    with args.config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["dataset"]["num_eval_games"] = -1

    train_eval = {
        "iid": "eval_in_distribution",
        "ood": "eval_out_of_distribution",
    }[args.split]
    dataset = {"iid": "valid_seen", "ood": "valid_unseen"}[args.split]

    client = ChatClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.served_model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.request_timeout,
    )
    client.check()

    from alfworld.agents.environment import get_environment

    base_env = get_environment("AlfredTWEnv")(config, train_eval=train_eval)
    gamefiles: list[str] | None = None
    if args.all_games:
        gamefiles = sorted(str(path) for path in base_env.game_files)
        args.rounds = len(gamefiles)

    episodes, started_at = load_resume(args.output, args)
    if len(episodes) >= args.rounds:
        print(f"Already complete: {len(episodes)}/{args.rounds} rounds in {args.output}")
        return 0

    print(
        f"Starting {args.split.upper()} evaluation: "
        f"{len(episodes)}/{args.rounds} rounds already complete"
    )

    while len(episodes) < args.rounds:
        count = min(args.batch_size, args.rounds - len(episodes))
        new_episodes = run_batch(
            base_env=base_env,
            start_index=len(episodes),
            count=count,
            base_seed=args.seed,
            max_steps=args.max_steps,
            history_length=args.history_length,
            workers=args.request_workers,
            client=client,
            projection=projection,
            gamefiles=gamefiles,
        )
        episodes.extend(new_episodes)
        summary = build_summary(
            args=args,
            distribution=args.split.upper(),
            dataset=dataset,
            episodes=episodes,
            started_at=started_at,
        )
        atomic_write_json(args.output, summary)
        evaluation = summary["evaluation"]
        print(
            f"Completed {evaluation['rounds_completed']}/{args.rounds}; "
            f"successes={evaluation['successes']}; SR={evaluation['sr']:.6f}"
        )

    print(f"Result written to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
