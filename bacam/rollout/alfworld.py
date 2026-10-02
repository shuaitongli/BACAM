#!/usr/bin/env python3
"""Roll out ALFWorld and directly persist final BACAM training states.

Unlike BFCL, this driver owns the environment loop.  It therefore records the
exact serialized prompt seen by the candidate and its raw response at generation
time; there is intentionally no post-hoc state reconstruction step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT = Path(__file__).resolve().parents[2]

from bacam.evaluation.eval_alfworld_expert import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DATA,
    DEFAULT_PROJECTION,
    build_prompt,
    initialize_episode,
    json_request,
    load_projection,
    scalar,
    task_type,
)


def serialize_qwen_user_prompt(content: str) -> str:
    """Serialize the one-user-message Qwen2.5 chat used by the vLLM server."""
    return (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        "<|im_end|>\n"
        f"<|im_start|>user\n{content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class CompletionClient:
    def __init__(self, api_base: str, model_name: str, timeout: float,
                 max_tokens: int, temperature: float, top_p: float) -> None:
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def check(self) -> None:
        response = json_request(
            f"{self.api_base}/models", None, api_key="EMPTY", timeout=self.timeout)
        available = [item.get("id") for item in response.get("data", [])]
        if self.model_name not in available:
            raise RuntimeError(
                f"served model {self.model_name!r} not found; available={available}")

    def generate(self, prompt: str, seed: int) -> str:
        response = json_request(
            f"{self.api_base}/completions",
            {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": seed,
            },
            api_key="EMPTY",
            timeout=self.timeout,
        )
        try:
            text = response["choices"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected completion response: {response}") from exc
        if not isinstance(text, str):
            raise RuntimeError(f"completion is not text: {text!r}")
        return text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def completed_episode_ids(rows: list[dict[str, Any]]) -> set[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["episode_id"])].append(row)
    completed = set()
    for episode_id, states in grouped.items():
        ordered = sorted(states, key=lambda item: int(item["turn"]))
        turns = [int(item["turn"]) for item in ordered]
        if turns != list(range(len(ordered))):
            raise RuntimeError(f"{episode_id}: non-contiguous saved turns {turns}")
        if not bool(ordered[-1].get("trajectory_terminal", False)):
            raise RuntimeError(f"{episode_id}: saved trajectory is not terminal")
        completed.add(episode_id)
    return completed


def write_summary(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]],
                  pool: list[tuple[str, str]]) -> None:
    completed = completed_episode_ids(rows)
    final_rows = [row for row in rows if row.get("trajectory_terminal")]
    summary = {
        "stage": args.stage,
        "round": args.round,
        "model": str(args.model),
        "pool_episodes": len(pool),
        "episodes": len(completed),
        "states": len(rows),
        "successes": sum(bool(row.get("won")) for row in final_rows),
        "complete": len(completed) == len(pool),
        "max_steps": args.max_steps,
        "history_length": args.history_length,
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "episodes_by_task": dict(Counter(task for task, _ in pool)),
        "completed_by_task": dict(Counter(
            str(row["task_type"]) for row in final_rows)),
        "output": str(args.output),
    }
    atomic_write_json(path, summary)


def validate_resume_summary(path: Path, args: argparse.Namespace,
                            pool: list[tuple[str, str]]) -> None:
    if not path.is_file():
        raise RuntimeError(
            f"cannot resume {args.output}: matching summary is missing at {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "stage": args.stage,
        "round": args.round,
        "model": str(args.model),
        "pool_episodes": len(pool),
        "max_steps": args.max_steps,
        "history_length": args.history_length,
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }
    mismatches = {
        key: {"saved": summary.get(key), "requested": value}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"refusing to resume incompatible ALFWorld states: {mismatches}")


def run_batch(base_env: Any, batch: list[tuple[int, str, str]], args: argparse.Namespace,
              client: CompletionClient, projection: Any) -> list[dict[str, Any]]:
    episodes = [
        initialize_episode(
            base_env,
            round_index=index,
            seed=args.seed + index,
            selected_gamefile=str(args.data_path / relative_gamefile),
        )
        for index, _, relative_gamefile in batch
    ]
    records_by_episode: dict[str, list[dict[str, Any]]] = {
        relative_gamefile: [] for _, _, relative_gamefile in batch
    }
    metadata = {
        episode.round_index: (task, relative_gamefile)
        for episode, (_, task, relative_gamefile) in zip(episodes, batch)
    }

    try:
        with ThreadPoolExecutor(max_workers=min(args.request_workers, len(episodes))) as executor:
            for step_index in range(args.max_steps):
                active = [episode for episode in episodes if not episode.done]
                if not active:
                    break
                environment_prompts = [
                    build_prompt(episode, args.history_length) for episode in active]
                serialized_prompts = [
                    serialize_qwen_user_prompt(prompt) for prompt in environment_prompts]
                futures = [
                    executor.submit(
                        client.generate, prompt, episode.seed + step_index)
                    for episode, prompt in zip(active, serialized_prompts)
                ]
                responses = [future.result() for future in futures]
                action_pools = [episode.admissible_actions for episode in active]
                actions, valids = projection(responses.copy(), action_pools)

                for episode, prompt, response, action, valid in zip(
                        active, serialized_prompts, responses, actions, valids):
                    task, relative_gamefile = metadata[episode.round_index]
                    previous_observation = episode.observation
                    previous_actions = list(episode.admissible_actions)
                    observations, _, dones, infos = episode.env.step([action])
                    info = {key: scalar(value) for key, value in infos.items()}
                    episode.history.append((previous_observation, action))
                    episode.observation = str(observations[0])
                    episode.admissible_actions = list(info["admissible_commands"])
                    episode.done = bool(dones[0])
                    episode.won = bool(info.get("won", False))
                    records_by_episode[relative_gamefile].append({
                        "domain": "alfworld",
                        "episode_id": relative_gamefile,
                        "task_type": task,
                        "turn": len(episode.history) - 1,
                        "prompt": prompt,
                        "response": response,
                        "observation": previous_observation,
                        "admissible_actions": previous_actions,
                        "projected_action": action,
                        "format_valid": bool(valid),
                        "action_admissible": action in previous_actions,
                        "done": episode.done,
                        "won": episode.won,
                        "trajectory_terminal": False,
                    })

        result = []
        for _, _, relative_gamefile in batch:
            episode_records = records_by_episode[relative_gamefile]
            if not episode_records:
                raise RuntimeError(f"{relative_gamefile}: rollout produced no states")
            episode_records[-1]["trajectory_terminal"] = True
            result.extend(episode_records)
        return result
    finally:
        for episode in episodes:
            episode.env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--round", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--projection-path", type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--request-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.request_workers <= 0 or args.max_steps <= 0:
        parser.error("batch size, workers, and max steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    projection = load_projection(args.projection_path)
    split = json.loads(args.split.read_text())
    if Path(split["data_root"]).resolve() != args.data_path.resolve():
        raise RuntimeError("ALFWorld split data_root does not match --data-path")
    config = json.loads((EXPERIMENT / "configs/experiment.json").read_text())
    rollout_episodes = int(config["data"]["rollout_episodes"]["alfworld"])
    if rollout_episodes % len(split["task_types"]):
        raise RuntimeError("ALFWorld rollout budget must divide across task types")
    per_task = rollout_episodes // len(split["task_types"])
    pool = []
    for task in split["task_types"]:
        seed = f'{args.seed}:{args.stage}:{args.round}:alfworld:{task}'
        games = sorted(
            split["optimization_by_task"][task],
            key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest())
        pool.extend((task, gamefile) for gamefile in games[:per_task])
    if len({gamefile for _, gamefile in pool}) != len(pool):
        raise RuntimeError("ALFWorld optimization pool contains duplicates")
    missing = [gamefile for _, gamefile in pool
               if not (args.data_path / gamefile).is_file()]
    if missing:
        raise FileNotFoundError(f"missing ALFWorld games: {missing[:3]}")

    args.model = args.model.resolve()
    args.output = args.output.resolve()
    summary_path = args.output.with_suffix(".summary.json")
    rows = read_jsonl(args.output) if args.resume else []
    if rows:
        validate_resume_summary(summary_path, args, pool)
    completed = completed_episode_ids(rows)
    pool_ids = {gamefile for _, gamefile in pool}
    if not completed <= pool_ids:
        raise RuntimeError("saved states contain episodes outside the frozen pool")

    client = CompletionClient(
        args.api_base, args.served_model_name, args.request_timeout,
        args.max_tokens, args.temperature, args.top_p)
    client.check()
    os.environ["ALFWORLD_DATA"] = str(args.data_path)
    with args.config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["dataset"]["num_train_games"] = -1
    from alfworld.agents.environment import get_environment
    base_env = get_environment("AlfredTWEnv")(config, train_eval="train")

    pending = [
        (index, task, gamefile)
        for index, (task, gamefile) in enumerate(pool)
        if gamefile not in completed
    ]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        rows.extend(run_batch(base_env, batch, args, client, projection))
        atomic_write_jsonl(args.output, rows)
        write_summary(summary_path, args, rows, pool)
        completed = completed_episode_ids(rows)
        print(f"ALFWorld states: {len(completed)}/{len(pool)} episodes, {len(rows)} states")

    write_summary(summary_path, args, rows, pool)
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
