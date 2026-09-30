#!/usr/bin/env python3
"""Roll out frozen WebShop train goals and directly persist final BACAM states."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


EXPERIMENT = Path(__file__).resolve().parents[1]
EVALUATION = EXPERIMENT / "evaluation"
sys.path.insert(0, str(EVALUATION))

from eval_alfworld_expert import json_request  # noqa: E402
from eval_webshop import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_INDEX,
    PROJECTION_PATH,
    PROMPT_PATH,
    action_is_admissible,
    build_prompt,
    create_environment_pool,
    format_available_actions,
    format_observation,
    initialize_episode,
    load_module,
    validate_paths,
    verify_sources,
)


def serialize_qwen_user_prompt(content: str) -> str:
    """Serialize the one-user-message Qwen2.5 chat used by vLLM."""
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
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["episode_id"]), []).append(row)
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


def write_summary(path: Path, args: argparse.Namespace,
                  rows: list[dict[str, Any]], pool: list[dict[str, Any]]) -> None:
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
        "mean_task_score": (
            sum(float(row.get("task_score", 0.0)) for row in final_rows)
            / max(len(final_rows), 1)
        ),
        "complete": len(completed) == len(pool),
        "max_steps": args.max_steps,
        "history_length": args.history_length,
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "output": str(args.output),
    }
    atomic_write_json(path, summary)


def validate_resume_summary(path: Path, args: argparse.Namespace,
                            pool: list[dict[str, Any]]) -> None:
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
            f"refusing to resume incompatible WebShop states: {mismatches}")


def validate_pool(pool: list[dict[str, Any]], server: Any) -> None:
    for item in pool:
        goal_index = int(item["goal_index"])
        goal = server.goals[goal_index]
        expected = {
            "asin": str(goal["asin"]),
            "category": str(goal["category"]),
            "instruction": str(goal["instruction_text"]),
        }
        actual = {key: str(item[key]) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"WebShop frozen goal {goal_index} drift: {actual} != {expected}")


def run_batch(envs: list[Any], batch: list[dict[str, Any]],
              args: argparse.Namespace, client: CompletionClient,
              projection: Any, templates: Any, parse_action: Any) -> list[dict[str, Any]]:
    episodes = [
        initialize_episode(
            env,
            ordinal=offset,
            goal_index=int(item["goal_index"]),
            seed=args.seed + int(item["goal_index"]),
        )
        for offset, (env, item) in enumerate(zip(envs, batch))
    ]
    records: dict[str, list[dict[str, Any]]] = {
        str(item["goal_index"]): [] for item in batch
    }

    with ThreadPoolExecutor(max_workers=min(args.request_workers, len(episodes))) as executor:
        for step_index in range(args.max_steps):
            active = [episode for episode in episodes if not episode.done]
            if not active:
                break
            environment_prompts = [
                build_prompt(episode, templates, args.history_length)
                for episode in active
            ]
            serialized_prompts = [
                serialize_qwen_user_prompt(prompt) for prompt in environment_prompts
            ]
            futures = [
                executor.submit(client.generate, prompt, episode.seed + step_index)
                for episode, prompt in zip(active, serialized_prompts)
            ]
            responses = [future.result() for future in futures]
            actions, valids = projection(responses.copy())

            for episode, prompt, response, action, valid in zip(
                    active, serialized_prompts, responses, actions, valids):
                previous_observation = episode.observation
                previous_actions = list(episode.actions)
                previous_available = dict(episode.raw_available)
                admissible = action_is_admissible(
                    action, previous_available, parse_action)
                raw_observation, score, done, _ = episode.env.step(action)
                episode.history.append((previous_observation, action))
                episode.done = bool(done)
                episode.task_score = float(score) if done else 0.0
                episode.won = bool(done and episode.task_score == 1.0)
                records[str(episode.goal_index)].append({
                    "domain": "webshop",
                    "episode_id": str(episode.goal_index),
                    "goal_index": episode.goal_index,
                    "seed": episode.seed,
                    "target_asin": episode.target_asin,
                    "category": episode.category,
                    "instruction": episode.task,
                    "turn": len(episode.history) - 1,
                    "prompt": prompt,
                    "response": response,
                    "observation": previous_observation,
                    "admissible_actions": previous_actions,
                    "projected_action": action,
                    "format_valid": bool(valid),
                    "action_admissible": admissible,
                    "done": episode.done,
                    "task_score": episode.task_score,
                    "won": episode.won,
                    "trajectory_terminal": False,
                })
                if not episode.done:
                    episode.observation = format_observation(
                        raw_observation, episode.task)
                    episode.raw_available = episode.env.get_available_actions()
                    episode.actions = format_available_actions(
                        episode.raw_available)

    output = []
    for item in batch:
        episode_id = str(item["goal_index"])
        episode_records = records[episode_id]
        if not episode_records:
            raise RuntimeError(f"WebShop goal {episode_id}: rollout produced no states")
        episode_records[-1]["trajectory_terminal"] = True
        output.extend(episode_records)
    return output


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
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--request-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.request_workers <= 0 or args.max_steps <= 0:
        parser.error("batch size, workers, and max steps must be positive")
    args.model = args.model.resolve()
    args.model_path = args.model
    args.data_path = args.data_path.resolve()
    args.index_path = args.index_path.resolve()
    args.output = args.output.resolve()
    return args


def main() -> None:
    args = parse_args()
    verify_sources()
    validate_paths(args)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    if Path(split["data_root"]).resolve() != args.data_path:
        raise RuntimeError("WebShop split data_root does not match --data-path")
    if int(split["seed"]) != args.seed:
        raise RuntimeError("WebShop split seed does not match --seed")
    config = json.loads((EXPERIMENT / "config/experiment.json").read_text())
    rollout_episodes = int(config["data"]["rollout_episodes"]["webshop"])
    categories = sorted({str(item["category"]) for item in split["optimization_goals"]})
    if rollout_episodes % len(categories):
        raise RuntimeError("WebShop rollout budget must divide across categories")
    per_category = rollout_episodes // len(categories)
    pool = []
    for category in categories:
        seed = f'{args.seed}:{args.stage}:{args.round}:webshop:{category}'
        goals = [item for item in split["optimization_goals"]
                 if str(item["category"]) == category]
        goals.sort(key=lambda item: hashlib.sha256(
            f'{seed}:{item["goal_index"]}'.encode()).hexdigest())
        pool.extend(goals[:per_category])
    goal_ids = [str(item["goal_index"]) for item in pool]
    if len(set(goal_ids)) != len(goal_ids):
        raise RuntimeError("WebShop optimization pool contains duplicate goals")
    train_start, train_end = split["source_goal_range"]["train"]
    if any(not int(train_start) <= int(goal_id) <= int(train_end)
           for goal_id in goal_ids):
        raise RuntimeError("WebShop optimization pool contains non-train goals")

    summary_path = args.output.with_suffix(".summary.json")
    rows = read_jsonl(args.output) if args.resume else []
    if rows:
        validate_resume_summary(summary_path, args, pool)
    completed = completed_episode_ids(rows)
    if not completed <= set(goal_ids):
        raise RuntimeError("saved WebShop states contain goals outside frozen pool")

    projection = load_module(
        "_pinned_webshop_projection_rollout", PROJECTION_PATH).webshop_projection
    templates = load_module("_pinned_webshop_prompts_rollout", PROMPT_PATH)
    client = CompletionClient(
        args.api_base, args.served_model_name, args.request_timeout,
        args.max_tokens, args.temperature, args.top_p)
    client.check()
    envs, parse_action, server = create_environment_pool(args)
    try:
        validate_pool(pool, server)
        pending = [item for item in pool if str(item["goal_index"]) not in completed]
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            rows.extend(run_batch(
                envs[:len(batch)], batch, args, client,
                projection, templates, parse_action))
            atomic_write_jsonl(args.output, rows)
            write_summary(summary_path, args, rows, pool)
            completed = completed_episode_ids(rows)
            print(
                f"WebShop states: {len(completed)}/{len(pool)} episodes, "
                f"{len(rows)} states")
        write_summary(summary_path, args, rows, pool)
        print(summary_path.read_text())
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
