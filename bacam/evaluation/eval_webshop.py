#!/usr/bin/env python3
"""Evaluate Qwen agents on the GiGPO WebShop task distribution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


EXPERIMENT = Path(__file__).resolve().parents[2]
from bacam.paths import PATHS, load_experiment

VERL_AGENT = Path(PATHS["BACAM_VERL_AGENT_ROOT"])
MODEL_CARD_COMMIT = "35b3da38293993f9bf4f7873dfb3262a361e956c"
DEFAULT_DATA = Path(PATHS["BACAM_DATA_ROOT"]) / "webshop"
DEFAULT_INDEX = DEFAULT_DATA / "search_engine_1k/indexes"
DEFAULT_MODEL = Path(PATHS["BACAM_MODEL_ROOT"]) / "GiGPO-Qwen2.5-7B-Instruct-WebShop"
DEFAULT_OUTPUT = EXPERIMENT / "artifacts/eval/webshop/expert_test_vllm.json"
WEBSHOP_ROOT = (
    VERL_AGENT / "agent_system/environments/env_package/webshop/webshop")
PROJECTION_PATH = (
    VERL_AGENT / "agent_system/environments/env_package/webshop/projection.py")
PROMPT_PATH = VERL_AGENT / "agent_system/environments/prompts/webshop.py"
MEMORY_PATH = VERL_AGENT / "agent_system/memory/memory.py"
TEXT_ENV_PATH = WEBSHOP_ROOT / "web_agent_site/envs/web_agent_text_env.py"
ENGINE_PATH = WEBSHOP_ROOT / "web_agent_site/engine/engine.py"
GOAL_PATH = WEBSHOP_ROOT / "web_agent_site/engine/goal.py"

EXPECTED_SOURCE_SHA256 = {
    PROJECTION_PATH: "0808d4643459acf2a4c496eab3fc8735025c440f28cc517066355a1f5fbc048f",
    PROMPT_PATH: "c7411b8a8fe1f8d587a5585a57ce3ae85ab87bbd7a021f2e39c03f97e214b738",
    MEMORY_PATH: "d9940bf9d49442f76667b80aff7e80b7c93f37bef015b06641e60229bb3f0d9c",
    TEXT_ENV_PATH: "a2a2cfe3e7ef493857ba68c28fd6d12ec012e0f8cb31a050112f6466084e3f55",
    ENGINE_PATH: "86bb46b593d3672da3a9fbe216f1d34a34a7a915f7ad31a101bdcc7f2a62ca4b",
    GOAL_PATH: "9703c8583244e2041182eb14856186b998145ab16ce2ba9aef8cb680877c8ed5",
}
EXPECTED_DATA_SHA256 = {
    "items_shuffle_1000.json": "30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269",
    "items_ins_v2_1000.json": "f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f",
    "items_human_ins.json": "cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3",
}
from bacam.evaluation.eval_alfworld_expert import ChatClient, atomic_write_json  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_sources() -> None:
    for path, expected in EXPECTED_SOURCE_SHA256.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"pinned verl-agent source drift for {path}: {actual} != {expected}")


def format_available_actions(available: dict[str, Any]) -> list[str]:
    if set(available) != {"has_search_bar", "clickables"}:
        raise RuntimeError(f"unexpected WebShop available action keys: {available.keys()}")
    actions = []
    if available["has_search_bar"]:
        actions.append("search[<your query>]")
    actions.extend(f"click[{value}]" for value in available["clickables"])
    return actions


def extract_task(raw_observation: str) -> str:
    parts = raw_observation.split(" [SEP] ")
    if len(parts) < 3 or parts[1] != "Instruction:":
        raise RuntimeError(f"unexpected initial WebShop observation: {raw_observation[:300]}")
    return parts[2]


def format_observation(raw_observation: str, task: str) -> str:
    parts = raw_observation.split(" [SEP] ")
    try:
        index = parts.index(task)
    except ValueError:
        return raw_observation
    return " [SEP] ".join(f"'{part}'" for part in parts[index + 1:])


def memory_context(history: list[tuple[str, str]], length: int) -> tuple[str, int]:
    recent = history[-length:]
    start = len(history) - len(recent)
    lines = [
        f"[Observation {start + offset + 1}: '{observation}', "
        f"Action {start + offset + 1}: '{action}']"
        for offset, (observation, action) in enumerate(recent)
    ]
    return "\n".join(lines), len(recent)


def build_prompt(episode: "Episode", templates: Any, history_length: int) -> str:
    available = "\n".join(f"'{action}'," for action in episode.actions)
    if not episode.history or history_length <= 0:
        return templates.WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=episode.task,
            current_observation=episode.observation,
            available_actions=available,
        )
    context, valid_length = memory_context(episode.history, history_length)
    prompt = templates.WEBSHOP_TEMPLATE.format(
        task_description=episode.task,
        step_count=len(episode.history),
        history_length=valid_length,
        action_history=context,
        current_step=len(episode.history) + 1,
        current_observation=episode.observation,
        available_actions=available,
    )
    if len(prompt) > 13000:
        prompt = templates.WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=episode.task,
            current_observation=episode.observation,
            available_actions=available,
        )
    return prompt


def action_is_admissible(action: str, available: dict[str, Any],
                         parse_action: Callable[[str], tuple[Any, Any]]) -> bool:
    name, argument = parse_action(action)
    if argument is not None:
        argument = argument.lower()
    if name == "search":
        return bool(available["has_search_bar"] and argument)
    if name == "click":
        return argument in available["clickables"]
    return False


@dataclass
class Episode:
    ordinal: int
    goal_index: int
    seed: int
    env: Any
    task: str
    category: str
    target_asin: str
    observation: str
    raw_available: dict[str, Any]
    actions: list[str]
    history: list[tuple[str, str]] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    task_score: float = 0.0
    won: bool = False
    final_info: dict[str, Any] = field(default_factory=dict)


def initialize_episode(env: Any, ordinal: int, goal_index: int,
                       seed: int) -> Episode:
    raw_observation, _ = env.reset(session=goal_index)
    task = extract_task(raw_observation)
    goal = env.server.goals[goal_index]
    if task != goal["instruction_text"]:
        raise RuntimeError(f"goal {goal_index}: rendered instruction drift")
    raw_available = env.get_available_actions()
    return Episode(
        ordinal=ordinal,
        goal_index=goal_index,
        seed=seed,
        env=env,
        task=task,
        category=str(goal["category"]),
        target_asin=str(goal["asin"]),
        observation=format_observation(raw_observation, task),
        raw_available=raw_available,
        actions=format_available_actions(raw_available),
    )


def run_batch(envs: list[Any], start_ordinal: int, goal_indices: list[int],
              args: argparse.Namespace, client: ChatClient, projection: Any,
              templates: Any, parse_action: Any) -> list[dict[str, Any]]:
    episodes = [
        initialize_episode(env, start_ordinal + offset, goal_index,
                           args.seed + goal_index)
        for offset, (env, goal_index) in enumerate(zip(envs, goal_indices))
    ]
    with ThreadPoolExecutor(max_workers=min(args.request_workers, len(episodes))) as executor:
        for step_index in range(args.max_steps):
            active = [episode for episode in episodes if not episode.done]
            if not active:
                break
            prompts = [build_prompt(episode, templates, args.history_length)
                       for episode in active]
            futures = [
                executor.submit(client.generate, prompt, episode.seed + step_index)
                for episode, prompt in zip(active, prompts)
            ]
            responses = [future.result() for future in futures]
            actions, valids = projection(responses.copy())
            for episode, response, action, valid in zip(
                    active, responses, actions, valids):
                previous_observation = episode.observation
                previous_actions = list(episode.actions)
                previous_available = dict(episode.raw_available)
                session_id = episode.env.session
                admissible = action_is_admissible(
                    action, previous_available, parse_action)
                raw_observation, score, done, _ = episode.env.step(action)
                episode.history.append((previous_observation, action))
                episode.trajectory.append({
                    "step": len(episode.history),
                    "observation": previous_observation,
                    "admissible_actions": previous_actions,
                    "response": response,
                    "action": action,
                    "format_valid": bool(valid),
                    "action_admissible": admissible,
                })
                episode.done = bool(done)
                episode.task_score = float(score) if done else 0.0
                episode.won = bool(done and episode.task_score == 1.0)
                if episode.done:
                    session = episode.env.server.user_sessions[session_id]
                    episode.final_info = {
                        "purchased_asin": session.get("asin"),
                        "purchased_options": session.get("options", {}),
                        "reward_components": session.get("verbose_info", {}),
                    }
                else:
                    episode.observation = format_observation(
                        raw_observation, episode.task)
                    episode.raw_available = episode.env.get_available_actions()
                    episode.actions = format_available_actions(
                        episode.raw_available)

    results = []
    for episode in episodes:
        results.append({
            "round": episode.ordinal + 1,
            "goal_index": episode.goal_index,
            "seed": episode.seed,
            "target_asin": episode.target_asin,
            "category": episode.category,
            "instruction": episode.task,
            "task_score": episode.task_score,
            "won": episode.won,
            "steps": len(episode.trajectory),
            "terminated": episode.done,
            "terminal_reason": "buy_now" if episode.done else "max_steps",
            "valid_format_actions": sum(
                bool(item["format_valid"]) for item in episode.trajectory),
            "admissible_actions_taken": sum(
                bool(item["action_admissible"]) for item in episode.trajectory),
            "final_info": episode.final_info,
            "trajectory": episode.trajectory,
        })
    return results


def build_summary(args: argparse.Namespace, episodes: list[dict[str, Any]],
                  started_at: str, data_hashes: dict[str, str],
                  source_hashes: dict[str, str]) -> dict[str, Any]:
    scores = [float(episode["task_score"]) for episode in episodes]
    successes = sum(bool(episode["won"]) for episode in episodes)
    rounds = [int(episode["steps"]) for episode in episodes]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_category[str(episode["category"])].append(episode)
    category_breakdown = {}
    for category, values in sorted(by_category.items()):
        category_breakdown[category] = {
            "episodes": len(values),
            "task_score": sum(float(item["task_score"]) for item in values) / len(values),
            "sr": sum(bool(item["won"]) for item in values) / len(values),
            "average_round": sum(int(item["steps"]) for item in values) / len(values),
        }
    return {
        "schema_version": 1,
        "status": "complete" if len(episodes) >= args.goal_count else "partial",
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "langfengQ/verl-agent",
            "model_card_commit": MODEL_CARD_COMMIT,
            "source_sha256": source_hashes,
            "data_sha256": data_hashes,
            "index_manifest": str(args.index_path.parent / "index_manifest.json"),
        },
        "model": {
            "path": str(args.model_path),
            "served_name": args.served_model_name,
            "inference_backend": args.inference_backend,
        },
        "evaluation": {
            "dataset": "WebShop 1k products + synthetic goals",
            "split": "test",
            "goal_start": args.goal_start,
            "goals_requested": args.goal_count,
            "goals_completed": len(episodes),
            "task_score": sum(scores) / len(scores) if scores else None,
            "successes": successes,
            "sr": successes / len(episodes) if episodes else None,
            "average_round": sum(rounds) / len(rounds) if rounds else None,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "history_length": args.history_length,
            "generation": {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            "category_breakdown": category_breakdown,
        },
        "episodes": episodes,
    }


def load_resume(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    if not args.resume or not args.output.is_file():
        return [], datetime.now(timezone.utc).isoformat()
    previous = json.loads(args.output.read_text())
    evaluation = previous.get("evaluation", {})
    expected = {
        "split": "test",
        "goal_start": args.goal_start,
        "goals_requested": args.goal_count,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "history_length": args.history_length,
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }
    mismatches = {key: (evaluation.get(key), value)
                  for key, value in expected.items()
                  if evaluation.get(key) != value}
    if previous.get("model", {}).get("path") != str(args.model_path):
        mismatches["model.path"] = (
            previous.get("model", {}).get("path"), str(args.model_path))
    if mismatches:
        raise RuntimeError(f"cannot resume incompatible WebShop result: {mismatches}")
    episodes = list(previous.get("episodes", []))
    expected_indices = list(range(args.goal_start, args.goal_start + len(episodes)))
    if [int(item["goal_index"]) for item in episodes] != expected_indices:
        raise RuntimeError("WebShop resume episodes are not a contiguous test prefix")
    return episodes, previous.get("started_at", datetime.now(timezone.utc).isoformat())


def validate_paths(args: argparse.Namespace) -> dict[str, str]:
    for name in ("config.json", "model.safetensors.index.json",
                 "tokenizer_config.json"):
        path = args.model_path / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    actual_hashes = {}
    for name, expected in EXPECTED_DATA_SHA256.items():
        path = args.data_path / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hashes[name] = sha256(path)
        if actual_hashes[name] != expected:
            raise RuntimeError(f"WebShop data drift for {path}")
    manifest_path = args.index_path.parent / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("input_sha256") != actual_hashes:
        raise RuntimeError("WebShop index inputs do not match evaluation data")
    if int(manifest.get("indexed_documents", 0)) != 1000:
        raise RuntimeError("WebShop 1k index is incomplete")
    return actual_hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-start", type=int, default=0)
    parser.add_argument("--goal-count", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--request-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    port = load_experiment(EXPERIMENT / "configs/experiment.json")["execution"]["ports"]["webshop_eval"]
    parser.add_argument("--api-base", default=f"http://127.0.0.1:{port}/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--served-model-name", default="webshop-expert")
    parser.add_argument("--inference-backend", default="vllm-0.11.0")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.goal_start < 0 or args.goal_count <= 0:
        parser.error("invalid goal range")
    if args.goal_start + args.goal_count > 500:
        parser.error("formal WebShop test split is goal index 0..499")
    if args.batch_size <= 0 or args.request_workers <= 0:
        parser.error("batch size and request workers must be positive")
    if args.max_steps <= 0 or args.history_length < 0 or args.max_tokens <= 0:
        parser.error("invalid step/history/token setting")
    args.model_path = args.model_path.resolve()
    args.data_path = args.data_path.resolve()
    args.index_path = args.index_path.resolve()
    args.output = args.output.resolve()
    return args


def create_environment_pool(args: argparse.Namespace) -> tuple[list[Any], Any, Any]:
    sys.path.insert(0, str(WEBSHOP_ROOT))
    from pyserini.search.lucene import LuceneSearcher
    from web_agent_site.engine import engine

    engine.HUMAN_ATTR_PATH = str(args.data_path / "items_human_ins.json")

    def external_index(num_products: int | None = None) -> Any:
        if num_products is not None:
            raise RuntimeError("WebShop data is already prefiltered")
        return LuceneSearcher(str(args.index_path))

    engine.init_search_engine = external_index
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    loader = WebAgentTextEnv(
        observation_mode="text",
        file_path=str(args.data_path / "items_shuffle_1000.json"),
        attr_path=str(args.data_path / "items_ins_v2_1000.json"),
        human_goals=False,
        num_products=None,
        seed=args.seed,
    )
    envs = [loader]
    for offset in range(1, args.batch_size):
        envs.append(WebAgentTextEnv(
            observation_mode="text", server=loader.server,
            seed=args.seed + offset))
    return envs, engine.parse_action, loader.server


def main() -> None:
    args = parse_args()
    verify_sources()
    data_hashes = validate_paths(args)
    source_hashes = {str(path): expected
                     for path, expected in EXPECTED_SOURCE_SHA256.items()}
    episodes, started_at = load_resume(args)
    if len(episodes) >= args.goal_count:
        print(f"Already complete: {args.output}")
        return

    projection = load_module("_pinned_webshop_projection", PROJECTION_PATH).webshop_projection
    templates = load_module("_pinned_webshop_prompts", PROMPT_PATH)
    client = ChatClient(
        args.api_base, args.api_key, args.served_model_name,
        args.max_tokens, args.temperature, args.top_p, args.request_timeout)
    client.check()
    envs, parse_action, _ = create_environment_pool(args)
    try:
        completed = len(episodes)
        while completed < args.goal_count:
            count = min(args.batch_size, args.goal_count - completed)
            goal_indices = list(range(
                args.goal_start + completed, args.goal_start + completed + count))
            episodes.extend(run_batch(
                envs[:count], completed, goal_indices, args, client,
                projection, templates, parse_action))
            completed = len(episodes)
            summary = build_summary(
                args, episodes, started_at, data_hashes, source_hashes)
            atomic_write_json(args.output, summary)
            metrics = summary["evaluation"]
            print(
                f"WebShop {completed}/{args.goal_count}: "
                f"task_score={metrics['task_score']:.6f}, "
                f"SR={metrics['sr']:.6f}, "
                f"avg_round={metrics['average_round']:.6f}")
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
