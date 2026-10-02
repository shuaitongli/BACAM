"""Generate local dataset inputs without distributing experiment-specific IDs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from bacam.paths import PATHS, ROOT, load_experiment


ALFWORLD_TASKS = (
    "pick_and_place_simple", "pick_two_obj_and_place", "look_at_obj_in_light",
    "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
)


def ordered(values, seed: int, domain: str):
    return sorted(values, key=lambda value: hashlib.sha256(
        f"{seed}:{domain}:{value}".encode()).hexdigest())


def read_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        ids = [str(json.loads(line)["id"]) for line in handle if line.strip()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def search_split(args, config: dict) -> dict:
    source = Path(PATHS["BACAM_DATA_ROOT"]) / "flashrag_eval/musique/dev.jsonl"
    ids = ordered(read_ids(source), config["seed"], "search")
    if not int(config["data"]["rollout_episodes"]["search"]) <= args.search_pool_size < len(ids):
        raise ValueError("Search pool must cover the rollout count and leave held-out questions")
    return {"source": str(source), "train": ids[:args.search_pool_size],
            "eval": ids[args.search_pool_size:]}


def tool_split(args, config: dict) -> dict:
    source = Path(PATHS["BACAM_BFCL_ROOT"]) / "bfcl_eval/data/BFCL_v3_multi_turn_base.json"
    ids = ordered(read_ids(source), config["seed"], "tool")
    held_out = 100  # Required by the held-out result collector.
    train = ids[held_out:]
    if len(train) < int(config["data"]["rollout_episodes"]["tool"]):
        raise ValueError("BFCL data must leave enough training cases after holding out 100")
    return {"categories": {"multi_turn_base": {"train": train, "eval": ids[:held_out]}}}


def alfworld_split(args, config: dict) -> dict:
    import yaml

    root = (Path(PATHS["BACAM_DATA_ROOT"]) / "alfworld").resolve()
    source_config = Path(PATHS["BACAM_VERL_AGENT_ROOT"]) / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
    with source_config.open(encoding="utf-8") as handle:
        train_dir = yaml.safe_load(handle)["dataset"]["data_path"]
    train_dir = Path(str(train_dir).replace("$ALFWORLD_DATA", str(root)))
    games = defaultdict(list)
    for path in sorted(train_dir.rglob("game.tw-pddl")):
        identity = path.relative_to(root).as_posix()
        for task in ALFWORLD_TASKS:
            if task in identity:
                games[task].append(identity)
                break
    rollout = int(config["data"]["rollout_episodes"]["alfworld"])
    if rollout % len(ALFWORLD_TASKS) or args.alfworld_per_task < rollout // len(ALFWORLD_TASKS):
        raise ValueError("ALFWorld pool must cover the per-task rollout budget")
    selected = {}
    for task in ALFWORLD_TASKS:
        if len(games[task]) < args.alfworld_per_task:
            raise ValueError(f"Insufficient ALFWorld training games for {task}")
        selected[task] = ordered(games[task], config["seed"], task)[:args.alfworld_per_task]
    return {"data_root": str(root), "task_types": list(ALFWORLD_TASKS),
            "optimization_episodes": sum(map(len, selected.values())),
            "optimization_by_task": selected}


def webshop_split(args, config: dict) -> dict:
    from bacam.evaluation.eval_webshop import create_environment_pool

    root = (Path(PATHS["BACAM_DATA_ROOT"]) / "webshop").resolve()
    envs, _, server = create_environment_pool(SimpleNamespace(
        data_path=root, index_path=root / "search_engine_1k/indexes",
        batch_size=1, seed=int(config["seed"])))
    try:
        groups = defaultdict(list)
        # Evaluation always uses goals 0..499.
        for index, goal in enumerate(server.goals[500:], start=500):
            groups[str(goal["category"])].append({
                "goal_index": index, "asin": str(goal["asin"]),
                "category": str(goal["category"]),
                "instruction": str(goal["instruction_text"]),
            })
        categories = sorted(groups)
        rollout = int(config["data"]["rollout_episodes"]["webshop"])
        if not categories or rollout % len(categories) or args.webshop_pool_size < rollout:
            raise ValueError("WebShop pool must cover a category-balanced rollout budget")
        per_category, extra = divmod(args.webshop_pool_size, len(categories))
        selected = []
        for position, category in enumerate(categories):
            count = per_category + int(position < extra)
            if len(groups[category]) < count:
                raise ValueError(f"Insufficient WebShop training goals for {category}")
            by_id = {str(item["goal_index"]): item for item in groups[category]}
            selected.extend(by_id[value] for value in ordered(
                by_id, config["seed"], category)[:count])
        return {"data_root": str(root), "seed": int(config["seed"]),
                "optimization_episodes": len(selected),
                "source_goal_range": {"train": [500, len(server.goals) - 1]},
                "optimization_goals": selected}
    finally:
        for env in envs:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("search", "tool", "alfworld", "webshop"), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--search-pool-size", type=int, default=400)
    parser.add_argument("--alfworld-per-task", type=int, default=24)
    parser.add_argument("--webshop-pool-size", type=int, default=128)
    args = parser.parse_args()
    if min(args.search_pool_size, args.alfworld_per_task, args.webshop_pool_size) <= 0:
        parser.error("Pool sizes must be positive")
    path = args.output_dir / f"{'bfcl' if args.domain == 'tool' else args.domain}_split.json"
    if path.exists():
        parser.error(f"Split already exists: {path}; refusing to replace it")
    config = load_experiment(ROOT / "configs/experiment.json")
    builders = {"search": search_split, "tool": tool_split,
                "alfworld": alfworld_split, "webshop": webshop_split}
    split = builders[args.domain](args, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"{args.domain}: generated {path}")


if __name__ == "__main__":
    main()
