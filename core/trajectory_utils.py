"""Deterministic helpers for grouping trajectories and selecting state views."""

from __future__ import annotations

import hashlib
from collections import defaultdict


def trajectory_key(domain: str, row: dict) -> tuple:
    del domain
    final = bool(row.get("final", False))
    return (int(row.get("turn", 0)), int(row.get("step", 0)),
            int(row.get("sample", 0)), 1 if final else 0)


def group_trajectories(domain: str, rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[f"{domain}:{row['episode_id']}"].append(row)
    return {
        trajectory_id: sorted(items, key=lambda row: trajectory_key(domain, row))
        for trajectory_id, items in grouped.items()
    }


def stable_order(seed: int, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def choose_states(groups: dict[str, list[dict]], seed: int,
                  budget: int) -> list[dict]:
    """Choose exactly ``budget`` states while keeping episodes near-equally weighted."""
    if budget <= 0 or not groups:
        raise ValueError("state budget and trajectory pool must be positive")
    available = sum(len(rows) for rows in groups.values())
    if available < budget:
        raise ValueError(f"only {available} states, need {budget}")

    ids = sorted(groups, key=lambda value: stable_order(seed, value))
    ordered = {
        trajectory_id: sorted(
            rows,
            key=lambda row: stable_order(
                seed, f"{trajectory_id}:{row['trajectory_position']}"),
        )
        for trajectory_id, rows in groups.items()
    }
    quota, extra = divmod(budget, len(ids))
    selected: list[dict] = []
    used = {trajectory_id: min(quota, len(ordered[trajectory_id]))
            for trajectory_id in ids}
    for trajectory_id in ids:
        selected.extend(ordered[trajectory_id][:used[trajectory_id]])

    for trajectory_id in ids[:extra]:
        if used[trajectory_id] < len(ordered[trajectory_id]):
            selected.append(ordered[trajectory_id][used[trajectory_id]])
            used[trajectory_id] += 1

    while len(selected) < budget:
        progressed = False
        for trajectory_id in ids:
            if used[trajectory_id] >= len(ordered[trajectory_id]):
                continue
            selected.append(ordered[trajectory_id][used[trajectory_id]])
            used[trajectory_id] += 1
            progressed = True
            if len(selected) == budget:
                break
        if not progressed:
            raise RuntimeError("state selection exhausted before reaching its budget")
    return selected


def choose_probe_trajectories(domain: str, groups: dict[str, list[dict]],
                              seed: int, budget: int) -> list[str]:
    """Choose complete trajectories used only by endpoints and probes."""
    if budget <= 0 or len(groups) < budget:
        raise ValueError(f"{domain}: need {budget} probe trajectories, got {len(groups)}")
    ordered = sorted(groups, key=lambda value: stable_order(seed, value))
    if domain not in {"alfworld", "webshop"}:
        return ordered[:budget]

    field = "task_type" if domain == "alfworld" else "category"
    by_category: dict[str, list[str]] = defaultdict(list)
    for trajectory_id in ordered:
        by_category[str(groups[trajectory_id][0].get(field, ""))].append(trajectory_id)
    categories = sorted(by_category)
    base, extra = divmod(budget, len(categories))
    offset = seed % len(categories)
    extra_categories = {categories[(offset + index) % len(categories)]
                        for index in range(extra)}
    chosen = []
    for category in categories:
        count = base + (1 if category in extra_categories else 0)
        if len(by_category[category]) < count:
            raise ValueError(f"{domain}/{category}: need {count} probe trajectories")
        chosen.extend(by_category[category][:count])
    return chosen


def annotate_trajectory(rows: list[dict], trajectory_id: str, domain: str,
                        role: str, source: str, direction: str) -> list[dict]:
    annotated = []
    for position, original in enumerate(rows):
        row = dict(original)
        row.update({
            "domain": domain,
            "role": role,
            "source": source,
            "kl_direction": direction,
            "trajectory_id": trajectory_id,
            "episode_id": str(row["episode_id"]),
            "trajectory_position": position,
        })
        annotated.append(row)
    return annotated


def assert_contiguous(rows: list[dict]) -> None:
    positions = [int(row["trajectory_position"]) for row in rows]
    if not rows or positions != list(range(len(rows))):
        raise ValueError(f"trajectory positions are not contiguous: {positions}")
