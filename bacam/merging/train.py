#!/usr/bin/env python3
"""FSDP gate training on independent states with behavior-aware KL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from bacam.merging.gate_ops import (build_gate, capture_gate_gradient, cosine_lr, gate_step,
                      gate_summary, update_plasticity_budgets, wrap,
                      write_theta)  # noqa: E402
from bacam.merging.loss import (behavior_aware_state_loss, compressed_topk_kl,
                     raw_kl_drift)  # noqa: E402
from bacam.merging.merge_stage import MergeStage  # noqa: E402


Trajectory = tuple[str, list[Path]]


def load_cache_manifest(stage: MergeStage) -> tuple[
        dict, dict[str, list[Path]], dict[str, list[Trajectory]]]:
    state_specs = json.loads((stage.state_dir / "train_manifest.json").read_text())["streams"]
    train_states: dict[str, list[Path]] = {}
    probe_trajectories: dict[str, list[Trajectory]] = {}
    for stream, spec in state_specs.items():
        if spec.get("source") != "current":
            raise RuntimeError(f"{stream}: BACAM requires candidate-generated trajectories")
        manifest_path = stage.cache_dir / stream / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{stream}: missing cache manifest {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("distribution_format") != "topk_plus_other_v1":
            raise RuntimeError(f"{stream}: wrong cache format")
        if int(manifest.get("top_k", -1)) != 32:
            raise RuntimeError(f"{stream}: Top-K must be 32")
        if manifest.get("training_aggregation") != "trajectory_equal":
            raise RuntimeError(f"{stream}: cache is not trajectory-equal")
        expected_direction = "forward" if spec["role"] == "new" else "reverse"
        if manifest.get("role") != spec["role"] or manifest.get("kl_direction", expected_direction) != expected_direction:
            raise RuntimeError(f"{stream}: cache role or KL direction does not match state manifest")
        selected = []
        grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
        for record in manifest["records"]:
            path = Path(record["path"])
            if not path.exists():
                raise FileNotFoundError(f"{stream}: missing cache shard {path}")
            if bool(record.get("train_selected")):
                selected.append(path)
            if bool(record.get("probe_selected")):
                grouped[str(record["trajectory_id"])].append(
                    (int(record["trajectory_position"]), path))
        ordered = []
        for trajectory_id, records in grouped.items():
            records.sort(key=lambda item: item[0])
            positions = [position for position, _ in records]
            if positions != list(range(len(records))):
                raise RuntimeError(f"{stream}/{trajectory_id}: non-contiguous cache positions")
            ordered.append((trajectory_id, [path for _, path in records]))
        ordered.sort(key=lambda item: item[0])
        expected_states = int(stage.config["data"]["states_per_stream"])
        expected_probe = int(stage.config["endpoint"]["trajectories_per_source"])
        if len(selected) != expected_states or len(ordered) != expected_probe:
            raise RuntimeError(f"{stream}: expected {expected_states} train states and "
                               f"{expected_probe} probe trajectories, got "
                               f"{len(selected)} and {len(ordered)}")
        weight_sum = sum(float(record.get("trajectory_weight", 0.0))
                         for record in manifest["records"])
        if abs(weight_sum - expected_states) > 1e-5:
            raise RuntimeError(f"{stream}: invalid trajectory weight sum {weight_sum}")
        train_states[stream] = selected
        probe_trajectories[stream] = ordered
    return state_specs, train_states, probe_trajectories


def token_kl(student: FSDP, payload: dict, device: torch.device) -> torch.Tensor:
    input_ids = payload["input_ids"].to(device=device, dtype=torch.long).unsqueeze(0)
    start, length = int(payload["response_start"]), int(payload["response_length"])
    positions = torch.arange(start - 1, start + length - 1, device=device)
    logits = student(input_ids=input_ids, use_cache=False, return_dict=True,
                     logits_to_keep=positions).logits[0]
    values = compressed_topk_kl(
        logits,
        payload["topk_ids"].to(device=device, dtype=torch.long),
        payload["topk_raw_probs"].to(device=device, dtype=torch.float32),
        payload["other_prob"].to(device=device, dtype=torch.float32),
        payload["kl_direction"],
        float(payload["temperature"]),
    )
    if not torch.isfinite(values).all():
        raise FloatingPointError("BACAM token KL is NaN or Inf")
    return values


def state_kl(student: FSDP, payload: dict, device: torch.device) -> torch.Tensor:
    return behavior_aware_state_loss(
        token_kl(student, payload, device), payload["critical_token_mask"])


@torch.no_grad()
def set_gate(gate: dict, value: float) -> None:
    for item in gate.values():
        item["g"].fill_(value)


@torch.no_grad()
def endpoint_values(student: FSDP,
                    trajectories: dict[str, list[Trajectory]],
                    device: torch.device, limit: int) -> dict[str, dict[str, object]]:
    rank, world = dist.get_rank(), dist.get_world_size()
    output = {}
    for stream, all_trajectories in trajectories.items():
        if len(all_trajectories) < limit:
            raise RuntimeError(f"{stream}: only {len(all_trajectories)} endpoint trajectories")
        selected = all_trajectories[:limit]
        local_trajectories = selected[rank::world]
        local = [(trajectory_id, path)
                 for trajectory_id, trajectory in local_trajectories
                 for path in trajectory]
        local_max = torch.tensor([len(local)], dtype=torch.long, device=device)
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
        padding_path = local[0][1]
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for position in range(int(local_max.item())):
            trajectory_id, path = (local[position] if position < len(local)
                                   else ("", padding_path))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            result = float(state_kl(student, payload, device))
            if position < len(local):
                sums[trajectory_id] += result
                counts[trajectory_id] += 1
        values = [sums[trajectory_id] / counts[trajectory_id]
                  for trajectory_id, _ in local_trajectories]
        packed = torch.tensor([sum(values), sum(v * v for v in values), len(values),
                               sum(counts.values())],
                              dtype=torch.float64, device=device)
        dist.all_reduce(packed)
        count = int(packed[2].item())
        mean = float(packed[0].item() / max(count, 1))
        variance = max(0.0, float(packed[1].item() / max(count, 1)) - mean * mean)
        output[stream] = {"mean": mean, "variance": variance, "n": count,
                          "states": int(packed[3].item())}
    return output


def measure_endpoints(student: FSDP, gate: dict, state_specs: dict,
                      trajectories: dict[str, list[Trajectory]],
                      device: torch.device, limit: int) -> dict:
    student.eval()
    set_gate(gate, 0.0)
    write_theta(student, gate)
    at_old = endpoint_values(student, trajectories, device, limit)
    set_gate(gate, 1.0)
    write_theta(student, gate)
    at_new = endpoint_values(student, trajectories, device, limit)

    endpoints = {}
    for stream, spec in state_specs.items():
        old_mean, new_mean = at_old[stream]["mean"], at_new[stream]["mean"]
        if spec["role"] == "old":
            scale = new_mean - old_mean
        else:
            scale = old_mean - new_mean
        endpoints[stream] = {
            "domain": spec["domain"], "role": spec["role"],
            "source": spec["source"], "kl_direction": spec["kl_direction"],
            "at_old": old_mean, "at_new": new_mean,
            "at_old_variance": at_old[stream]["variance"],
            "at_new_variance": at_new[stream]["variance"],
            "scale": scale, "n": at_old[stream]["n"],
            "endpoint_states": at_old[stream]["states"],
            "endpoint_trajectories": limit,
            "aggregation": "trajectory_equal",
        }
    student.train()
    return endpoints


def save_gate(stage: MergeStage, gate: dict, dual: dict, world: int, rank: int) -> None:
    target = stage.artifact_dir / "gate_state"
    target.mkdir(parents=True, exist_ok=True)
    torch.save({name: item["g"] for name, item in gate.items()}, target / f"rank{rank:02d}.pt")
    dist.barrier()
    if rank == 0:
        config_sha = hashlib.sha256(stage.config_path.read_bytes()).hexdigest()
        (target / "metadata.json").write_text(json.dumps({
            "method_version": stage.config["method_version"],
            "stage": stage.name, "round": stage.round, "world_size": world,
            "config_sha256": config_sha, "dual": dual,
        }, indent=2) + "\n")
    dist.barrier()


def gate_constraint_names(state_specs: dict) -> list[str]:
    return sorted(name for name, spec in state_specs.items() if spec["role"] == "old")


def restore_gate(stage: MergeStage, gate: dict, rank: int, world: int,
                 state_specs: dict) -> dict:
    names = gate_constraint_names(state_specs)
    if stage.previous_artifact_dir is None:
        set_gate(gate, 0.0)
        return {name: float(stage.config["trust_region"]["dual_init"]) for name in names}
    root = stage.previous_artifact_dir / "gate_state"
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("method_version") != stage.config["method_version"]:
        raise RuntimeError("checkpoint method version does not match BACAM")
    if int(metadata["world_size"]) != world:
        raise RuntimeError("BACAM gate resume requires the same world size")
    saved = torch.load(root / f"rank{rank:02d}.pt", map_location="cpu", weights_only=False)
    if set(saved) != set(gate):
        raise RuntimeError("saved gate parameter names do not match")
    for name, value in saved.items():
        if value.shape != gate[name]["g"].shape:
            raise RuntimeError(f"{name}: saved gate shape mismatch")
        gate[name]["g"].copy_(value)
    dual = {name: float(value) for name, value in metadata["dual"].items()}
    if sorted(dual) != names:
        raise RuntimeError(f"saved dual keys {sorted(dual)} != {names}")
    return dual


def export_dense(stage: MergeStage, student: FSDP, rank: int) -> None:
    policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(student, StateDictType.FULL_STATE_DICT, policy):
        full = student.state_dict()
    if rank == 0:
        stage.export_dir.mkdir(parents=True, exist_ok=True)
        raw = AutoModelForCausalLM.from_pretrained(
            stage.old_model, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        raw.load_state_dict(full, strict=True)
        raw.config.use_cache = True
        raw.save_pretrained(stage.export_dir, safe_serialization=True, max_shard_size="4GB")
        AutoTokenizer.from_pretrained(stage.old_model, local_files_only=True,
                                      trust_remote_code=True).save_pretrained(stage.export_dir)
    dist.barrier()


def reduce_running(running: dict[str, dict[str, float]], device: torch.device) -> dict[str, dict[str, float]]:
    names = sorted(running)
    packed = torch.zeros(4 * len(names), dtype=torch.float64, device=device)
    for index, name in enumerate(names):
        value = running[name]
        packed[4 * index] = value["state_kl"]
        packed[4 * index + 1] = value["trajectory_weight"]
        packed[4 * index + 2] = value["states"]
        packed[4 * index + 3] = value["tokens"]
    dist.all_reduce(packed)
    output = {}
    for index, name in enumerate(names):
        output[name] = {
            "state_kl": float(packed[4 * index].item() /
                               max(packed[4 * index + 1].item(), 1e-12)),
            "trajectory_weight": float(packed[4 * index + 1].item()),
            "states": int(packed[4 * index + 2].item()),
            "tokens": int(packed[4 * index + 3].item()),
        }
    return output


def train_state_batch(student: FSDP, batch: list[Path], weight: float,
                      device: torch.device, world: int,
                      states_per_step: int, running: dict[str, float]) -> None:
    rank = dist.get_rank()
    local_batch = batch[rank::world]
    if len(batch) != states_per_step or len(local_batch) * world != states_per_step:
        raise RuntimeError("state batch must divide evenly across FSDP ranks")
    for path in local_batch:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        value = state_kl(student, payload, device)
        trajectory_weight = float(payload["trajectory_weight"])
        if trajectory_weight <= 0:
            raise RuntimeError(f"training state has invalid trajectory weight {trajectory_weight}")
        (float(weight) * trajectory_weight * value /
         float(len(local_batch))).backward()
        running["state_kl"] += trajectory_weight * float(value.detach())
        running["trajectory_weight"] += trajectory_weight
        running["states"] += 1.0
        running["tokens"] += float(payload["response_length"])
        del payload, value


def backward_probe_stream(student: FSDP, trajectories: list[Trajectory],
                          device: torch.device, world: int, limit: int) -> None:
    if limit > len(trajectories):
        raise RuntimeError(f"probe needs {limit} trajectories")
    selected = trajectories[:limit]
    local_trajectories = selected[dist.get_rank()::world]
    local = [(path, float(world) / (limit * len(trajectory)))
             for _, trajectory in local_trajectories for path in trajectory]
    local_max = torch.tensor([len(local)], dtype=torch.long, device=device)
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
    padding_path = local[0][0]
    student.zero_grad(set_to_none=True)
    for position in range(int(local_max.item())):
        path, coefficient = (local[position] if position < len(local)
                             else (padding_path, 0.0))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        value = state_kl(student, payload, device)
        if position < len(local):
            (value * coefficient).backward()
        else:
            (value * 0.0).backward()
        del payload, value


def calibrate_plasticity_budgets(
        stage: MergeStage, student: FSDP, gate: dict, dual: dict,
        state_specs: dict, trajectories: dict[str, list[Trajectory]],
        device: torch.device, world: int, gate_lo: float,
        gate_hi: float) -> Path:
    cfg = stage.config["stability_plasticity"]
    limit = int(cfg["probe_trajectories_per_stream"])
    if cfg.get("probe_step") != "one_new_only_gate_step":
        raise RuntimeError("unsupported stability-plasticity probe step")
    new_streams = [name for name, spec in state_specs.items() if spec["role"] == "new"]
    old_streams = sorted(name for name, spec in state_specs.items() if spec["role"] == "old")
    if len(new_streams) != 1 or not old_streams:
        raise RuntimeError("stability-plasticity calibration requires one new and at least one old stream")

    new_stream = new_streams[0]
    backward_probe_stream(
        student, trajectories[new_stream], device, world, limit)
    plasticity_gradient = capture_gate_gradient(student, gate)
    probe_lr = cosine_lr(stage.gate_lr, stage.global_start_step, stage.total_steps)
    probe_update = gate_step(student, gate, probe_lr, gate_lo, gate_hi)

    stream_stats = {}
    for stream in old_streams:
        backward_probe_stream(
            student, trajectories[stream], device, world, limit)
        stream_stats[stream] = update_plasticity_budgets(
            student, gate, plasticity_gradient)

    restored_dual = restore_gate(stage, gate, dist.get_rank(), world, state_specs)
    if restored_dual != dual:
        raise RuntimeError("probe changed the restored dual state")
    write_theta(student, gate)
    student.zero_grad(set_to_none=True)
    del plasticity_gradient

    target = stage.artifact_dir / "gate_state" / "plasticity_budget.json"
    if dist.get_rank() == 0:
        target.parent.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for name, item in gate.items():
            if item["active_global"] == 0:
                continue
            old = {stream: stream_stats[stream][name] for stream in old_streams}
            tensors[name] = {
                "plasticity_rms": old[old_streams[0]]["plasticity_rms"],
                "old_streams": old,
                "plasticity_budget": float(item["plasticity_budget"]),
            }
        budgets = sorted(item["plasticity_budget"] for item in tensors.values())
        target.write_text(json.dumps({
            "method_version": stage.config["method_version"],
            "stage": stage.name, "round": stage.round,
            "probe_trajectories_per_stream": limit,
            "probe_gate_lr": probe_lr, "new_stream": new_stream,
            "old_streams": old_streams, "probe_update": probe_update,
            "budget_summary": {
                "min": budgets[0], "median": budgets[len(budgets) // 2],
                "max": budgets[-1],
            },
            "tensors": tensors,
        }, indent=2) + "\n")
    dist.barrier()
    return target


def main() -> None:
    experiment = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=experiment)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--round", required=True)
    args = parser.parse_args()
    stage = MergeStage(args.experiment, args.stage, args.round)
    stage.prepare()
    cfg = stage.config
    seed = int(cfg["seed"])
    random.seed(seed + stage.round_index)
    torch.manual_seed(seed + stage.round_index)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    states_per_step = int(cfg["optimization"]["states_per_optimizer_step_per_stream"])
    if states_per_step % world:
        raise RuntimeError("states per optimizer step must be divisible by world size")

    state_specs, train_states, probe_trajectories = load_cache_manifest(stage)
    student = wrap(stage.old_model, device)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    store = torch.device(cfg["optimization"].get("gate_state_device", "cpu"))
    if store.type == "cuda":
        store = device
    gate = build_gate(student, stage.new_expert, device, 0.0, store)
    endpoints = measure_endpoints(
        student, gate, state_specs, probe_trajectories, device,
        int(cfg["endpoint"]["trajectories_per_source"]))
    if rank == 0:
        (stage.artifact_dir / "endpoints.json").write_text(json.dumps(endpoints, indent=2) + "\n")

    gate_lo, gate_hi = (float(value) for value in cfg["optimization"]["gate_range"])
    if cfg["optimization"]["gate_lr_schedule"] != "cosine":
        raise RuntimeError("only cosine gate learning rate is implemented")
    dual = restore_gate(stage, gate, rank, world, state_specs)
    write_theta(student, gate)
    student.train()
    budget_path = None
    if bool(cfg.get("stability_plasticity", {}).get("enabled", False)):
        budget_path = calibrate_plasticity_budgets(
            stage, student, gate, dual, state_specs, probe_trajectories,
            device, world, gate_lo, gate_hi)

    raw_kl_budget = float(cfg["trust_region"]["raw_kl_budget"])
    dual_lr = float(cfg["trust_region"]["dual_lr"])
    dual_max = float(cfg["trust_region"]["dual_max"])
    target_steps = stage.steps
    total_steps = stage.total_steps
    global_start_step = stage.global_start_step
    orders = {}
    for offset, (name, values) in enumerate(sorted(train_states.items())):
        copied = values.copy()
        random.Random(seed + stage.round_index * 1000 + offset).shuffle(copied)
        orders[name] = copied

    log = (stage.log_dir / "train.jsonl").open("w", encoding="utf-8") if rank == 0 else None
    started = time.time()
    for optimizer_step in range(target_steps):
        student.zero_grad(set_to_none=True)
        running = {name: {"state_kl": 0.0, "trajectory_weight": 0.0,
                          "states": 0.0, "tokens": 0.0}
                   for name in train_states}
        for name in sorted(train_states):
            order = orders[name]
            batch = [order[(optimizer_step * states_per_step + index) % len(order)]
                     for index in range(states_per_step)]
            train_state_batch(
                student, batch,
                1.0 if state_specs[name]["role"] == "new" else dual[name],
                device, world, states_per_step, running[name])

        lr_now = cosine_lr(stage.gate_lr, global_start_step + optimizer_step, total_steps)
        moved = gate_step(student, gate, lr_now, gate_lo, gate_hi)
        student.zero_grad(set_to_none=True)
        reduced = reduce_running(running, device)
        raw_kl = {name: value["state_kl"] for name, value in reduced.items()}
        old_drift = {name: float(raw_kl_drift(
            raw_kl[name], endpoints[name]["at_old"])) for name in dual}
        for name in dual:
            dual[name] = min(dual_max, max(0.0, dual[name] + dual_lr *
                                            (old_drift[name] - raw_kl_budget)))
        if log is not None:
            log.write(json.dumps({
                "method_version": cfg["method_version"],
                "round": stage.round, "optimizer_step": optimizer_step + 1,
                "global_step": global_start_step + optimizer_step + 1,
                "raw_kl": raw_kl, "old_drift": old_drift,
                "dual": dict(dual), "raw_kl_budget": raw_kl_budget,
                "gate_lr": lr_now,
                "trajectory_weight_sum": {
                    name: value["trajectory_weight"] for name, value in reduced.items()},
                "state_count": {name: value["states"] for name, value in reduced.items()},
                "token_count": {name: value["tokens"] for name, value in reduced.items()},
                **moved, "elapsed_seconds": time.time() - started,
            }) + "\n")
            log.flush()

    if log is not None:
        log.close()
    save_gate(stage, gate, dual, world, rank)
    export_dense(stage, student, rank)
    summary_gate = gate_summary(gate, gate_lo, gate_hi, device)
    if rank == 0:
        summary = {
            "method_version": cfg["method_version"], "stage": stage.name,
            "round": stage.round, "old_model": str(stage.old_model),
            "new_expert": str(stage.new_expert), "export_dir": str(stage.export_dir),
            "optimizer_steps": target_steps, "global_step_start": global_start_step,
            "global_step_end": global_start_step + target_steps,
            "total_optimizer_steps": total_steps, "gate_lr": stage.gate_lr,
            "raw_kl_budget": raw_kl_budget, "final_dual": dual,
            "endpoints": endpoints,
            "world_size": world, "states_per_optimizer_step_per_stream": states_per_step,
            "state_manifest": str(stage.state_dir / "train_manifest.json"),
            "plasticity_budget": str(budget_path) if budget_path else None,
            "elapsed_seconds": time.time() - started, **summary_gate,
        }
        (stage.artifact_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
