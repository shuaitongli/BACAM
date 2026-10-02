#!/usr/bin/env python3
"""FSDP helpers for the parameter-wise box gate used by BACAM."""

from __future__ import annotations

import functools
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


def wrap(path: Path, device: torch.device) -> FSDP:
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=os.environ.get("TSMC_ATTN", "flash_attention_2"),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                       reduce_dtype=torch.float32,
                                       buffer_dtype=torch.bfloat16),
        device_id=device,
        use_orig_params=True,
    )


def build_gate(student: FSDP, expert_path: Path, device: torch.device, init: float,
               store: torch.device) -> dict:
    old = {name: parameter.data.detach().to(store, copy=True)
           for name, parameter in student.named_parameters()}
    expert = wrap(expert_path, device)
    state = {}
    for name, parameter in expert.named_parameters():
        if name not in old or parameter.data.shape != old[name].shape:
            raise RuntimeError(f"{name}: old/new shard mismatch")
        new = parameter.data.detach().to(store, copy=True)
        state[name] = {
            "old": old[name], "new": new,
            "g": torch.full(parameter.data.shape, float(init), dtype=torch.float32, device=store),
            "active": int((new != old[name]).sum()),
            "plasticity_budget": 1.0,
        }
    if set(old) != set(state):
        raise RuntimeError("old/new parameter names do not match")
    del expert
    torch.cuda.empty_cache()
    order = [name for name, _ in student.named_parameters()]
    counts = torch.tensor([state[name]["active"] for name in order],
                          dtype=torch.float64, device=device)
    dist.all_reduce(counts)
    for name, value in zip(order, counts.tolist()):
        state[name]["active_global"] = int(value)
    return state


def conflict_budget_from_stats(p_squared: float, s_squared: float, dot: float,
                               active: int) -> dict[str, float]:
    """Return one tensor's conflict-aware plasticity budget."""
    if active <= 0 or p_squared <= 0.0 or s_squared <= 0.0:
        return {"plasticity_rms": 0.0 if active <= 0 else math.sqrt(max(p_squared, 0.0) / active),
                "stability_rms": 0.0 if active <= 0 else math.sqrt(max(s_squared, 0.0) / active),
                "cosine": 0.0, "conflict": 0.0, "candidate_budget": 1.0}
    plasticity_rms = math.sqrt(p_squared / active)
    stability_rms = math.sqrt(s_squared / active)
    cosine = max(-1.0, min(1.0, dot / math.sqrt(p_squared * s_squared)))
    conflict = max(0.0, -cosine)
    budget = 1.0 - conflict * stability_rms / (plasticity_rms + stability_rms)
    return {"plasticity_rms": plasticity_rms, "stability_rms": stability_rms,
            "cosine": cosine, "conflict": conflict,
            "candidate_budget": max(0.0, min(1.0, budget))}


@torch.no_grad()
def capture_gate_gradient(student: FSDP, state: dict) -> dict[str, torch.Tensor]:
    """Capture the current local gate-gradient shard for a later probe comparison."""
    captured = {}
    for name, parameter in student.named_parameters():
        item = state[name]
        if parameter.grad is None or item["active_global"] == 0:
            continue
        delta = item["new"].to(parameter.device).float() - item["old"].to(parameter.device).float()
        captured[name] = (parameter.grad.float() * delta).to(
            device=item["g"].device, dtype=torch.bfloat16, copy=True)
    return captured


@torch.no_grad()
def update_plasticity_budgets(student: FSDP, state: dict,
                              plasticity_gradient: dict[str, torch.Tensor]) -> dict:
    """Compare the current stability gradient with a saved plasticity gradient."""
    order = [name for name, _ in student.named_parameters()]
    positions = {name: index for index, name in enumerate(order)}
    device = next(student.parameters()).device
    moments = torch.zeros((len(order), 3), dtype=torch.float64, device=device)
    for name, parameter in student.named_parameters():
        item = state[name]
        if (parameter.grad is None or item["active_global"] == 0
                or name not in plasticity_gradient):
            continue
        delta = item["new"].to(parameter.device).float() - item["old"].to(parameter.device).float()
        stability = parameter.grad.float() * delta
        plasticity = plasticity_gradient[name].to(parameter.device).float()
        index = positions[name]
        moments[index, 0] = plasticity.pow(2).sum().double()
        moments[index, 1] = stability.pow(2).sum().double()
        moments[index, 2] = (plasticity * stability).sum().double()
    dist.all_reduce(moments)

    output = {}
    for name in order:
        item = state[name]
        index = positions[name]
        values = conflict_budget_from_stats(
            float(moments[index, 0]), float(moments[index, 1]),
            float(moments[index, 2]), int(item["active_global"]))
        item["plasticity_budget"] = min(
            float(item.get("plasticity_budget", 1.0)), values["candidate_budget"])
        output[name] = {**values, "plasticity_budget": item["plasticity_budget"]}
    return output


def apply_plasticity_budget(update: torch.Tensor, budget: float) -> torch.Tensor:
    """Scale only positive gate movement toward the new expert."""
    return torch.where(update > 0, update * float(budget), update)


@torch.no_grad()
def write_theta(student: FSDP, state: dict) -> None:
    for name, parameter in student.named_parameters():
        item = state[name]
        gate = item["g"].to(parameter.device)
        old = item["old"].to(parameter.device).float()
        new = item["new"].to(parameter.device).float()
        parameter.data.copy_(((1.0 - gate) * old + gate * new).to(parameter.data.dtype))


@torch.no_grad()
def gate_step(student: FSDP, state: dict, lr: float, lo: float, hi: float) -> dict:
    order = [name for name, _ in student.named_parameters()]
    positions = {name: index for index, name in enumerate(order)}
    device = next(student.parameters()).device
    sumsq = torch.zeros(len(order), dtype=torch.float64, device=device)
    active = torch.tensor([state[name]["active_global"] for name in order],
                          dtype=torch.float64, device=device)

    def delta(item: dict, target: torch.device) -> torch.Tensor:
        return item["new"].to(target).float() - item["old"].to(target).float()

    for name, parameter in student.named_parameters():
        if parameter.grad is None or state[name]["active_global"] == 0:
            continue
        gradient = parameter.grad.float() * delta(state[name], parameter.device)
        sumsq[positions[name]] = gradient.pow(2).sum().double()
    dist.all_reduce(sumsq)
    global_rms = float((sumsq.sum() / active.sum().clamp_min(1)).sqrt())
    tensor_rms = (sumsq[active > 0] / active[active > 0]).sqrt()
    quantiles = torch.quantile(
        tensor_rms, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0],
                                 dtype=tensor_rms.dtype, device=device))

    moved = torch.zeros(2, dtype=torch.float64, device=device)
    for name, parameter in student.named_parameters():
        item = state[name]
        if parameter.grad is None or item["active_global"] == 0:
            continue
        if global_rms <= 0:
            continue
        gradient = parameter.grad.float() * delta(item, parameter.device)
        gate = item["g"].to(parameter.device)
        before = gate.clone()
        update = apply_plasticity_budget(
            gradient * (-lr / global_rms), item.get("plasticity_budget", 1.0))
        gate.add_(update).clamp_(lo, hi)
        moved[0] += (gate - before).abs().sum().double()
        moved[1] += item["active"]
        item["g"].copy_(gate)
    dist.all_reduce(moved)
    write_theta(student, state)
    return {
        "mean_abs_gate_move": float(moved[0] / moved[1].clamp_min(1)),
        "global_gate_grad_rms": global_rms,
        "tensor_gate_grad_rms_min": float(quantiles[0]),
        "tensor_gate_grad_rms_median": float(quantiles[1]),
        "tensor_gate_grad_rms_p90": float(quantiles[2]),
        "tensor_gate_grad_rms_p99": float(quantiles[3]),
        "tensor_gate_grad_rms_max": float(quantiles[4]),
    }


@torch.no_grad()
def gate_summary(state: dict, lo: float, hi: float, device: torch.device) -> dict:
    total = torch.zeros((), dtype=torch.float64, device=device)
    weighted = torch.zeros((), dtype=torch.float64, device=device)
    free = torch.zeros((), dtype=torch.float64, device=device)
    bins = torch.zeros(10, dtype=torch.float64, device=device)
    for item in state.values():
        gate = item["g"]
        active = item["new"] != item["old"]
        count = int(active.sum())
        total += gate.numel()
        free += count
        if count:
            values = gate[active].float()
            weighted += values.sum(dtype=torch.float64).to(device)
            edges = torch.linspace(lo, hi, 11, device=values.device)[1:-1]
            index = torch.bucketize(values.clamp(lo, hi), edges, right=True)
            bins += index.bincount(minlength=10).double().to(device)
    for tensor in (total, weighted, free, bins):
        dist.all_reduce(tensor)
    if int(bins.sum()) != int(free):
        raise RuntimeError("gate histogram count mismatch")
    return {
        "parameters": int(total), "free_coordinates": int(free),
        "mean_gate_on_free": float(weighted / free.clamp_min(1)),
        "gate_histogram": [int(value) for value in bins],
    }


def cosine_lr(base: float, step: int, total: int) -> float:
    return base * 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))
