#!/usr/bin/env python3
"""Cache teacher distributions and behavior-token masks on selected states."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_stage import MergeStage  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def model_fingerprint(path: Path) -> str:
    digest = hashlib.sha256((path / "config.json").read_bytes())
    index = path / "model.safetensors.index.json"
    if index.exists():
        digest.update(index.read_bytes())
    else:
        for shard in sorted(path.glob("*.safetensors")):
            digest.update(shard.name.encode())
            digest.update(str(shard.stat().st_size).encode())
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer) -> str:
    vocabulary = sorted(tokenizer.get_vocab().items())
    payload = json.dumps({"vocab": vocabulary,
                          "special_ids": sorted(tokenizer.all_special_ids),
                          "chat_template": tokenizer.chat_template},
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


BEHAVIOR_TAGS = {
    "tool": (("<tool_call>", "</tool_call>"), ("<response>", "</response>")),
    "search": (("<search>", "</search>"), ("<answer>", "</answer>")),
    "alfworld": (("<action>", "</action>"), ("[action>", "</action>"),
                 ("[action]", "[/action]"), ("<action]", "</action>")),
    "webshop": (("<action>", "</action>"), ("[action>", "</action>"),
                ("[action]", "[/action]"), ("<action]", "</action>")),
}


def training_trajectory_weights(states: list[dict], expected: int) -> dict[str, float]:
    counts = Counter(str(state["trajectory_id"]) for state in states
                     if bool(state["train_selected"]))
    total = sum(counts.values())
    if total != expected or not counts:
        raise RuntimeError("invalid trajectory-equal training view")
    return {trajectory_id: total / (len(counts) * count)
            for trajectory_id, count in counts.items()}


def encode_response(tokenizer, response: str, domain: str) -> tuple[list[int], list[bool], bool]:
    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    response_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    spans = []
    for opening, closing in BEHAVIOR_TAGS[domain]:
        start = 0
        while True:
            left = response.find(opening, start)
            if left < 0:
                break
            right = response.find(closing, left + len(opening))
            end = len(response) if right < 0 else right + len(closing)
            spans.append((left, end))
            start = end
    mask = [any(token_end > left and token_start < right
                for left, right in spans)
            for token_start, token_end in offsets]
    if tokenizer.eos_token_id is None:
        raise RuntimeError("canonical tokenizer has no EOS token")
    response_ids.append(int(tokenizer.eos_token_id))
    mask.append(bool(spans))
    return response_ids, mask, bool(spans)


def cache_stream(stage: MergeStage, stream: str, spec: dict, role: str, model,
                 tokenizer, teacher: Path, device: torch.device,
                 destination: Path) -> dict:
    cfg = stage.config
    top_k = int(cfg["distillation"]["teacher_top_k"])
    temperature = float(cfg["distillation"]["temperature"])
    data_cfg = cfg["data"]
    max_response = int(data_cfg["max_response_tokens"])
    max_sequence = int(data_cfg["max_sequence_tokens"])
    if top_k != 32:
        raise RuntimeError(f"BACAM requires teacher_top_k=32, got {top_k}")

    states_path = Path(spec["path"])
    states = read_jsonl(states_path)
    trajectory_weights = training_trajectory_weights(
        states, int(data_cfg["states_per_stream"]))
    cache_dir = destination / stream
    cache_dir.mkdir(parents=True, exist_ok=True)
    records, masses = [], []

    for index, state in enumerate(tqdm(states, desc=f"BACAM cache {stream}")):
        direction = state.get("kl_direction")
        expected_direction = "forward" if role == "new" else "reverse"
        if direction != expected_direction:
            raise RuntimeError(f"{stream}: expected {expected_direction}, got {direction}")
        shard = cache_dir / f"{index:04d}.pt"
        prompt_ids = tokenizer.encode(state["prompt"], add_special_tokens=False)
        response_ids, critical_mask, has_behavior = encode_response(
            tokenizer, state["response"], spec["domain"])
        trajectory_id = str(state["trajectory_id"])
        trajectory_weight = (
            trajectory_weights[trajectory_id]
            if bool(state["train_selected"]) else 0.0
        )
        if len(response_ids) == 0:
            raise RuntimeError(f"{stream}[{index}]: empty response")
        if len(response_ids) > max_response:
            raise RuntimeError(f"{stream}[{index}]: response would be truncated")
        input_ids = prompt_ids + response_ids
        if not prompt_ids or len(input_ids) > max_sequence:
            raise RuntimeError(f"{stream}[{index}]: invalid sequence length {len(input_ids)}")

        tokens = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        positions = torch.arange(len(prompt_ids) - 1,
                                 len(prompt_ids) + len(response_ids) - 1,
                                 device=device)
        ids_parts, probs_parts, other_parts = [], [], []
        with torch.inference_mode():
            hidden = model.model(input_ids=tokens, use_cache=False,
                                 return_dict=True).last_hidden_state[0]
            for chunk in positions.split(32):
                logits = model.lm_head(hidden[chunk]).float() / temperature
                probs = torch.softmax(logits, dim=-1)
                values, indices = torch.topk(probs, k=top_k, dim=-1)
                other = (1.0 - values.sum(-1)).clamp_min(0.0)
                if not torch.allclose(values.sum(-1) + other,
                                      torch.ones_like(other), atol=1e-4, rtol=1e-4):
                    raise RuntimeError(f"{stream}[{index}]: cached probabilities do not sum to one")
                ids_parts.append(indices.to(dtype=torch.int32, device="cpu"))
                probs_parts.append(values.to(dtype=torch.float32, device="cpu"))
                other_parts.append(other.to(dtype=torch.float32, device="cpu"))
            del hidden

        payload = {
            "distribution_format": "topk_plus_other_v1",
            "method_version": stage.config["method_version"],
            "domain": spec["domain"], "role": role, "source": spec["source"],
            "state_index": index, "episode_id": state["episode_id"],
            "trajectory_id": trajectory_id,
            "trajectory_position": int(state["trajectory_position"]),
            "train_selected": bool(state["train_selected"]),
            "probe_selected": bool(state["probe_selected"]),
            "trajectory_weight": trajectory_weight,
            "kl_direction": direction,
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
            "input_ids": torch.tensor(input_ids, dtype=torch.int32),
            "response_start": len(prompt_ids), "response_length": len(response_ids),
            "response_truncated": False,
            "critical_token_mask": torch.tensor(critical_mask, dtype=torch.bool),
            "has_behavior_span": has_behavior,
            "topk_ids": torch.cat(ids_parts),
            "topk_raw_probs": torch.cat(probs_parts),
            "other_prob": torch.cat(other_parts),
            "temperature": temperature,
        }
        torch.save(payload, shard)
        records.append({"state_index": index, "path": str(shard),
                        "trajectory_id": state["trajectory_id"],
                        "trajectory_position": int(state["trajectory_position"]),
                        "train_selected": bool(state["train_selected"]),
                        "probe_selected": bool(state["probe_selected"]),
                        "trajectory_weight": trajectory_weight,
                        "has_behavior_span": has_behavior,
                        "tokens": len(response_ids)})
        masses.append(float(payload["topk_raw_probs"].sum(-1).mean()))
        del tokens, positions, payload

    if len(records) != len(states):
        raise RuntimeError(f"{stream}: cache count does not match states")
    manifest = {
        "method_version": stage.config["method_version"],
        "distribution_format": "topk_plus_other_v1",
        "stage": stage.name, "stream": stream, "role": role,
        "source": spec["source"], "domain": spec["domain"],
        "kl_direction": spec["kl_direction"],
        "teacher": str(teacher), "teacher_fingerprint": model_fingerprint(teacher),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "states_source": str(states_path), "top_k": top_k, "temperature": temperature,
        "records": records, "skipped": [], "response_truncated": 0,
        "training_aggregation": "trajectory_equal",
        "train_trajectories": len(trajectory_weights),
        "trajectory_weight_sum": sum(float(item["trajectory_weight"])
                                     for item in records),
        "mean_topk_probability_mass": sum(masses) / max(len(masses), 1),
        "trajectory_ids": list(dict.fromkeys(item["trajectory_id"] for item in records)),
        "train_states": sum(bool(item["train_selected"]) for item in records),
        "probe_trajectory_ids": list(spec["probe_trajectory_ids"]),
        "missing_behavior_spans": sum(not bool(item["has_behavior_span"])
                                      for item in records),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"stream": stream, "cached": len(records),
            "trajectories": len(manifest["trajectory_ids"]),
            "mean_topk_probability_mass": manifest["mean_topk_probability_mass"]}


def main() -> None:
    experiment = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=experiment)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--round", default="r0")
    parser.add_argument("--role", choices=("old", "new", "all"), default="all")
    parser.add_argument("--stream", help="cache one manifest stream")
    parser.add_argument("--device", default="cuda:0")
    # Kept as a no-op for command compatibility. BACAM always refreshes every shard.
    parser.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    stage = MergeStage(args.experiment, args.stage, args.round)
    stage.prepare()
    device = torch.device(args.device)

    manifest_path = stage.state_dir / "train_manifest.json"
    destination = stage.cache_dir
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing state manifest: {manifest_path}")
    streams = json.loads(manifest_path.read_text())["streams"]
    candidates = ([(args.stream, streams[args.stream])] if args.stream else
                  list(streams.items()))
    pending = []
    for stream, spec in candidates:
        if spec["source"] != "current":
            continue
        domain = str(spec["domain"])
        in_old = domain in stage.old_domains
        in_new = domain in stage.new_domains
        if in_old == in_new:
            raise RuntimeError(f"{stream}: domain {domain!r} is not assigned to exactly one teacher")
        role = "old" if in_old else "new"
        if spec.get("role") != role:
            raise RuntimeError(
                f"{stream}: manifest role {spec.get('role')!r} disagrees with config role {role!r}")
        if args.role != "all" and role != args.role:
            continue
        pending.append((stream, spec, role))
    if not pending:
        raise RuntimeError(f"no {args.role} streams found in {manifest_path}")

    summaries = []
    canonical_tokenizer = AutoTokenizer.from_pretrained(
        stage.old_model, local_files_only=True, trust_remote_code=True)
    canonical_fingerprint = tokenizer_fingerprint(canonical_tokenizer)
    for role in ("old", "new"):
        selected = [(stream, spec) for stream, spec, stream_role in pending
                    if stream_role == role]
        if not selected:
            continue
        teacher = stage.old_model if role == "old" else stage.new_expert
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher, local_files_only=True, trust_remote_code=True)
        if tokenizer_fingerprint(teacher_tokenizer) != canonical_fingerprint:
            raise RuntimeError(f"{role} teacher tokenizer does not match the old-model tokenizer")
        model = AutoModelForCausalLM.from_pretrained(
            teacher, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True).to(device).eval()
        model.config.use_cache = False
        for stream, spec in selected:
            summaries.append(cache_stream(stage, stream, spec, role, model,
                                          canonical_tokenizer, teacher, device,
                                          destination))
        del model
        torch.cuda.empty_cache()
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
