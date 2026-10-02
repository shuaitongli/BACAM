#!/usr/bin/env python3
"""Reconstruct exact RLLA prompts from this experiment's own student rollout.

Prerequisite: `scripts/rollout/tool.sh <stage> <round> <model>` has run the model in
the real BFCL environment on the training pool, via `bfcl generate --run-ids`.  This script
only reformats that fresh log into (prompt, response) states; no score, ground-truth call,
or held-out ID is read.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


from bacam.paths import PATHS

BFCL_ROOT = Path(PATHS["BACAM_BFCL_ROOT"])
sys.path.insert(0, str(BFCL_ROOT))

from bfcl_eval._llm_response_generation import process_multi_turn_test_case  # noqa: E402
from bfcl_eval.model_handler.local_inference.rlla import RLLAHandler  # noqa: E402
from bfcl_eval.utils import load_file  # noqa: E402

from bacam.rollout.rollout_stage import Stage  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def step_number(name: str) -> int:
    return int(name.removeprefix("step_"))


def main() -> None:
    experiment = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=experiment)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stage = Stage(args.experiment, args.stage)
    stage.prepare()

    split = json.loads((args.experiment / "data/splits/bfcl_split.json").read_text())
    category_name = "multi_turn_base"
    all_train_ids = set(split["categories"][category_name]["train"])
    eval_ids = set(split["categories"][category_name]["eval"])
    if all_train_ids & eval_ids:
        raise RuntimeError("BFCL train/eval overlap")
    candidate_categories = json.loads(args.candidate_ids.read_text())
    if set(candidate_categories) != {category_name}:
        raise RuntimeError("Tool candidates must contain multi_turn_base only")
    train_ids = {
        item
        for category in candidate_categories.values()
        for item in category
    }
    if len(train_ids) != sum(len(values) for values in candidate_categories.values()):
        raise RuntimeError("Tool rollout candidate IDs repeat across categories")
    if not train_ids <= all_train_ids:
        raise RuntimeError("Tool rollout candidates contain non-training IDs")
    if train_ids & eval_ids:
        raise RuntimeError("Held-out BFCL ID leaked into Tool rollout candidates")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    handler = RLLAHandler("bacam-wtsa-reconstruct", temperature=0.001)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tool_student_states.jsonl"
    records: list[dict] = []
    seen_episode_ids: set[str] = set()
    failed_episode_ids: set[str] = set()
    token_count_deltas: list[int] = []

    for category in candidate_categories:
        prompt_path = BFCL_ROOT / "bfcl_eval/data" / f"BFCL_v3_{category}.json"
        prompts = {row["id"]: row for row in load_file(prompt_path)}
        result_path = args.rollout_dir / f"BFCL_v3_{category}_result.json"
        results = {row["id"]: row for row in read_jsonl(result_path)}

        category_train = set(candidate_categories[category])
        missing = category_train - results.keys()
        if missing:
            raise RuntimeError(f"{category}: missing {len(missing)} training rollouts")

        for episode_id in sorted(category_train):
            test_entry = copy.deepcopy(prompts[episode_id])
            process_multi_turn_test_case([test_entry])
            inference_data = handler._pre_query_processing_prompting(test_entry)
            result = results[episode_id]
            if "inference_log" not in result:
                # These are genuine student rollout failures caused by a prompt
                # exceeding the model context window.  They contain no generated
                # student action and remain ineligible for training.
                failed_episode_ids.add(episode_id)
                continue
            turn_logs = [
                item for item in result["inference_log"]
                if isinstance(item, dict) and "begin_of_turn_query" in item
            ]
            episode_records: list[dict] = []
            episode_token_count_deltas: list[int] = []
            incomplete_episode = not turn_logs

            for turn_index, turn_log in enumerate(turn_logs):
                current_message = turn_log["begin_of_turn_query"]
                if turn_index == 0:
                    inference_data = handler.add_first_turn_message_prompting(inference_data, current_message)
                else:
                    inference_data = handler._add_next_turn_user_message_prompting(inference_data, current_message)

                step_names = sorted(
                    (key for key in turn_log if key.startswith("step_")),
                    key=step_number,
                )
                for step_name in step_names:
                    step_index = step_number(step_name)
                    step_log = turn_log[step_name]
                    assistant = next((x for x in step_log if x.get("role") == "assistant"), None)
                    if assistant is None or not isinstance(assistant.get("content"), str):
                        incomplete_episode = True
                        break

                    prompt = handler._format_prompt(
                        copy.deepcopy(inference_data["message"]),
                        inference_data["function"],
                    )
                    response = assistant["content"]
                    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                    response_tokens = len(tokenizer.encode(response, add_special_tokens=False))
                    logged_prompt_tokens = result["input_token_count"][turn_index][step_index]
                    logged_response_tokens = result["output_token_count"][turn_index][step_index]
                    episode_token_count_deltas.append(prompt_tokens - int(logged_prompt_tokens))

                    episode_records.append({
                        "domain": "tool",
                        "episode_id": episode_id,
                        "category": category,
                        "turn": turn_index,
                        "step": step_index,
                        "prompt": prompt,
                        "response": response,
                        "prompt_tokens": prompt_tokens,
                        "response_tokens": response_tokens,
                        "logged_prompt_tokens": logged_prompt_tokens,
                        "logged_response_tokens": logged_response_tokens,
                    })

                    handler_log = next(
                        (x for x in step_log if x.get("role") == "handler_log" and "model_response_decoded" in x),
                        None,
                    )
                    model_response_data = {"model_responses": response}
                    inference_data = handler._add_assistant_message_prompting(inference_data, model_response_data)
                    if handler_log is None:
                        continue
                    decoded = handler_log["model_response_decoded"]
                    tool_results = [x["content"] for x in step_log if x.get("role") == "tool"]
                    if decoded and tool_results:
                        model_response_data["model_responses_decoded"] = decoded
                        inference_data = handler._add_execution_results_prompting(
                            inference_data,
                            tool_results,
                            model_response_data,
                        )

                if incomplete_episode:
                    break

            if incomplete_episode or not episode_records:
                failed_episode_ids.add(episode_id)
                continue
            records.extend(episode_records)
            token_count_deltas.extend(episode_token_count_deltas)
            seen_episode_ids.add(episode_id)

    if seen_episode_ids | failed_episode_ids != train_ids:
        raise RuntimeError(
            f"Expected {len(train_ids)} episodes, reconstructed {len(seen_episode_ids)} "
            f"and observed {len(failed_episode_ids)} terminal rollout errors"
        )
    if any(record["episode_id"] in eval_ids for record in records):
        raise RuntimeError("Held-out BFCL state leaked into training records")

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "stage": stage.name,
        "full_optimization_episodes": len(all_train_ids),
        "rollout_candidate_episodes": len(train_ids),
        "candidates_per_category": {
            key: len(values) for key, values in candidate_categories.items()
        },
        "episodes": len(seen_episode_ids),
        "terminal_rollout_errors": len(failed_episode_ids),
        "terminal_rollout_error_ids": sorted(failed_episode_ids),
        "states": len(records),
        "max_prompt_tokens": max(record["prompt_tokens"] for record in records),
        "max_response_tokens": max(record["response_tokens"] for record in records),
        "prompt_token_delta_min": min(token_count_deltas),
        "prompt_token_delta_max": max(token_count_deltas),
        "prompt_token_delta_nonzero": sum(delta != 0 for delta in token_count_deltas),
        "heldout_states": 0,
        "output": str(output_path),
    }
    (output_dir / "tool_student_states.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
