#!/usr/bin/env python3
"""Rebuild per-turn ReSearch states from this stage's own rollout.

Prerequisite: `scripts/rollout/search.sh <stage> <round> <model>` has driven the
model through the real flashrag ReSearchPipeline over the interaction pool, with the
resident retriever.

ReSearchPipeline keeps only the whole trajectory (`output.final_response`), so the turns
are recovered from it.  The pipeline builds that string as

    query  = init_query
    query += f"{output_str} <result>\\n{retrieval}\\n</result>"    # each search step
    query += f"{output_str}"                                       # final step
    final_response = query.replace(init_query, "")

so splitting on the result blocks recovers the model outputs, and every prefix is the
exact prompt the model saw.  The reconstruction is checked by reassembly, not assumed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

from bacam.paths import PATHS

sys.path.insert(0, str(Path(PATHS["BACAM_RESEARCH_ROOT"]) / "src"))

from verl.utils.dataset.template import re_search_template_sys  # noqa: E402

from bacam.data.states_io import write_states  # noqa: E402
from bacam.rollout.rollout_stage import Stage  # noqa: E402

RESULT_BLOCK = re.compile(r"( <result>\n.*?\n</result>)", re.DOTALL)


def init_query(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": re_search_template_sys},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)


def turns(prompt0: str, final_response: str) -> list[tuple[str, str]]:
    """[(prompt, response)] per turn; raises if the split does not reassemble."""
    parts = RESULT_BLOCK.split(final_response)
    outputs, separators = parts[0::2], parts[1::2]
    if "".join(a + b for a, b in zip(outputs, separators + [""])) != final_response:
        raise RuntimeError("result-block split does not reassemble")
    records, prompt = [], prompt0
    for index, output in enumerate(outputs):
        if output.strip():
            records.append((prompt, output))
        if index < len(separators):
            prompt = prompt + output + separators[index]
    return records


def main() -> None:
    experiment = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=experiment)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stage = Stage(args.experiment, args.stage)
    stage.prepare()
    split = json.loads((args.experiment / "data/splits/search_split.json").read_text())
    train_ids, eval_ids = set(split["train"]), set(split["eval"])
    pool = json.loads((args.experiment / "data/splits/search_pool_dataset.json").read_text())

    rollout = args.rollout
    rows = json.loads(rollout.read_text())
    if len(rows) != int(pool["questions"]):
        raise RuntimeError(
            f"Search rollout has {len(rows)} rows, expected {pool['questions']}"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                              trust_remote_code=True)

    records, empty, missing_output = [], 0, 0
    for row in rows:
        item_id = str(row["id"])
        if item_id in eval_ids:
            raise RuntimeError(f"held-out question {item_id} present in the rollout")
        if item_id not in train_ids:
            raise RuntimeError(f"{item_id} is in neither split half")
        final = (row.get("output") or {}).get("final_response")
        if not final:
            missing_output += 1
            continue
        pairs = turns(init_query(tokenizer, row["question"]), final)
        if not pairs:
            empty += 1
        for turn, (prompt, response) in enumerate(pairs):
            records.append({
                "domain": "search", "episode_id": item_id, "turn": turn,
                "prompt": prompt, "response": response,
            })

    write_states(
        args.output_dir / "search_student_states.jsonl",
        records,
        {
            "stage": stage.name,
            "rollout": str(rollout),
            "pool_questions": int(pool["questions"]),
            "rollout_rows": len(rows),
            "questions_without_output": missing_output,
            "questions_with_no_usable_turn": empty,
            "episodes_with_states": len({r["episode_id"] for r in records}),
            "states": len(records),
            "mean_turns_per_episode": len(records) / max(len({r["episode_id"] for r in records}), 1),
            "heldout_states": 0,
        },
    )


if __name__ == "__main__":
    main()
