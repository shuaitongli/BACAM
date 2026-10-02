# Data format and examples

All IDs, paths, instructions, and responses below are fictional. These examples
describe the input format; they are not the experiment's data or a runnable dataset.
The original experiment's selected IDs are not distributed. Generate the four
split files locally with `python -m bacam.data.generate_splits --domain <domain>`
after installing the corresponding dataset and agent environment. For WebShop,
build the search index first. Existing split files are never overwritten.

A split lists which examples are used for training and which are reserved for
evaluation. An episode is a complete agent interaction; a state is one input and
response within it. JSONL means one JSON record per line. A stream groups training
states for one agent task. Probe episodes are used to estimate update conflicts,
not as a separate evaluation benchmark. A manifest records generated files and
their metadata.

## 1. Dataset splits

The generator writes the following files under `data/splits/`. The state builder currently
reads all four split files, including when building an early merging stage.
Training and held-out IDs must be disjoint and must refer to your installed data.

### `search_split.json`

`train` and `eval` contain question IDs from the MuSiQue source JSONL.
`source` records where that dataset came from.

```json
{
  "source": "synthetic-format-example",
  "train": ["example_search_train_001"],
  "eval": ["example_search_eval_001"]
}
```

### `bfcl_split.json`

Only `multi_turn_base` is used. Replace the example IDs with BFCL case IDs.

```json
{
  "categories": {
    "multi_turn_base": {
      "train": ["multi_turn_base_example_train_001"],
      "eval": ["multi_turn_base_example_eval_001"]
    }
  }
}
```

### `alfworld_split.json`

Game paths are relative to `data_root`. Set `data_root` to the actual absolute
ALFWorld data directory passed to the rollout script. Split JSON files do not
expand environment-variable placeholders.

```json
{
  "data_root": "/path/to/datasets/alfworld",
  "optimization_episodes": 1,
  "task_types": ["pick_and_place_simple"],
  "optimization_by_task": {
    "pick_and_place_simple": ["train/example_task/example_trial/game.tw-pddl"]
  }
}
```

The rollout count is divided evenly across `task_types`. Each task type needs
enough games. IID/OOD held-out evaluation is selected separately by the evaluator.

### `webshop_split.json`

Each goal entry must match the environment's goal at `goal_index`: the runner
checks `asin`, `category`, and `instruction` against the loaded environment.
`data_root` and `seed` must also match the rollout settings.

```json
{
  "data_root": "/path/to/datasets/webshop",
  "seed": 20260816,
  "optimization_episodes": 1,
  "source_goal_range": {"train": [500, 6909]},
  "optimization_goals": [
    {
      "goal_index": 500,
      "asin": "EXAMPLE_PRODUCT_001",
      "category": "Example category",
      "instruction": "Find a blue ceramic mug under 20 dollars."
    }
  ]
}
```

The current evaluator uses goals 0–499 as held-out data. The range above follows
the existing pipeline's training boundary; the goal contents are fictional.

## 2. Rollout states

The domain rollout scripts produce:

```text
data/<stage>/<round>/current_raw/states/<domain>_student_states.jsonl
```

**Each JSONL line is one state/assistant response, not a complete episode.**
Multiple lines sharing `episode_id` belong to the same episode.

| Field | Meaning |
| --- | --- |
| `episode_id` | ID included in that domain's training split; ALFWorld uses the relative game path, WebShop uses the goal index as a string |
| `turn` | Turn order within the episode |
| `step` | Within-turn step order, used for Tool |
| `prompt` | Exact serialized model input before generation, including the chat template, instruction, and available context |
| `response` | Generated assistant continuation, including behavior tags |
| `task_type` / `category` | ALFWorld / WebShop grouping for probe selection |
| `trajectory_terminal` | Required on the last ALFWorld/WebShop state; marks a finished rollout, not necessarily task success |

For example, these two abbreviated Search states form one episode:

```jsonl
{"domain":"search","episode_id":"example_search_train_001","turn":0,"prompt":"[Serialized chat: system instructions and user question about Exampletown.]","response":"<think>I need to look up the location.</think><search>Exampletown location</search>"}
{"domain":"search","episode_id":"example_search_train_001","turn":1,"prompt":"[Serialized chat with prior context and retrieval result: Exampletown is in Exampleland.]","response":"<think>The result identifies the country.</think><answer>Exampleland</answer>"}
```

Illustrative states for the other domains:

```jsonl
{"domain":"tool","episode_id":"multi_turn_base_example_train_001","turn":0,"step":0,"prompt":"[Serialized chat including tool definitions and a request for the weather in Exampletown.]","response":"<think>I should query the weather tool.</think><tool_call>{\"name\":\"get_weather\",\"parameters\":{\"city\":\"Exampletown\"}}</tool_call>"}
{"domain":"alfworld","episode_id":"train/example_task/example_trial/game.tw-pddl","task_type":"pick_and_place_simple","turn":0,"prompt":"[Serialized chat: put the mug on the table; the agent is holding the mug.]","response":"<think>The table is available.</think><action>put mug 1 in/on diningtable 1</action>","trajectory_terminal":true}
{"domain":"webshop","episode_id":"500","category":"Example category","turn":0,"prompt":"[Serialized chat showing the selected blue mug and the Buy Now button.]","response":"<think>This item matches the request.</think><action>click[Buy Now]</action>","trajectory_terminal":true}
```

Prompts here are placeholders for readability. Real records must contain the exact
model input, not these descriptions. Environment feedback belongs in the next
state's prompt. Rollout scripts also write summary metadata for completion/resume;
do not invent completion summaries for these example rows.

## 3. Generated training inputs

`bacam/data/build_states.py` validates split membership and trajectory lengths, then writes
`data/<stage>/<round>/states/train/<domain>_current.jsonl` and `train_manifest.json`.
It adds teacher role, KL direction, trajectory position, and the `train_selected`
and `probe_selected` flags. Training and probe selections may overlap.

`bacam/data/cache_teacher.py` reads these generated files and writes `.pt` teacher caches
and manifests under `data/<stage>/<round>/teacher_cache/`. These contain response
tokens, critical-token masks, and teacher top-32 probabilities plus tail mass.
You do not need to supply those tensors by hand.

## Dataset size

The default builder requires at least 256 valid states and 24 valid probe episodes
per stream after filtering. ALFWorld/WebShop probe selection also has per-category
quotas. These small examples do not meet those requirements.

The generator defaults to 144 ALFWorld episodes and 128 WebShop goals;
evaluation also requires the expected held-out sizes.
The examples illustrate schemas, not a replacement for those full experiment inputs.
