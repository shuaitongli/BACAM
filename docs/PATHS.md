# Local path configuration

BACAM uses the following default resource locations. Set `BACAM_*` variables in
`configs/paths.env` to use your own directories and agent environments.
`<BACAM>` denotes the repository root.

| Variable | Default |
| --- | --- |
| `BACAM_MODEL_ROOT` | `<BACAM>/models` |
| `BACAM_MERGE_ROOT` | `<BACAM>/outputs/models` |
| `BACAM_DATA_ROOT` | `<BACAM>/datasets` |
| `BACAM_EXTERNAL_ROOT` | `<BACAM>/third_party` |
| `BACAM_RESEARCH_ROOT` | `<external root>/ReSearch` |
| `BACAM_BFCL_ROOT` | `<external root>/bfcl-toolrl` |
| `BACAM_VERL_AGENT_ROOT` | `<external root>/verl-agent` |

Model directory names under `BACAM_MODEL_ROOT` remain those listed in
`configs/experiment.json`. Dataset directories are `alfworld`, `webshop`, and
`flashrag_eval` under `BACAM_DATA_ROOT`. Split files and generated states are saved
under `<BACAM>/data`; logs and diagnostic artifacts are saved under
`<BACAM>/logs` and `<BACAM>/artifacts`.

## Set paths

From the repository root:

```bash
cp configs/paths.env.example configs/paths.env
# Edit configs/paths.env to match your installation.
source configs/paths.env
```

Source `configs/paths.env` in the same shell before
running a pipeline or a standalone Python evaluator; it is not loaded implicitly.
Existing exported `BACAM_*` values override defaults. Relative directory paths
resolve against the BACAM checkout, regardless of the working directory.

## Python environments and executables

`BACAM_PYTHON` selects the training/cache interpreter. Shell entry points default
to `python` on PATH; Python entry points default to their current interpreter.
`BACAM_TOOL_PYTHON`, `BACAM_SEARCH_PYTHON`, `BACAM_ALFWORLD_PYTHON`, and
`BACAM_WEBSHOP_PYTHON` can point to separate environments, and otherwise inherit
the main interpreter.
`BACAM_SGLANG_PYTHON` selects the Search model-serving environment separately
from the Search evaluation/retrieval environment.

`BACAM_TORCHRUN`, `BACAM_VLLM`, and `BACAM_BFCL` default to the corresponding
commands on PATH. If environments differ, set both the Python and command paths
explicitly, for example the training Python together with its `torchrun`, and the
Tool Python together with its `bfcl`. Values must be a command name or a single
executable path, not a shell command with arguments.

## GPUs and ports

`configs/experiment.json` contains the default four physical GPU IDs and service
ports. Set `BACAM_GPUS="0,1,2,3"` before launching to override the GPU list.
The four entries are assigned to Tool, Search, ALFWorld, and WebShop respectively;
training and teacher caching use the same list. The current pipeline requires
four distinct GPUs.

`BACAM_PORT_OFFSET` shifts all configured ports, including retrieval and torchrun.
Individual overrides such as `BACAM_RETRIEVER_PORT` or `BACAM_MASTER_PORT` take
precedence. Rollout/evaluation variables follow `BACAM_<TASK>_ROLLOUT_PORT` and
`BACAM_<TASK>_EVAL_PORT`; Tool uses `BACAM_TOOL_PORT` for both phases.
Invalid GPU lists, invalid ports, and duplicate configured ports fail before launch.
This checks configuration, not whether another process already occupies a port.

Set overrides in `configs/paths.env` and source them before running commands.
External retrieval index/model paths and its GPU selection remain in the ReSearch
retriever YAML, optionally selected by `BACAM_RETRIEVER_CONFIG`.
Port separation alone does not make shared external BFCL files safe for concurrent
runs.
