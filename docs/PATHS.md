# Local path configuration

An environment variable is a setting exported in your shell, such as
`export BACAM_MODEL_ROOT="models"`. `source config/paths.env` loads these settings
into the current shell. The repository root is the BACAM directory containing
`README.md`. A service port identifies a local server used by agent requests;
two independent services cannot listen on the same address and port.

`core/paths.py` defines shared defaults for Python and shell entry points.
Shell scripts load these values through `scripts/paths.sh`. No original server
directories or Conda environment names are required by BACAM's own scripts.

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
`config/experiment.json`. Dataset directories are `alfworld`, `webshop`, and
`flashrag_eval` under `BACAM_DATA_ROOT`. Frozen split inputs and generated states
still live under `<BACAM>/data`; logs and diagnostic artifacts stay under
`<BACAM>/logs` and `<BACAM>/artifacts`.

## Set paths

From the repository root:

```bash
cp config/paths.env.example config/paths.env
# Edit config/paths.env to match your installation.
source config/paths.env
python core/paths.py
```

The local `paths.env` file is ignored by Git. Source it in the same shell before
running a pipeline or a standalone Python evaluator; it is not loaded implicitly.
Existing exported `BACAM_*` values override defaults. Relative directory paths
resolve against the BACAM checkout, regardless of the working directory.

## Python environments and executables

`BACAM_PYTHON` selects the training/cache interpreter. Shell entry points default
to `python` on PATH; Python entry points default to their current interpreter.
`BACAM_TOOL_PYTHON`, `BACAM_SEARCH_PYTHON`, `BACAM_ALFWORLD_PYTHON`, and
`BACAM_WEBSHOP_PYTHON` can point to separate environments, and otherwise inherit
the main interpreter.

`BACAM_TORCHRUN`, `BACAM_VLLM`, and `BACAM_BFCL` default to the corresponding
commands on PATH. If environments differ, set both the Python and command paths
explicitly, for example the training Python together with its `torchrun`, and the
Tool Python together with its `bfcl`. Values must be a command name or a single
executable path, not a shell command with arguments.

The experiment configuration uses `${BACAM_MODEL_ROOT}` and
`${BACAM_MERGE_ROOT}` placeholders. Stage resolution and result collection expand
them through `load_experiment`. A plain JSON reader sees the unexpanded template.

## GPUs and ports

`config/experiment.json` contains the default four physical GPU IDs and service
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

Set overrides in `config/paths.env` and source them before running commands.
External retrieval index/model paths and its GPU selection remain in the ReSearch
retriever YAML, optionally selected by `BACAM_RETRIEVER_CONFIG`.
Port separation alone does not make shared external BFCL files safe for concurrent
runs.
