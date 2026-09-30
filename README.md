# BACAM

Official implementation of **BACAM: Behavior-Aware Continual Agent Merging For Multi-turn Interaction**.

BACAM merges specialized language-model agents into a single model. It learns
parameter-wise interpolation gates to acquire a new agent's capabilities while
preserving the behavior of agents merged earlier.

This repository provides the merging algorithm, on-policy rollout adapters,
teacher-distribution caching, training pipelines, evaluation, and diagnostic plots.
The default merge order is WebShop, Tool, Search, then ALFWorld (WTSA).

WebShop performs shopping tasks, Tool calls functions, Search answers questions
using retrieved documents, and ALFWorld interacts with a text-based household
environment. Run the commands below from the BACAM directory. Basic Linux shell
and Python environment management are assumed.

## Terms used in this guide

- Expert: a model specialized in one agent task. Student: the model being merged.
- Gate: a learned value that controls the contribution of the old and new weights.
- Rollout or trajectory: one complete interaction with an environment. A state
  records one model input and response within that interaction.
- Teacher cache: saved teacher probabilities for the student's generated responses,
  reused during training. The teacher does not generate replacement trajectories.
- KL: a measure of the difference between model output probabilities. Plasticity
  means acquiring a new capability; stability means retaining previous capabilities.
- Conflict budget: a coefficient that reduces updates harmful to previous tasks.
- Held-out data: examples reserved for evaluation and not used for training.

## Method

- Behavior-aware distillation balances critical action regions and other response
  tokens. The new agent is supervised by its expert; previous agents are supervised
  by the previously merged model.
- Old-task KL drift constraints adapt a separate stability multiplier for each
  previous agent.
- Tensor-wise gradient conflict budgets limit gate updates toward the new expert
  when those updates conflict with previous agents.

For each model parameter, the merged weight is
`theta = (1 - g) * theta_old + g * theta_new`, with `g` constrained to `[0, 1]`.
Plasticity uses forward KL and stability uses reverse KL. KL is computed on the
teacher's top-32 tokens plus the remaining probability mass. Gate gradients use
global RMS normalization; the conflict budget is shared within each tensor, not
the interpolation gate itself.

## Setup

Use Linux with an NVIDIA GPU, a compatible driver, and a CUDA-enabled PyTorch
installation. The training pipeline uses four GPUs, FSDP, BF16, and FlashAttention.
The core Python dependencies are pinned in `requirements.txt`.
FSDP distributes model training across GPUs; BF16 is a reduced-precision number
format; FlashAttention is an optimized attention implementation. Installing the
core dependencies does not install the four agent environments.

From the repository root:

```bash
conda create -n bacam python -y
conda activate bacam
python -m pip install -r requirements.txt
python -m pip install packaging ninja psutil setuptools
python -m pip install flash-attn --no-build-isolation

export BACAM_PYTHON="$(command -v python)"
export BACAM_TORCHRUN="$(command -v torchrun)"
```

Building [FlashAttention](https://github.com/Dao-AILab/flash-attention#installation-and-features)
requires a compatible CUDA toolkit with `nvcc` and a C++ compiler. Agent-specific
environment setup is provided by the external repositories listed below.

Configure local model, dataset, and external repository paths:

```bash
cp config/paths.env.example config/paths.env
# Edit config/paths.env for your installation, then load it.
source config/paths.env
python core/paths.py
```

GPU IDs and service ports are configured in `config/experiment.json` and can be
overridden through environment variables. See [PATHS.md](docs/PATHS.md).

## Expert models

Download the four complete model repositories, including weights, configuration,
and tokenizer files. The directory names below match the experiment configuration.
`BACAM_MODEL_ROOT` defaults to `models/` inside this repository.

| Agent | Hugging Face checkpoint |
| --- | --- |
| WebShop | [langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop](https://huggingface.co/langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop) |
| Tool | [emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold](https://huggingface.co/emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold) |
| Search | [agentrl/ReSearch-Qwen-7B-Instruct](https://huggingface.co/agentrl/ReSearch-Qwen-7B-Instruct) |
| ALFWorld | [langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld](https://huggingface.co/langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld) |

```bash
source scripts/paths.sh
export HF_ENDPOINT=https://huggingface.co

hf download langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop \
  --local-dir "$BACAM_MODEL_ROOT/GiGPO-Qwen2.5-7B-Instruct-WebShop"
hf download emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold \
  --local-dir "$BACAM_MODEL_ROOT/Qwen2.5-7B-Instruct-ToolRL-grpo-cold"
hf download agentrl/ReSearch-Qwen-7B-Instruct \
  --local-dir "$BACAM_MODEL_ROOT/ReSearch-Qwen-7B-Instruct"
hf download langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld \
  --local-dir "$BACAM_MODEL_ROOT/GiGPO-Qwen2.5-7B-Instruct-ALFWorld"
```

See the [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli)
for CLI installation and authentication. Add `--revision <commit>` to pin a
checkpoint for reproducibility. Model downloads do not include environment
datasets or the retrieval corpus/index.

## External repositories

| Agent | Repository |
| --- | --- |
| Search | [ReSearch (ReCall, re-search branch)](https://github.com/Agent-RL/ReCall/tree/re-search) |
| Tool | [ToolRL](https://github.com/qiancheng0/ToolRL), [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) |
| ALFWorld | [verl-agent / GiGPO](https://github.com/langfengQ/verl-agent), [ALFWorld](https://github.com/alfworld/alfworld) |
| WebShop | [verl-agent / GiGPO](https://github.com/langfengQ/verl-agent), [WebShop](https://github.com/princeton-nlp/WebShop) |

Install each external agent and its environment according to its original
repository's README, then set the checkout locations through `config/paths.env`.
These repositories are not bundled here. The adapters require
the ReSearch launch/evaluation interfaces and BFCL's RLLA handler; some environment
adapters check external source-file hashes. An arbitrary upstream checkout may
not satisfy these interfaces. BFCL runners modify shared registration and case-ID
files, so separate GPUs and ports alone do not isolate concurrent runs.

## Data preparation

Prepare the environment datasets, the Search retrieval corpus/index, and the four
split files under `data/splits/`. Training and held-out examples must be disjoint.
Tool training and evaluation use only BFCL `multi_turn_base`.

| Agent | Dataset used | Download/source |
| --- | --- | --- |
| Search | MuSiQue development questions in FlashRAG format | [FlashRAG datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets); ReSearch also provides `data/download_dataset.sh` |
| Tool | BFCL `multi_turn_base` cases, tool definitions, and reference answers | [BFCL data directory](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/data) in the original repository |
| ALFWorld | TextWorld household game files: training games, `valid_seen`, and `valid_unseen` | [ALFWorld download instructions](https://github.com/alfworld/alfworld#quickstart), using `alfworld-download` |
| WebShop | The 1,000-product subset and its shopping instructions | [WebShop setup instructions](https://github.com/princeton-nlp/WebShop#-setup), using the small dataset option |

Search additionally needs the Wikipedia corpus and retrieval index described in
[ReSearch's README](https://github.com/Agent-RL/ReCall/tree/re-search).
The default Search question file is `datasets/flashrag_eval/musique/dev.jsonl`.
WebShop expects `items_shuffle_1000.json`, `items_ins_v2_1000.json`, and
`items_human_ins.json` under `datasets/webshop/`; build its index with
`rollout/build_webshop_index.py` in the WebShop environment. Dataset locations can
be changed using `BACAM_DATA_ROOT`.

[DATA_FORMAT.md](docs/DATA_FORMAT.md) describes the required split schemas and
provides fictional examples. Actual split files and a complete dataset preparation
recipe are not included; the examples alone cannot run the default pipeline.
Each rollout JSONL line is a state and its generated response, not a full episode.
Training manifests and teacher caches are generated automatically.

## Training and evaluation

Before running, make sure all four expert model directories exist, the external
agent environments are installed, all four dataset split files have been created,
and the Search retrieval service and WebShop search index have been configured.
The GPU IDs in `config/paths.env` must refer to four available physical GPUs;
the provided default IDs may not match your machine.

After configuring dependencies, model paths, datasets, splits, GPUs, and ports:

```bash
source config/paths.env
bash evaluation/start_retriever.sh
bash scripts/run_all.sh
```

To continue an interrupted run with its existing outputs:

```bash
source config/paths.env
bash scripts/run_all.sh --resume
```

WebShop is the initial expert. Each subsequent merging stage generates student
trajectories, filters states, caches the appropriate teacher distributions, trains
the gates, exports a dense model, and evaluates the capabilities learned so far.

Merged weights are written to `BACAM_MERGE_ROOT` (`outputs/models` by default).
Generated training data and teacher caches are under `data/`; evaluation results
and plots are under `artifacts/`; execution logs are under `logs/`.
Capability checks use 90% of each expert's reference score in `baseline/config.json`.

## Code layout

```text
core/         Gate updates, losses, state building, teacher caching, training
rollout/      Agent interaction, trajectory conversion, dataset/index preparation
evaluation/   Held-out evaluation and result collection
analysis/     Training curves, capability plots, tensor gate/budget summaries
scripts/      Run entry points, merge pipelines, rollout launchers
config/       Training, paths, GPU and port configuration
baseline/     Expert reference scores
docs/         Path configuration and data formats
```

The main algorithm files are `core/gate_ops.py`, `core/loss.py`, and `core/train.py`.
`core/cache_teacher.py` prepares teacher distributions and behavior masks;
`core/build_states.py` builds training and budget-probe inputs. The full workflow
starts at `scripts/run_all.sh`. See [PATHS.md](docs/PATHS.md) for resource settings.

Weights, datasets, generated trajectories, caches, checkpoints, results, and logs
are not included in this repository. State construction, teacher caching, budget
calibration, loss/backward computation, gate updates, checkpoint restoration, and
model save/reload have been checked together with a temporary small CPU model.
Real agent rollouts and multi-GPU training have not been verified from a fresh
installation.

## License

BACAM's original code is released under the [MIT License](LICENSE).
External repositories, model weights, and datasets remain subject to their own
licenses and usage conditions.
