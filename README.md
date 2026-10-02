# BACAM

Official implementation of **BACAM: Behavior-Aware Continual Agent Merging for Multi-Turn Interaction**.

BACAM sequentially merges specialized language-model agents into a single dense
model, acquiring each incoming capability while preserving previously integrated
behaviors. It learns parameter-wise interpolation gates on the merged candidate's
own interaction trajectories, combining task-level behavioral constraints with
tensor-level conflict-aware update budgets. The gates are folded into the final
weights, adding no inference-time parameters or computation.

We merge four Qwen2.5-7B-Instruct-based experts for **WebShop, Tool, Search, and
ALFWorld**. In the default order (WTSA), BACAM achieves **62.82% average success
rate**, exceeding the strongest evaluated merging baseline by **21.69 percentage
points**.

The repository includes merging, agent rollouts, teacher caching, evaluation, and
gate/budget analysis. See [experiment results](docs/RESULTS.md) for the paper's
comparison tables, ablations, and stage-wise scores.

## Contents

- [Method](#method)
- [Results](#results)
- [Installation](#installation)
- [Expert models](#expert-models)
- [External repositories](#external-repositories)
- [Data preparation](#data-preparation)
- [Training and evaluation](#training-and-evaluation)
- [Code layout](#code-layout)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Method

![Overview of BACAM: candidate-generated trajectories, task-level behavioral control, and tensor-level conflict budgets.](assets/figures/overview.png)

[Vector figure (PDF)](assets/figures/overview.pdf)

BACAM combines three components:

1. **New-task Plasticity Control (NPC).** Forward KL to the incoming expert guides
   capability acquisition on candidate-generated histories, with explicit weighting
   of behavior-critical response regions.
2. **Old-task Stability Control (OSC).** Reverse KL to the previous merged model
   constrains drift on earlier tasks; a separate adaptive multiplier enforces each
   task's stability budget.
3. **Conflict-Aware Plasticity Budgeting (CAPB).** Local gradient probes estimate
   tensor-level conflicts and sensitivity, restricting gate updates toward the new
   expert where they threaten earlier capabilities.

The two source models remain frozen. Only the parameter-wise gate is optimized:

```text
theta_merged = theta_old + g * (theta_new - theta_old),  g in [0, 1]
```

Earlier experts need not be retained. Their task inputs and environments remain
available for interaction, while the previous merged model provides stability
supervision. All experts must share the same architecture and tokenizer.

## Results

Final success rates (%) after merging four experts. Continual methods use WTSA;
batch methods merge the same experts. Avg is the mean of the five task columns,
counting ALFWorld IID and OOD separately.

| Method | WebShop | Tool | Search | ALFWorld IID | ALFWorld OOD | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weight Average | 4.80 | 0.00 | 21.76 | 35.00 | 37.31 | 19.78 |
| Task Arithmetic | 32.00 | 2.00 | 20.82 | 66.43 | 58.21 | 35.89 |
| TIES-Merging | 38.40 | 0.00 | 15.17 | 53.57 | 56.72 | 32.77 |
| TSV-Merge | 39.40 | 1.00 | 23.10 | 68.57 | 66.42 | 39.70 |
| WUDI-Merging | 38.20 | 0.00 | 25.14 | 72.14 | 70.15 | 41.13 |
| AdaMerging | 16.80 | 1.00 | 15.27 | 62.86 | 59.70 | 31.13 |
| RAM++ | 37.20 | 1.00 | 20.33 | 61.43 | 50.00 | 33.99 |
| OPCM | 22.60 | 1.00 | 21.52 | 46.43 | 41.79 | 26.67 |
| NUFILT | 10.00 | 1.00 | 20.53 | 36.43 | 32.84 | 20.16 |
| **BACAM** | **77.20** | **38.00** | **23.75** | **88.57** | **86.57** | **62.82** |

This is a selection of the evaluated baselines; the [full comparison and ablations](docs/RESULTS.md)
include all methods. Tool uses BFCL `multi_turn_base`; Search uses MuSiQue.

### Performance across merging stages

![Success rates after each integration under WTSA, ASTW, TWAS, and SAWT.](assets/figures/merging_stages.png)

W = WebShop, T = Tool, S = Search, A = ALFWorld. Cell values are success rates (%);
shading indicates performance relative to the corresponding expert. A dash marks
a task whose expert has not yet been integrated. Order affects acquisition and
retention: final average success rates range from 56.02% to 62.82%.

[Stage-wise scores and tensor diagnostics](docs/RESULTS.md#performance-across-merging-stages)
· [Vector figure (PDF)](assets/figures/merging_stages.pdf)

## Installation

Use Linux with an NVIDIA GPU, a compatible driver, and a CUDA-enabled PyTorch
installation. The training pipeline uses four GPUs, FSDP, BF16, and FlashAttention.
The core Python dependencies are pinned in `requirements.txt`.
Clone the repository and install the core dependencies below. Install the agent
environments from their respective repositories. After cloning, run subsequent
commands from the BACAM directory.

```bash
git clone https://github.com/shuaitongli/BACAM.git
cd BACAM

conda create -n bacam python=3.12 -y
conda activate bacam
python -m pip install -r requirements.txt
python -m pip install packaging ninja psutil setuptools
python -m pip install flash-attn --no-build-isolation
```

Building [FlashAttention](https://github.com/Dao-AILab/flash-attention#installation-and-features)
requires a compatible CUDA toolkit with `nvcc` and a C++ compiler. Agent-specific
environment setup is provided by the external repositories listed below.

The default locations are `models/` for expert weights, `datasets/` for datasets,
and `third_party/` for external repositories. GPU IDs and service ports are in
`configs/experiment.json`. The launchers use the active environment's Python
and commands on PATH by default; no editable package installation or local
`paths.env` file is required.

If resources are stored elsewhere or agents use separate environments, configure
their paths and executables as described in [PATHS.md](docs/PATHS.md) before
running. The optional settings file is only a convenience for these overrides.

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
hf download langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop \
  --local-dir models/GiGPO-Qwen2.5-7B-Instruct-WebShop
hf download emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold \
  --local-dir models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold
hf download agentrl/ReSearch-Qwen-7B-Instruct \
  --local-dir models/ReSearch-Qwen-7B-Instruct
hf download langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld \
  --local-dir models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld
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
repository's README. Place the checkouts at `third_party/ReSearch`,
`third_party/bfcl-toolrl`, and `third_party/verl-agent`, or override their locations
as described in [PATHS.md](docs/PATHS.md).
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
`python -m bacam.data.build_webshop_index` from the repository root in the WebShop
environment. Dataset locations can
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
The GPU IDs in `configs/experiment.json` must refer to four available physical GPUs;
the provided default IDs may not match your machine.

After configuring dependencies, model paths, datasets, splits, GPUs, and ports:

```bash
bash scripts/services/start_retriever.sh
bash scripts/run_all.sh
```

To continue an interrupted run with its existing outputs:

```bash
bash scripts/run_all.sh --resume
```

WebShop is the initial expert. Each subsequent merging stage generates student
trajectories, filters states, caches the appropriate teacher distributions, trains
the gates, exports a dense model, and evaluates the capabilities learned so far.

Merged weights are written to `BACAM_MERGE_ROOT` (`outputs/models` by default).
Generated training data and teacher caches are under `data/`; evaluation results
and plots are under `artifacts/`; execution logs are under `logs/`.
Capability checks use 90% of each expert's reference score in `configs/expert_references.json`.

## Code layout

```text
BACAM/
├── bacam/                   Python package
│   ├── merging/             Gate updates, behavioral losses, training
│   ├── data/                States, teacher caches, dataset/index preparation
│   ├── rollout/             Agent interactions and trajectory conversion
│   ├── evaluation/          Held-out evaluation and result collection
│   ├── analysis/            Capability plots and tensor diagnostics
│   └── paths.py             Shared resource configuration
├── scripts/
│   ├── run_all.sh           Complete merging workflow
│   ├── cache_teacher.sh     Parallel teacher caching
│   ├── pipelines/           Per-stage merge pipelines
│   ├── rollout/             Agent rollout launchers
│   ├── evaluate/            Evaluation launchers
│   ├── services/            Retrieval service
│   └── lib/                 Shared Shell helpers
├── configs/
│   ├── experiment.json      Merging, GPU and port settings
│   ├── expert_references.json  Expert reference scores
│   └── paths.env.example    Local resource settings template
├── docs/                    Configuration, data formats, paper results
├── assets/figures/          Paper figures (PNG and PDF)
├── pyproject.toml           Python package metadata
└── requirements.txt         Core dependencies
```

The main algorithm files are `bacam/merging/gate_ops.py`, `bacam/merging/loss.py`, and `bacam/merging/train.py`.
`bacam/data/cache_teacher.py` prepares teacher distributions and behavior masks;
`bacam/data/build_states.py` builds training and budget-probe inputs. The full workflow
starts at `scripts/run_all.sh`. See [PATHS.md](docs/PATHS.md) for resource settings.

Standalone Python entry points use module execution, for example
`python -m bacam.analysis.plot_capability_trends`. Shell launchers expose the
checkout to all configured agent interpreters through `PYTHONPATH`; they do not
require installing the training dependencies into each agent environment.

Model weights, datasets, generated trajectories, caches, checkpoints, and execution
logs are not bundled. The core pipeline has been checked with a small CPU model;
full agent rollouts and multi-GPU training have not been validated from a fresh
installation.

## Acknowledgements

BACAM uses experts and interaction environments from
[ReSearch](https://github.com/Agent-RL/ReCall/tree/re-search),
[ToolRL](https://github.com/qiancheng0/ToolRL),
[GiGPO / verl-agent](https://github.com/langfengQ/verl-agent),
[BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard),
[ALFWorld](https://github.com/alfworld/alfworld), and
[WebShop](https://github.com/princeton-nlp/WebShop).

We also thank [mergekit](https://github.com/arcee-ai/mergekit) for providing
model-merging baseline implementations used in our experiments.

## License

BACAM's original code is released under the [MIT License](LICENSE).
External repositories, model weights, and datasets remain subject to their own
licenses and usage conditions.
