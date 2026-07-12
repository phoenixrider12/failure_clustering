# Unsupervised Discovery of Failure Taxonomies from Deployment Logs (IROS 2026)

### [Project Website](https://mllm-failure-clustering.github.io/) &nbsp;|&nbsp; [Paper (arXiv:2506.06570)](https://arxiv.org/abs/2506.06570)

Codebase for our paper **"Unsupervised Discovery of Failure Taxonomies from Deployment Logs."**
We present a framework that turns the raw failure logs of an autonomous system into a **semantic taxonomy of its failure modes**, without any manual annotation, and then uses that taxonomy for downstream safety tasks — **runtime monitoring**, **targeted data collection**, and **policy refinement**.

The framework has three stages:

| Stage | Name | What it does | Script(s) |
|-------|------|--------------|-----------|
| **1** | Failure Reasoning | A vision-language model (Gemini 2.5 Pro) turns each failure trajectory into a natural-language `(trajectory, failure_reason)` explanation. Long rollouts are compressed with **semantic observation downsampling** (CLIP-based frame selection around the failure). | [`step1_failure_reasoning/`](step1_failure_reasoning/) |
| **2** | Taxonomy Discovery | A reasoning LLM (o4-mini) clusters the free-text explanations into a failure taxonomy, using an **ensemble-and-aggregate** strategy for stability. | [`step2_taxonomy_discovery/`](step2_taxonomy_discovery/) |
| **3** | Trajectory Assignment | Every trajectory is mapped to one (or more) discovered failure mode. | [`step3_trajectory_assignment/`](step3_trajectory_assignment/) |

We demonstrate it on **three systems**:

- **Autonomous Driving** — ego-car crashes from the [Nexar dashcam dataset](https://www.kaggle.com/competitions/nexar-collision-prediction/data).
- **Vision-Based Indoor Navigation** — collisions of the [LB-WayPtNav](https://github.com/mllm-failure-clustering/Visual-Navigation-Release/tree/failure_clustering) robot.
- **Robot Manipulation** — a household kitchen robot failing tasks, using the [REFLECT](https://github.com/real-stanford/reflect) dataset (simulated kitchen tasks + real UR5 demos).

---

## Repository structure

```
failure_clustering/
├── failure_taxonomy/            # Shared library used by every stage
│   ├── llm.py                   #   unified Gemini / OpenAI access (env-based keys, retries)
│   ├── prompts.py               #   all reasoning / clustering / assignment / monitor prompts
│   ├── frame_selection.py       #   semantic observation downsampling (CLIP) + baselines
│   ├── io_utils.py              #   frame loading, output parsing, JSONL records
│   ├── schemas.py               #   pydantic schemas for structured LLM outputs
│   └── config.py                #   per-case-study defaults
│
├── step1_failure_reasoning/
│   └── get_descriptions.py      # Stage 1  (all 3 case studies via --case-study)
├── step2_taxonomy_discovery/
│   ├── get_clusters.py          # Stage 2a  ensemble taxonomy proposal
│   ├── aggregate_clusters.py    # Stage 2b  consolidate the ensemble
│   └── convert_to_json.py       # Stage 2c  structure the taxonomy -> JSONL
├── step3_trajectory_assignment/
│   └── assign_to_clusters.py    # Stage 3
│
├── runtime_monitoring/
│   ├── monitor.py               # taxonomy-guided runtime monitor
│   └── run_monitoring.py        # evaluate the monitor (TPR/TNR/FPR/FNR/F1)
│
├── evaluation/
│   ├── description_metrics.py   # CS / ROUGE-L / LLM-Judge for Stage-1 quality
│   ├── assignment_metrics.py    # weighted-F1 assignment accuracy vs expert labels
│   ├── taxonomy_metrics.py      # semantic alignment vs an expert taxonomy
│   └── baselines.py             # BERTopic clustering + cosine-similarity assignment
│
├── figures/
│   └── plot_results.py          # reproduce the paper's result figures
│
└── docs/                        # per-case-study dataset + run instructions
    ├── driving.md
    ├── navigation.md
    └── manipulation.md
```

All pipeline stages share one small library (`failure_taxonomy/`) so that a single, well-tested code path handles all three case studies — the case study is selected with `--case-study {driving,navigation,manipulation}`.

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

CLIP-based frame selection (Stage 1, manipulation) uses `torch` + `transformers`; a GPU is recommended but not required.

## API keys

The pipeline calls hosted models (Gemini for reasoning, OpenAI for clustering/assignment/monitoring). **Keys are always read from environment variables — none are stored in the code.** Add them once to your `~/.bashrc` (then `source ~/.bashrc`):

```bash
export GOOGLE_API_KEY="your-gemini-key"      # (or GEMINI_API_KEY) https://aistudio.google.com/apikey
export OPENAI_API_KEY="your-openai-key"      # https://platform.openai.com/api-keys
# optional:
export ANTHROPIC_API_KEY="your-anthropic-key"   # alternative LLM-judge
export TOGETHER_API_KEY="your-together-key"      # Llama/DeepSeek clustering baselines
```

Every script reads keys from these variables via `failure_taxonomy.llm`. (A `.env` file is also supported as a fallback — copy `.env.example` to `.env`.)

## Datasets

Datasets are **not** included in this repository. Download the one(s) you need and pass the local path to each script with `--input` / `--dataset`. Per-case-study download and directory-layout instructions are in:

- [`docs/driving.md`](docs/driving.md)
- [`docs/navigation.md`](docs/navigation.md)
- [`docs/manipulation.md`](docs/manipulation.md)

---

## Running the pipeline

The three stages are identical across case studies; only the inputs differ. Below is the **manipulation** case study end-to-end (swap `--case-study` and paths for driving/navigation — see the docs).

### Stage 1 — Failure reasoning

```bash
cd step1_failure_reasoning
python get_descriptions.py --case-study manipulation \
    --input      /path/to/reflect_dataset/real_data \
    --tasks-json /path/to/reflect_dataset/tasks_real_world.json \
    --output     ../results/manipulation/descriptions.jsonl
```

For driving / navigation the input is a directory of trajectory sub-folders (each an image sequence):

```bash
python get_descriptions.py --case-study navigation \
    --input  /path/to/navigation/failure_trajectories \
    --output ../results/navigation/descriptions.jsonl
```

### Stage 2 — Taxonomy discovery

```bash
cd step2_taxonomy_discovery
python get_clusters.py      --case-study manipulation \
    --descriptions ../results/manipulation/descriptions.jsonl \
    --output-dir   ../results/manipulation/cluster_reports
python aggregate_clusters.py --case-study manipulation \
    --reports-dir ../results/manipulation/cluster_reports \
    --output      ../results/manipulation/aggregated_taxonomy.txt
python convert_to_json.py \
    --input  ../results/manipulation/aggregated_taxonomy.txt \
    --output ../results/manipulation/taxonomy.jsonl
```

### Stage 3 — Trajectory assignment

```bash
cd step3_trajectory_assignment
python assign_to_clusters.py --case-study manipulation \
    --taxonomy     ../results/manipulation/taxonomy.jsonl \
    --descriptions ../results/manipulation/descriptions.jsonl \
    --output       ../results/manipulation/assignments.jsonl
```

---

## Runtime monitoring with the failure taxonomy

The discovered failure modes seed a runtime monitor that flags impending failures from a short window of recent frames. It reports the detection metrics in the paper (TPR, TNR, FPR, FNR, F1).

```bash
cd runtime_monitoring
python run_monitoring.py --case-study navigation \
    --dataset /path/to/navigation/trajectories \
    --labels  /path/to/navigation/labels.csv \
    --prompt-kind taxonomy \
    --output ../results/navigation/monitoring.csv
```

`--prompt-kind generic` runs the ablation monitor with no taxonomy (SAFE/UNSAFE only).

### Engaging the safeguard policy & targeted data collection

For indoor navigation we also integrate the monitor into the deployed policy (triggering an expert fallback when a failure is predicted) and perform **targeted data collection** around failure-cluster regions, followed by **policy refinement**. These run inside the LB-WayPtNav simulator — see [`docs/navigation.md`](docs/navigation.md) for the commands against the [Visual-Navigation-Release](https://github.com/mllm-failure-clustering/Visual-Navigation-Release/tree/failure_clustering) repo.

---

## Evaluation

```bash
cd evaluation

# Stage-1 description quality (manipulation): CS / ROUGE-L / METEOR (+ LLM-J)
python description_metrics.py --descriptions ../results/manipulation/descriptions.jsonl --llm-judge

# Taxonomy alignment vs an expert taxonomy (semantic alignment score)
python taxonomy_metrics.py \
    --taxonomy ../results/manipulation/taxonomy.jsonl \
    --expert   /path/to/expert_taxonomy.jsonl

# Assignment accuracy vs expert labels (weighted F1)
python assignment_metrics.py --csv /path/to/assignment_eval.csv --pred-col pred_ID --true-col human_ID

# Baselines used in the paper's comparisons
python baselines.py cluster --descriptions ../results/manipulation/descriptions.jsonl \
    --output ../results/manipulation/bertopic_taxonomy.jsonl
python baselines.py assign  --taxonomy ../results/manipulation/taxonomy.jsonl \
    --descriptions ../results/manipulation/descriptions.jsonl \
    --output ../results/manipulation/assignments_cosine.jsonl
```

## Figures

```bash
cd figures
python plot_results.py --output-dir .        # regenerate all paper figures
```

---

## Citation

```bibtex
@misc{gupta2026unsuperviseddiscoveryfailuretaxonomies,
      title={Unsupervised Discovery of Failure Taxonomies from Deployment Logs}, 
      author={Aryaman Gupta and Yusuf Umut Ciftci and Somil Bansal},
      year={2026},
      eprint={2506.06570},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.06570}, 
}
```
