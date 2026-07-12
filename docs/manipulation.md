# Robot Manipulation (REFLECT kitchen tasks)

Household kitchen robot that fails to complete tasks. Failure logs are videos of
task rollouts (simulated kitchen tasks and real UR5 demonstrations).

## Dataset

We use the [REFLECT](https://github.com/real-stanford/reflect) dataset. Organise
it as:

```
reflect_dataset/
├── sim_data/                     # simulated kitchen tasks
│   ├── makeCoffee/ ...
├── real_data/                    # real UR5 demonstrations
│   ├── makeCoffee1/
│   │   └── videos/color.mp4      # front-camera rollout video
│   └── ...
├── tasks_real_world.json         # task metadata (real)
└── tasks_sim.json                # task metadata (sim)
```

The task-metadata JSON maps each task id to:

```json
{
  "makeCoffee1": {
    "name": "make coffee",
    "success_condition": "coffee is brewed into the mug",
    "actions": ["pick up the mug", "place mug under dispenser", "..."],
    "gt_failure_step": "0:37",           // seconds, or MM:SS
    "gt_failure_reason": "the mug was placed off-centre so coffee spilled",
    "general_folder_name": "makeCoffee1",
    "object_list": ["mug", "coffee machine"]
  }
}
```

`gt_failure_reason` / `gt_failure_step` are used only for **evaluation** and for
locating the failure frame; they are never shown to the model when it reasons
about the failure.

## Run

```bash
# Stage 1 — semantic downsampling around the failure + Gemini reasoning
cd step1_failure_reasoning
python get_descriptions.py --case-study manipulation \
    --input      /path/to/reflect_dataset/real_data \
    --tasks-json /path/to/reflect_dataset/tasks_real_world.json \
    --output     ../results/manipulation/descriptions.jsonl \
    --frame-selection clip           # paper method; also: pixel, fps

# Stages 2 & 3 — identical to the README (taxonomy discovery + assignment)
```

### Frame-selection ablation (paper Fig. "frame sampling")

Compare semantic downsampling against fixed-rate sampling:

```bash
# fixed 1 / 0.5 / 0.25 fps baselines
python get_descriptions.py --case-study manipulation ... --frame-selection fps --fps 0.25
```

Then score each run with `evaluation/description_metrics.py`.

## Evaluation

- **Description quality** (CS / ROUGE-L / LLM-J), used for the VLM comparison,
  fine-tuned-model comparison and frame-sampling ablation:
  ```bash
  python evaluation/description_metrics.py \
      --descriptions results/manipulation/descriptions.jsonl --llm-judge
  ```
- **Taxonomy alignment** vs the expert taxonomy (semantic alignment score):
  ```bash
  python evaluation/taxonomy_metrics.py \
      --taxonomy results/manipulation/taxonomy.jsonl \
      --expert   /path/to/expert_taxonomy.jsonl
  ```
- **Assignment accuracy** (weighted F1) vs expert labels — build a CSV with a
  predicted-label column and an expert-label column and run
  `evaluation/assignment_metrics.py`.

## Fine-tuned-model & VLM baselines

The paper compares Gemini 2.5 Pro against other VLMs (LLaVA-NeXT, Qwen2.5-VL,
o4-mini, Cosmos-Reason1) and fine-tuned failure models (AHA-13B, RoboFAC-7B). To
reproduce these, generate `descriptions.jsonl` with the corresponding model and
score it the same way. RoboFAC and its ManiSkill failure environments come from
[MINT-SJTU/RoboFAC](https://github.com/MINT-SJTU/RoboFAC); the VLM backends are
run through their standard HuggingFace / API interfaces.
