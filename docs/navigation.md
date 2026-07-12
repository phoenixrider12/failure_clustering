# Vision-Based Indoor Navigation (LB-WayPtNav)

A vision-based ground robot that collides while navigating a simulated indoor
office environment. Failure logs are RGB rollouts ending in a collision.

## System & dataset

The robot and simulator are [LB-WayPtNav](https://github.com/mllm-failure-clustering/Visual-Navigation-Release/tree/failure_clustering).

The pre-processed collision trajectories we use are provided on the project's
[dataset drive](https://drive.google.com/drive/folders/1lEUGI4vUhGsxbk_jUz2k1o2IvyrS-J3U),
which contains two folders — **`waypointnav/`** (this case study) and `driving/`
(the [autonomous driving](driving.md) case study). Download the `waypointnav/`
folder:

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1lEUGI4vUhGsxbk_jUz2k1o2IvyrS-J3U
# keep the waypointnav/ sub-folder
```

Collision rollouts are directories of RGB frames, one sub-folder per trajectory:

```
waypointnav/
├── failure_trajectories/
│   ├── 0/  0.png 1.png 2.png ...
│   ├── 1/  ...
└── labels.csv                 # trajectory,label  (1 = collision, 0 = success)
```

`labels.csv` (for monitoring evaluation) is derived from each episode's metadata
(`episode_type_string` == `Success`/`Failure` in the simulator's
`trajectories/metadata.pkl`).

## Run

```bash
# Stage 1 — VLM reasoning over the frame sequence
cd step1_failure_reasoning
python get_descriptions.py --case-study navigation \
    --input  /path/to/navigation/failure_trajectories \
    --output ../results/navigation/descriptions.jsonl

# Stages 2 & 3 — taxonomy discovery + assignment (see README)
```

## Runtime monitoring

```bash
cd runtime_monitoring
python run_monitoring.py --case-study navigation \
    --dataset /path/to/navigation/failure_trajectories \
    --labels  /path/to/navigation/labels.csv \
    --prompt-kind taxonomy \
    --ensemble-shifts 3 \
    --output ../results/navigation/monitoring.csv
```

`--ensemble-shifts 3` requires three trailing windows to agree before raising a
failure alarm (reduces false positives, as in the paper).

## Safeguard policy, targeted data collection & policy refinement

These run inside the LB-WayPtNav simulator against the
[Visual-Navigation-Release](https://github.com/mllm-failure-clustering/Visual-Navigation-Release/tree/failure_clustering)
repo (branch `failure_clustering`). After cloning and setting it up per its
README:

```bash
# Runtime monitor integrated into the policy (expert fallback on predicted failure)
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python executables/rgb/resnet50/rgb_waypoint_trainer.py \
    test --job-dir logs --params params/rgb_trainer/reproduce_LB_WayPtNav_results/rgb_waypoint_trainer_finetune_params.py -d 0

# Targeted data collection around failure-cluster regions
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python executables/rgb/resnet50/rgb_waypoint_trainer.py \
    generate-data --job-dir logs --params params/rgb_trainer/reproduce_LB_WayPtNav_results/rgb_waypoint_trainer_finetune_params.py -d 0

# Policy refinement: fine-tune on the augmented dataset
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python executables/rgb/resnet50/rgb_waypoint_trainer.py \
    train --job-dir logs --params params/rgb_trainer/reproduce_LB_WayPtNav_results/rgb_waypoint_trainer_finetune_params.py -d 0
```

The failure-cluster assignments from Stage 3 (`assignments.jsonl`) identify the
environment regions to re-collect expert data in. This targeted refinement
reduces the navigation failure rate from 46% to 18% (vs 34% for uniform
sampling) in the paper.
