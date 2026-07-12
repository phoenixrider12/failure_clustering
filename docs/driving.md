# Autonomous Driving (Nexar dashcam)

Ego-car crashes recorded by a forward-facing dashcam. Failure logs are short
clips ending in a collision.

## Dataset

We use the open-source [Nexar Collision Prediction dataset](https://www.kaggle.com/competitions/nexar-collision-prediction/data)
(~1,500 real-world crash videos).

The pre-processed failure trajectories we use are provided on the project's
[dataset drive](https://drive.google.com/drive/folders/1lEUGI4vUhGsxbk_jUz2k1o2IvyrS-J3U),
which contains two folders — **`driving/`** (this case study) and `waypointnav/`
(the [indoor navigation](navigation.md) case study). Download the `driving/`
folder:

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1lEUGI4vUhGsxbk_jUz2k1o2IvyrS-J3U
# keep the driving/ sub-folder
```

Each clip is a directory of frames (cropped and sampled at 3 fps by the paper),
one sub-folder per trajectory:

```
driving/
├── failure_trajectories/
│   ├── 00001/  frame_000000.png frame_000001.png ...
│   ├── 00002/  ...
└── labels.csv                 # trajectory,label  (1 = crash, 0 = safe)
```

`labels.csv` is used only for **runtime-monitoring evaluation**. It has two
columns, `trajectory` and `label`; derive it from the Nexar target annotations
(e.g. `target_time_minus_1000ms` → 1 for a crash at that horizon, else 0).

## Run

```bash
# Stage 1 — VLM reasoning over the frame sequence
cd step1_failure_reasoning
python get_descriptions.py --case-study driving \
    --input  /path/to/driving/failure_trajectories \
    --output ../results/driving/descriptions.jsonl

# Stages 2 & 3 — taxonomy discovery + assignment (see README)
```

## Runtime monitoring

```bash
cd runtime_monitoring
python run_monitoring.py --case-study driving \
    --dataset /path/to/driving/failure_trajectories \
    --labels  /path/to/driving/labels.csv \
    --prompt-kind taxonomy \
    --output ../results/driving/monitoring.csv
```

`--prompt-kind generic` is the ablation monitor with no discovered failure modes.
The paper additionally reports a supervised baseline (a VideoMAE crash classifier)
and a "no-context" LLM monitor; those are external comparison systems and are not
part of this method codebase.
