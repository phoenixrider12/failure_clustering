"""Run the taxonomy-guided runtime monitor over a failure/success dataset.

For every trajectory the monitor inspects a trailing window of frames and
predicts SAFE vs. failure; predictions are compared against ground-truth labels
to produce the detection metrics reported in the paper (TPR, TNR, FPR, FNR,
F1-score).

Dataset layout expected
-----------------------
``--dataset`` points to a directory of trajectory sub-folders, each containing
that trajectory's ordered image frames::

    dataset/
      traj_0001/  frame_000000.png frame_000001.png ...
      traj_0002/  ...

``--labels`` is a CSV mapping each trajectory to a ground-truth label
(1 = failure, 0 = success)::

    trajectory,label
    traj_0001,1
    traj_0002,0

(The original per-system datasets store this differently -- driving in a
targets CSV, navigation in per-episode metadata; convert those into the simple
CSV above once, and this single runner handles both systems.)

Example
-------
    python run_monitoring.py --case-study navigation \\
        --dataset /path/to/navigation/trajectories \\
        --labels  /path/to/navigation/labels.csv \\
        --prompt-kind taxonomy \\
        --output ../results/navigation/monitoring.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from failure_taxonomy import config, io_utils  # noqa: E402
from monitor import FailureMonitor, confusion_metrics, print_metrics  # noqa: E402


def load_labels(path: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        # Accept either (trajectory,label) headers or the first two columns.
        traj_col = "trajectory" if "trajectory" in reader.fieldnames else reader.fieldnames[0]
        label_col = "label" if "label" in reader.fieldnames else reader.fieldnames[1]
        for row in reader:
            labels[str(row[traj_col]).strip()] = int(float(row[label_col]))
    return labels


def windows_for(frame_paths: list[str], window: int, ensemble_shifts: int) -> list[list[str]]:
    """Build one or more trailing frame windows for a trajectory.

    With ``ensemble_shifts == 1`` a single window ending at the last frame is
    used. With more shifts, additional windows ending at earlier frames are
    added; the trajectory is flagged as a failure only if *every* window flags a
    failure (matches the navigation ensemble in the paper, which reduces false
    positives).
    """
    n = len(frame_paths)
    result = []
    for shift in range(ensemble_shifts):
        end = n - shift
        start = max(0, end - window)
        if end - start >= 1:
            result.append(frame_paths[start:end])
    return result or [frame_paths[-window:]]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case-study", required=True, choices=["driving", "navigation"])
    p.add_argument("--dataset", required=True, help="Directory of trajectory sub-folders.")
    p.add_argument("--labels", required=True, help="CSV mapping trajectory -> {0,1} ground truth.")
    p.add_argument("--output", default=None, help="Optional CSV to write per-trajectory predictions.")
    p.add_argument("--prompt-kind", choices=["taxonomy", "generic"], default="taxonomy",
                   help="taxonomy = proposed monitor; generic = ablation without failure modes.")
    p.add_argument("--model", default=None, help="Vision-capable model id (default: per case study).")
    p.add_argument("--window", type=int, default=None, help="Frames per monitor query.")
    p.add_argument("--ensemble-shifts", type=int, default=1,
                   help="Number of trailing windows to require agreement over (>=1).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = config.get_config(args.case_study)
    model = args.model or cfg.monitor_model
    window = args.window or cfg.monitor_window

    monitor = FailureMonitor(args.case_study, prompt_kind=args.prompt_kind, model=model)
    labels = load_labels(args.labels)

    y_true, y_pred, rows = [], [], []
    trajectories = list(io_utils.iter_trajectory_dirs(args.dataset))
    for i, (name, path) in enumerate(trajectories):
        if name not in labels:
            print(f"[{i + 1}/{len(trajectories)}] {name}: no label, skipping")
            continue
        frame_paths = io_utils.list_image_files(path)
        if not frame_paths:
            print(f"[{i + 1}/{len(trajectories)}] {name}: no frames, skipping")
            continue

        preds = [monitor.predict(w) for w in windows_for(frame_paths, window, args.ensemble_shifts)]
        # Failure only if every window flags a failure (agreement).
        pred_failure = int(all(monitor.is_failure(p) for p in preds))

        y_true.append(labels[name])
        y_pred.append(pred_failure)
        rows.append({"trajectory": name, "label": labels[name],
                     "pred": pred_failure, "monitor_output": " | ".join(preds)})
        print(f"[{i + 1}/{len(trajectories)}] {name}: gt={labels[name]} pred={pred_failure} ({preds})")

    metrics = confusion_metrics(y_true, y_pred)
    print_metrics(metrics)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["trajectory", "label", "pred", "monitor_output"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-trajectory predictions written to {args.output}")


if __name__ == "__main__":
    main()
