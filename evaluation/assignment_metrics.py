"""Evaluate per-trajectory cluster assignments against expert labels.

Compares the pipeline's Stage-3 assignments to human-annotated ground-truth
cluster labels and reports accuracy plus macro/weighted precision, recall and
F1 (the weighted-F1 assignment number reported in the paper), along with
standard clustering-agreement metrics (ARI, NMI, V-measure).

Input is a CSV with one row per trajectory and two label columns -- the
predicted cluster and the ground-truth cluster. Either column may contain a
single label or a comma/semicolon/pipe-separated set (multi-label); a
prediction counts as correct if it overlaps the ground-truth set.

Example
-------
    python assignment_metrics.py --csv ../results/manipulation/assignment_eval.csv \\
        --pred-col pred_ID --true-col human_ID
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    v_measure_score,
)


def parse_labels(value) -> set:
    """Parse a cell into a set of labels (handles multi-label separators)."""
    if pd.isna(value):
        return set()
    text = str(value).strip()
    for sep in (",", ";", "|"):
        if sep in text:
            parts = [p.strip() for p in text.split(sep)]
            break
    else:
        parts = [text]
    out = set()
    for p in parts:
        if not p:
            continue
        try:
            out.add(int(p))
        except ValueError:
            out.add(p)
    return out


def compute(df: pd.DataFrame, pred_col: str, true_col: str) -> dict:
    y_true, y_pred = [], []
    correct = total = 0

    for _, row in df.iterrows():
        true_set = parse_labels(row.get(true_col))
        pred_set = parse_labels(row.get(pred_col))
        total += 1
        if true_set & pred_set:
            correct += 1
        # Use the primary label from each side for the label-based metrics.
        y_true.append(next(iter(true_set)) if true_set else "None")
        y_pred.append(next(iter(pred_set)) if pred_set else "None")

    labels = sorted({str(x) for x in y_true + y_pred})
    y_true_s = [str(x) for x in y_true]
    y_pred_s = [str(x) for x in y_pred]

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "precision_macro": precision_score(y_true_s, y_pred_s, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_s, y_pred_s, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_s, y_pred_s, labels=labels, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true_s, y_pred_s, labels=labels, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true_s, y_pred_s, labels=labels, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true_s, y_pred_s, labels=labels, average="weighted", zero_division=0),
        "ari": adjusted_rand_score(y_true_s, y_pred_s),
        "nmi": normalized_mutual_info_score(y_true_s, y_pred_s),
        "v_measure": v_measure_score(y_true_s, y_pred_s),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="CSV with predicted and ground-truth label columns.")
    p.add_argument("--pred-col", default="pred_ID", help="Name of the predicted-label column.")
    p.add_argument("--true-col", default="human_ID", help="Name of the ground-truth-label column.")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    for col in (args.pred_col, args.true_col):
        if col not in df.columns:
            raise SystemExit(f"Column {col!r} not in CSV. Available: {list(df.columns)}")

    m = compute(df, args.pred_col, args.true_col)
    print("\n=== Assignment metrics ===")
    print(f"Trajectories:        {m['total']}")
    print(f"Accuracy (overlap):  {m['accuracy']:.4f} ({m['correct']}/{m['total']})")
    print(f"Macro    P/R/F1:     {m['precision_macro']:.4f} / {m['recall_macro']:.4f} / {m['f1_macro']:.4f}")
    print(f"Weighted P/R/F1:     {m['precision_weighted']:.4f} / {m['recall_weighted']:.4f} / {m['f1_weighted']:.4f}")
    print(f"ARI / NMI / V-meas:  {m['ari']:.4f} / {m['nmi']:.4f} / {m['v_measure']:.4f}")


if __name__ == "__main__":
    main()
