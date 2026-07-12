"""Evaluate a discovered taxonomy against an expert (ground-truth) taxonomy.

The **Semantic Alignment Score (SAS)** measures how well the automatically
discovered failure taxonomy matches an expert-defined one. An LLM judge scores
the similarity (1-10) of every discovered cluster against every expert cluster,
producing a similarity table; the table is min-max scaled per expert column and
summarised as:

    precision = mean over discovered clusters of their best expert match   (row max)
    recall    = mean over expert clusters of their best discovered match    (col max)
    SAS       = F1(precision, recall)
    coverage  = fraction of expert clusters matched above a similarity threshold

This is the taxonomy-alignment metric reported in the paper's clustering
comparison (BERTopic vs. LLM ensemble).

Example
-------
    python taxonomy_metrics.py \\
        --taxonomy ../results/manipulation/taxonomy.jsonl \\
        --expert   ../results/manipulation/expert_taxonomy.jsonl \\
        --heatmap  ../results/manipulation/alignment_heatmap.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from failure_taxonomy.llm import DEFAULT_LLM_MODEL, llm_prompt  # noqa: E402


def load_clusters(path: str) -> list[dict]:
    clusters = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            clusters.append({"name": c["cluster_name"], "keywords": c.get("keywords", [])})
    return clusters


def query_similarity(predicted: dict, expert: list[dict], model: str) -> list[float]:
    """Ask the LLM judge to score ``predicted`` against every expert cluster (1-10)."""
    expert_block = "\n".join(
        f"Cluster {i + 1}:\nName: {g['name']}\nKeywords: {', '.join(g['keywords'])}"
        for i, g in enumerate(expert)
    )
    prompt = (
        "Given the following predicted cluster and a list of ground truth clusters, where each cluster "
        "represents a failure mode of a robot, evaluate the similarity of the predicted cluster with each "
        "ground truth cluster on a scale of 1 to 10. For each pair, score it high if they mean exactly the "
        "same failure mode, else score it low.\n\n"
        f"Predicted Cluster:\nName: {predicted['name']}\nKeywords: {', '.join(predicted['keywords'])}\n\n"
        f"Ground Truth Clusters:\n{expert_block}\n\n"
        "Provide the similarity scores as a comma-separated list of numbers (one for each ground truth cluster)."
    )
    text = llm_prompt(prompt, model=model)
    scores = []
    for tok in text.replace("\n", ",").split(","):
        tok = tok.strip()
        try:
            scores.append(float(tok))
        except ValueError:
            continue
    # Pad/truncate to the number of expert clusters.
    if len(scores) < len(expert):
        scores += [0.0] * (len(expert) - len(scores))
    return scores[: len(expert)]


def min_max_scale_columns(matrix: np.ndarray) -> np.ndarray:
    """Min-max scale each column of the similarity table to [0, 1]."""
    scaled = np.zeros_like(matrix, dtype=float)
    for col in range(matrix.shape[1]):
        c = matrix[:, col]
        lo, hi = c.min(), c.max()
        scaled[:, col] = (c - lo) / (hi - lo) if hi > lo else 0.0
    return scaled


def semantic_alignment(discovered: list[dict], expert: list[dict], model: str, threshold: float = 0.6) -> dict:
    table = np.zeros((len(discovered), len(expert)))
    for i, c in enumerate(discovered):
        print(f"  scoring discovered cluster {i + 1}/{len(discovered)}: {c['name']}")
        table[i, :] = query_similarity(c, expert, model)

    scaled = min_max_scale_columns(table)
    precision = float(np.mean(np.max(scaled, axis=1))) if scaled.size else 0.0
    recall = float(np.mean(np.max(scaled, axis=0))) if scaled.size else 0.0
    sas = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    coverage = float(np.mean(np.max(scaled, axis=0) >= threshold)) if scaled.size else 0.0
    return {
        "similarity_table": table, "scaled_table": scaled,
        "precision": precision, "recall": recall, "sas": sas, "coverage": coverage,
    }


def save_heatmap(scaled, discovered, expert, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 9))
    plt.imshow(scaled, cmap="Blues", aspect="auto")
    plt.colorbar(label="Scaled similarity")
    plt.xticks(range(len(expert)), [g["name"] for g in expert], rotation=90, fontsize=8)
    plt.yticks(range(len(discovered)), [c["name"] for c in discovered], fontsize=8)
    plt.xlabel("Expert-defined taxonomy")
    plt.ylabel("Discovered clusters")
    for i in range(scaled.shape[0]):
        for j in range(scaled.shape[1]):
            plt.text(j, i, f"{scaled[i, j]:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if scaled[i, j] > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--taxonomy", required=True, help="Discovered taxonomy JSONL (from Stage 2c).")
    p.add_argument("--expert", required=True, help="Expert / ground-truth taxonomy JSONL.")
    p.add_argument("--model", default=DEFAULT_LLM_MODEL, help="LLM judge model id.")
    p.add_argument("--threshold", type=float, default=0.6, help="Coverage similarity threshold.")
    p.add_argument("--heatmap", default=None, help="Optional path to save an alignment heatmap.")
    return p.parse_args()


def main():
    args = parse_args()
    discovered = load_clusters(args.taxonomy)
    expert = load_clusters(args.expert)
    print(f"Discovered clusters: {len(discovered)}   Expert clusters: {len(expert)}\n")

    result = semantic_alignment(discovered, expert, args.model, args.threshold)
    print("\n--- Semantic alignment metrics ---")
    print(f"Precision (discovered -> expert): {result['precision']:.3f}")
    print(f"Recall (expert coverage):         {result['recall']:.3f}")
    print(f"Semantic Alignment Score (SAS):   {result['sas']:.3f}")
    print(f"Concept coverage (>= {args.threshold}):        {result['coverage']:.3f}")

    if args.heatmap:
        save_heatmap(result["scaled_table"], discovered, expert, args.heatmap)


if __name__ == "__main__":
    main()
