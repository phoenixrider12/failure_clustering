"""Non-LLM baselines used in the paper's comparisons.

Two embedding-based baselines against which the LLM pipeline is compared:

* ``cluster`` -- **BERTopic** taxonomy discovery: cluster the free-text failure
  reasons with sentence embeddings instead of an LLM ensemble (the "BERTopic"
  bar in the clustering comparison).

* ``assign`` -- **cosine-similarity** trajectory assignment: assign each
  trajectory to the taxonomy cluster whose (name + keywords) embedding is most
  similar to the trajectory's failure-reason embedding, instead of using the LLM
  (the similarity baseline for assignment accuracy).

Both use ``sentence-transformers`` embeddings (all-MiniLM-L6-v2 by default).

Examples
--------
    python baselines.py cluster --descriptions ../results/manipulation/descriptions.jsonl \\
        --output ../results/manipulation/bertopic_taxonomy.jsonl

    python baselines.py assign --taxonomy ../results/manipulation/taxonomy.jsonl \\
        --descriptions ../results/manipulation/descriptions.jsonl \\
        --output ../results/manipulation/assignments_cosine.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from failure_taxonomy import io_utils  # noqa: E402
from failure_taxonomy.schemas import Cluster  # noqa: E402

DEFAULT_EMBEDDER = "all-MiniLM-L6-v2"


def _embedder(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


# --------------------------------------------------------------------------- #
# BERTopic clustering baseline
# --------------------------------------------------------------------------- #
def run_cluster(args) -> None:
    from bertopic import BERTopic

    reasons = io_utils.read_failure_reasons(args.descriptions)
    print(f"Clustering {len(reasons)} failure reasons with BERTopic ({args.embedder})...")

    topic_model = BERTopic(embedding_model=_embedder(args.embedder), verbose=True)
    topics, _ = topic_model.fit_transform(reasons)
    info = topic_model.get_topic_info()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for _, row in info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:  # BERTopic outlier topic
                continue
            keywords = [w for w, _ in topic_model.get_topic(topic_id)]
            cluster = Cluster(
                cluster_name=row.get("Name", f"Topic {topic_id}"),
                occurrence=str(row["Count"]),
                keywords=keywords,
                notes="",
            )
            f.write(cluster.model_dump_json() + "\n")
    print(f"BERTopic taxonomy ({len(info) - 1} topics) written to {args.output}")


# --------------------------------------------------------------------------- #
# Cosine-similarity assignment baseline
# --------------------------------------------------------------------------- #
def run_assign(args) -> None:
    import numpy as np
    from sentence_transformers import util

    clusters = [Cluster(**r) for r in io_utils.read_records(args.taxonomy)]
    cluster_texts = [f"{c.cluster_name}. {', '.join(c.keywords)}. {c.notes}".strip() for c in clusters]

    model = _embedder(args.embedder)
    cluster_emb = model.encode(cluster_texts, convert_to_tensor=True)

    records = io_utils.read_records(args.descriptions)
    print(f"Assigning {len(records)} trajectories to {len(clusters)} clusters by cosine similarity...")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for rec in records:
            query = (rec.get("failure_reason", "") + " " + rec.get("trajectory", "")).strip()
            if not query:
                continue
            q_emb = model.encode(query, convert_to_tensor=True)
            sims = util.cos_sim(q_emb, cluster_emb)[0].cpu().numpy()
            best = int(np.argmax(sims))
            f.write(json.dumps({
                "filename": rec["filename"],
                "assignments": clusters[best].cluster_name,
                "similarity": float(sims[best]),
            }) + "\n")
    print(f"Cosine-similarity assignments written to {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("cluster", help="BERTopic taxonomy-discovery baseline.")
    c.add_argument("--descriptions", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--embedder", default=DEFAULT_EMBEDDER)

    a = sub.add_parser("assign", help="Cosine-similarity trajectory-assignment baseline.")
    a.add_argument("--taxonomy", required=True)
    a.add_argument("--descriptions", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "cluster":
        run_cluster(args)
    else:
        run_assign(args)


if __name__ == "__main__":
    main()
