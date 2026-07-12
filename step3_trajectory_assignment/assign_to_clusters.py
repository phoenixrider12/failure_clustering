"""Stage 3: Trajectory assignment.

Assign every failure trajectory to one (or more) of the discovered
failure-mode clusters. Each trajectory's ``trajectory`` + ``failure_reason``
text is classified by the reasoning LLM against the structured taxonomy from
Stage 2. Runs are resumable (already-assigned trajectories are skipped).

The resulting per-trajectory assignments are what the paper evaluates against
expert labels (weighted-F1 assignment accuracy) and what targeted data
collection uses to find the environment regions to re-collect data in.

Example
-------
    python assign_to_clusters.py --case-study manipulation \\
        --taxonomy ../results/manipulation/taxonomy.jsonl \\
        --descriptions ../results/manipulation/descriptions.jsonl \\
        --output ../results/manipulation/assignments.jsonl
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from failure_taxonomy import config, io_utils, prompts  # noqa: E402
from failure_taxonomy.llm import llm_structured  # noqa: E402
from failure_taxonomy.schemas import Cluster, TrajectoryAssignment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("assign_to_clusters")


def load_taxonomy(path: str) -> list[Cluster]:
    return [Cluster(**record) for record in io_utils.read_records(path)]


def build_cluster_options(clusters: list[Cluster]) -> list[str]:
    """Render each cluster as ``"<name> : <keywords> — <notes>"`` for the prompt."""
    options = []
    for c in clusters:
        line = f"{c.cluster_name} : {', '.join(c.keywords)}"
        if c.notes:
            line += f" — {c.notes}"
        options.append(line)
    options.append("Other: for trajectories that do not fit any cluster above")
    return options


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case-study", required=True, choices=list(config.CASE_STUDIES))
    p.add_argument("--taxonomy", required=True, help="Structured taxonomy JSONL (from Stage 2c).")
    p.add_argument("--descriptions", required=True, help="Stage-1 descriptions JSONL.")
    p.add_argument("--output", required=True, help="Output JSONL of per-trajectory assignments.")
    p.add_argument("--model", default=None, help="LLM model id (default: per case study).")
    p.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between calls.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = config.get_config(args.case_study)
    model = args.model or cfg.llm_model

    clusters = load_taxonomy(args.taxonomy)
    options = build_cluster_options(clusters)
    prompt_template = prompts.assignment_prompt(options, case_study=args.case_study)
    logger.info("Loaded %d clusters; assigning with %s.", len(clusters), model)

    trajectories = io_utils.read_records(args.descriptions)
    done = io_utils.processed_ids(args.output, key="filename")

    n_assigned = 0
    for item in trajectories:
        name = item["filename"]
        if name in done:
            continue

        user_prompt = (
            prompt_template
            .replace("{trajectory}", item.get("trajectory", ""))
            .replace("{failure_reason}", item.get("failure_reason", ""))
        )
        messages = [
            {"role": "system", "content": prompts.ASSIGNMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            result = llm_structured(messages, TrajectoryAssignment, model=model)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to assign %s: %s", name, exc)
            continue

        io_utils.append_record(
            args.output, {"filename": name, "assignments": result.assignments}
        )
        n_assigned += 1
        logger.info("%s -> %s", name, result.assignments)
        time.sleep(args.sleep)

    logger.info("Done. Assigned %d trajectories; output at %s", n_assigned, args.output)


if __name__ == "__main__":
    main()
