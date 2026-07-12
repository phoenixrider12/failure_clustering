"""Stage 2c: Structure the taxonomy.

Convert the free-text aggregated taxonomy into a clean JSONL file where each
line is a cluster with ``cluster_name``, ``occurrence``, ``keywords`` and
``notes``. This machine-readable taxonomy is consumed by Stage 3 (assignment)
and by the runtime monitor.

The output format is enforced with a structured-output schema
(:class:`failure_taxonomy.schemas.ClustersResponse`), so parsing is exact rather
than heuristic.

Example
-------
    python convert_to_json.py \\
        --input ../results/manipulation/aggregated_taxonomy.txt \\
        --output ../results/manipulation/taxonomy.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from failure_taxonomy import config, prompts  # noqa: E402
from failure_taxonomy.llm import DEFAULT_LLM_MODEL, llm_structured  # noqa: E402
from failure_taxonomy.schemas import ClustersResponse  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Aggregated taxonomy text file.")
    p.add_argument("--output", required=True, help="Output JSONL path for the structured taxonomy.")
    p.add_argument("--model", default=DEFAULT_LLM_MODEL, help="LLM model id.")
    p.add_argument("--case-study", choices=list(config.CASE_STUDIES), default=None,
                   help="Optional; recorded for provenance only.")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.input) as f:
        taxonomy_text = f.read()

    print(f"Structuring taxonomy with {args.model}...")
    messages = [{"role": "user", "content": prompts.CONVERT_TO_JSON_INSTRUCTION + taxonomy_text}]
    result = llm_structured(messages, ClustersResponse, model=args.model)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for cluster in result.clusters:
            f.write(cluster.model_dump_json() + "\n")

    print(f"Done. {len(result.clusters)} clusters written to {args.output}")


if __name__ == "__main__":
    main()
