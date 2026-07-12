"""Stage 2b: Ensemble aggregation.

Consolidate the candidate taxonomies produced by ``get_clusters.py`` into a
single coherent taxonomy with non-overlapping failure-mode clusters, using the
reasoning LLM as an aggregator.

Example
-------
    python aggregate_clusters.py --case-study manipulation \\
        --reports-dir ../results/manipulation/cluster_reports \\
        --output ../results/manipulation/aggregated_taxonomy.txt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from failure_taxonomy import config, prompts  # noqa: E402
from failure_taxonomy.llm import llm_prompt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case-study", required=True, choices=list(config.CASE_STUDIES))
    p.add_argument("--reports-dir", required=True,
                   help="Directory of candidate taxonomy reports from get_clusters.py.")
    p.add_argument("--output", required=True, help="Output path for the aggregated taxonomy text.")
    p.add_argument("--model", default=None, help="LLM model id (default: per case study).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = config.get_config(args.case_study)
    model = args.model or cfg.llm_model

    report_files = sorted(f for f in os.listdir(args.reports_dir) if f.endswith(".txt"))
    if not report_files:
        raise SystemExit(f"No .txt reports found in {args.reports_dir}")

    prompt = prompts.AGGREGATION_INTRO[args.case_study]
    for i, name in enumerate(report_files):
        with open(os.path.join(args.reports_dir, name)) as f:
            prompt += f"REPORT {i + 1}: {name}\n{f.read()}\n\n"

    print(f"Aggregating {len(report_files)} candidate taxonomies with {model}...")
    output = llm_prompt(prompt, model=model)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(output)
    print(f"Done. Aggregated taxonomy written to {args.output}")


if __name__ == "__main__":
    main()
