"""Stage 2a: Ensemble taxonomy proposal.

Given the free-text failure reasons from Stage 1, prompt a reasoning LLM
(``o4-mini`` by default) several times with different clustering prompts. Each
run independently proposes a candidate failure taxonomy. The candidates are
consolidated in the next step (``aggregate_clusters.py``).

Running the LLM with an *ensemble* of prompts and then aggregating is what makes
the discovered taxonomy stable (Fig. "clustering comparison": single run vs.
ensemble+aggregation).

Example
-------
    python get_clusters.py --case-study manipulation \\
        --descriptions ../results/manipulation/descriptions.jsonl \\
        --output-dir ../results/manipulation/cluster_reports
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from failure_taxonomy import config, io_utils, prompts  # noqa: E402
from failure_taxonomy.llm import llm_prompt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case-study", required=True, choices=list(config.CASE_STUDIES))
    p.add_argument("--descriptions", required=True,
                   help="Stage-1 descriptions JSONL (source of failure reasons).")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write the candidate taxonomy reports into.")
    p.add_argument("--model", default=None, help="LLM model id (default: per case study).")
    p.add_argument("--n-prompts", type=int, default=None,
                   help="Number of ensemble prompts to run (default: per case study).")
    p.add_argument("--include-trajectory", action="store_true",
                   help="Include the trajectory text alongside each failure reason.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = config.get_config(args.case_study)
    model = args.model or cfg.llm_model
    ensemble = prompts.CLUSTERING_PROMPTS[args.case_study]
    n_prompts = args.n_prompts or min(cfg.n_ensemble_prompts, len(ensemble))

    reasons = io_utils.read_failure_reasons(args.descriptions, include_trajectory=args.include_trajectory)
    descriptions_block = "\n".join(reasons)
    print(f"Loaded {len(reasons)} failure reasons; running {n_prompts} ensemble prompt(s) with {model}.")

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(n_prompts):
        prompt = ensemble[i] + "\n\n" + descriptions_block
        print(f"  [{i + 1}/{n_prompts}] proposing taxonomy...")
        output = llm_prompt(prompt, model=model)

        out_path = os.path.join(args.output_dir, f"candidate_taxonomy_{i}.txt")
        with open(out_path, "w") as f:
            f.write(output)
        print(f"      saved -> {out_path}")

    print(f"Done. {n_prompts} candidate taxonomies in {args.output_dir}")


if __name__ == "__main__":
    main()
