"""Evaluate Stage-1 failure descriptions against ground-truth failure reasons.

For the manipulation study each rollout has an expert ground-truth failure
reason, so the quality of the VLM's predicted ``failure_reason`` can be scored
directly. This script reports the three metrics used in the paper's VLM /
fine-tuned-model / frame-sampling comparisons:

* **CS**     -- SBERT cosine similarity (semantic similarity)
* **ROUGE-L**-- longest-common-subsequence overlap F1
* **LLM-J**  -- an LLM judge scoring semantic equivalence (optional; needs an API key)

ROUGE-1/2 and METEOR are also reported for completeness.

Input is a Stage-1 descriptions JSONL (records with ``failure_reason`` and
``ground_truth`` fields), i.e. the direct output of
``step1_failure_reasoning/get_descriptions.py`` on the manipulation dataset.

Example
-------
    python description_metrics.py \\
        --descriptions ../results/manipulation/descriptions.jsonl --llm-judge
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from failure_taxonomy import io_utils  # noqa: E402


def load_pairs(path: str) -> tuple[list[str], list[str]]:
    """Return (ground_truths, predictions) from a descriptions JSONL."""
    gts, preds = [], []
    for rec in io_utils.read_records(path):
        gt = rec.get("ground_truth", "").strip()
        pred = rec.get("failure_reason", "").strip()
        if gt and pred:
            gts.append(gt)
            preds.append(pred)
    return gts, preds


def sbert_cosine(gts: list[str], preds: list[str]) -> list[float]:
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_gt = model.encode(gts, convert_to_tensor=True)
    emb_pr = model.encode(preds, convert_to_tensor=True)
    return [float(util.cos_sim(emb_gt[i], emb_pr[i])) for i in range(len(gts))]


def rouge_meteor(gts: list[str], preds: list[str]) -> dict:
    import nltk
    from nltk import word_tokenize
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    for pkg in ("punkt", "punkt_tab", "wordnet"):
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl, met = [], [], [], []
    for gt, pred in zip(gts, preds):
        scores = scorer.score(gt, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
        try:
            met.append(meteor_score([word_tokenize(gt)], word_tokenize(pred)))
        except Exception:  # noqa: BLE001
            met.append(meteor_score([gt.split()], pred.split()))
    return {"rouge1": np.mean(r1), "rouge2": np.mean(r2), "rougeL": np.mean(rl), "meteor": np.mean(met)}


def llm_judge(gts: list[str], preds: list[str], model: str) -> float:
    """LLM-judge score: fraction of predictions judged semantically equivalent."""
    from failure_taxonomy.llm import llm_prompt

    total = 0.0
    for gt, pred in zip(gts, preds):
        prompt = (
            "Determine if the following two statements describe a similar reason behind the failure of a "
            "household robot performing a kitchen task:\n\n"
            f"Ground Truth Failure Reason: {gt}\n"
            f"Predicted Failure Reason: {pred}\n\n"
            "Respond with 'Yes' if they are similar, otherwise 'No'."
        )
        try:
            answer = llm_prompt(prompt, model=model).strip().lower()
            total += 1.0 if answer.startswith("yes") else 0.0
        except Exception as exc:  # noqa: BLE001
            print(f"  LLM-judge error: {exc}")
    return total / len(gts) if gts else 0.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--descriptions", required=True, help="Stage-1 descriptions JSONL with ground_truth.")
    p.add_argument("--llm-judge", action="store_true", help="Also compute the LLM-judge (LLM-J) score.")
    p.add_argument("--judge-model", default="o4-mini", help="Model id for the LLM judge.")
    return p.parse_args()


def main():
    args = parse_args()
    gts, preds = load_pairs(args.descriptions)
    if not gts:
        raise SystemExit("No (ground_truth, failure_reason) pairs found in the input.")
    print(f"Evaluating {len(gts)} predictions.\n")

    cs = sbert_cosine(gts, preds)
    rm = rouge_meteor(gts, preds)

    print("=== Description metrics ===")
    print(f"CS  (SBERT cosine): {np.mean(cs):.4f}")
    print(f"ROUGE-1:            {rm['rouge1']:.4f}")
    print(f"ROUGE-2:            {rm['rouge2']:.4f}")
    print(f"ROUGE-L:            {rm['rougeL']:.4f}")
    print(f"METEOR:            {rm['meteor']:.4f}")
    if args.llm_judge:
        print(f"LLM-J:             {llm_judge(gts, preds, args.judge_model):.4f}")


if __name__ == "__main__":
    main()
