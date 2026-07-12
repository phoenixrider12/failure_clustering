"""Taxonomy-guided runtime failure monitor.

Once a failure taxonomy has been discovered (Stages 1-3), it can seed a
*runtime monitor*: at deployment time the monitor looks at a short window of the
most recent frames and decides whether the system is ``SAFE`` or about to fail.
Seeding the monitor prompt with the discovered failure modes ("taxonomy" prompt)
substantially outperforms a taxonomy-free SAFE/UNSAFE prompt ("generic"
ablation) and supervised baselines on out-of-distribution scenarios.

This module provides :class:`FailureMonitor`, a thin wrapper that encodes a
window of frames and queries a vision-capable LLM with the appropriate
monitoring prompt. The prompt text lives in
:data:`failure_taxonomy.prompts.MONITOR_PROMPTS`.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from failure_taxonomy import prompts  # noqa: E402
from failure_taxonomy.io_utils import encode_image_base64  # noqa: E402
from failure_taxonomy.llm import build_image_message, llm_chat  # noqa: E402


class FailureMonitor:
    """Vision-LLM runtime monitor for a given case study.

    Args:
        case_study: ``"driving"`` or ``"navigation"``.
        prompt_kind: ``"taxonomy"`` (proposed monitor, seeded with the
            discovered failure modes) or ``"generic"`` (ablation, SAFE/UNSAFE).
        model: Vision-capable model id.
    """

    SAFE_LABEL = "SAFE"

    def __init__(self, case_study: str, prompt_kind: str = "taxonomy", model: str = "o4-mini"):
        if case_study not in prompts.MONITOR_PROMPTS:
            raise KeyError(f"No monitor prompt for case study {case_study!r}")
        if prompt_kind not in ("taxonomy", "generic"):
            raise ValueError("prompt_kind must be 'taxonomy' or 'generic'")
        self.case_study = case_study
        self.prompt = prompts.MONITOR_PROMPTS[case_study][prompt_kind]
        self.model = model

    def predict(self, frame_paths: list[str]) -> str:
        """Return the monitor's label for a window of frames (ordered oldest->newest)."""
        base64_images = [encode_image_base64(p) for p in frame_paths]
        messages = build_image_message(self.prompt, base64_images, detail="high")
        return llm_chat(messages, model=self.model).strip()

    def is_failure(self, label: str) -> bool:
        """A prediction is a *failure* alarm unless it is exactly ``SAFE``."""
        return label is not None and label.strip().upper() != self.SAFE_LABEL


def confusion_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute TP/FP/TN/FN and the detection rates reported in the paper.

    Labels: 1 = failure, 0 = success.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    def safe_div(a, b):
        return a / b if b else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)  # == TPR
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "TPR": recall,
        "TNR": safe_div(tn, tn + fp),
        "FPR": safe_div(fp, fp + tn),
        "FNR": safe_div(fn, fn + tp),
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def print_metrics(metrics: dict) -> None:
    print("\n=== Runtime monitoring metrics ===")
    print(f"TP={metrics['TP']}  FP={metrics['FP']}  TN={metrics['TN']}  FN={metrics['FN']}")
    print(f"TPR (recall): {metrics['TPR']:.4f}   TNR: {metrics['TNR']:.4f}")
    print(f"FPR:          {metrics['FPR']:.4f}   FNR: {metrics['FNR']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"F1-score:     {metrics['F1']:.4f}")
