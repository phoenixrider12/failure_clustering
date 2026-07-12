"""Per-case-study configuration.

Only *method* defaults live here (models, frame budgets, monitor window sizes).
Dataset locations are passed explicitly on the command line so the release does
not hard-code any absolute paths -- see each script's ``--help`` and the README
for how to point the scripts at your downloaded datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CaseStudyConfig:
    name: str
    display_name: str
    # Stage 1 (failure reasoning)
    reasoning_model: str = "gemini-2.5-pro"
    max_frames: int = 20  # frames sent to the VLM per trajectory
    # Stage 2/3 (taxonomy + assignment)
    llm_model: str = "o4-mini"
    n_ensemble_prompts: int = 4
    # Runtime monitoring
    monitor_model: str = "o4-mini"
    monitor_window: int = 5  # number of trailing frames per monitor query
    notes: str = ""


CASE_STUDIES: dict[str, CaseStudyConfig] = {
    "driving": CaseStudyConfig(
        name="driving",
        display_name="Autonomous Driving (Nexar dashcam)",
        monitor_window=5,
        notes="Failure logs are dashcam clips of ego-car crashes.",
    ),
    "navigation": CaseStudyConfig(
        name="navigation",
        display_name="Indoor Navigation (LB-WayPtNav)",
        monitor_window=5,
        notes="Failure logs are RGB rollouts of a robot colliding indoors.",
    ),
    "manipulation": CaseStudyConfig(
        name="manipulation",
        display_name="Robot Manipulation (REFLECT kitchen tasks)",
        max_frames=25,
        notes="Failure logs are videos of a household robot failing kitchen tasks.",
    ),
}


def get_config(case_study: str) -> CaseStudyConfig:
    if case_study not in CASE_STUDIES:
        raise KeyError(
            f"Unknown case study {case_study!r}. Choose from {list(CASE_STUDIES)}."
        )
    return CASE_STUDIES[case_study]
