"""
failure_taxonomy
================

Shared library for *Enhancing Robot Safety via MLLM-Based Semantic
Interpretation of Failure Data* (the "failure taxonomy" framework).

The framework turns raw failure logs of an autonomous system into a semantic
taxonomy of failure modes in three stages, and then uses that taxonomy for
downstream safety tasks (runtime monitoring, targeted data collection):

    Stage 1  Failure Reasoning      -> per-trajectory natural-language failure
                                       explanations from a vision-language model
    Stage 2  Taxonomy Discovery     -> LLM ensemble clustering of the
                                       explanations into a failure taxonomy
    Stage 3  Trajectory Assignment  -> map every trajectory to a failure mode

This package centralises everything the per-stage scripts share:

    llm               unified LLM / VLM API access (Gemini + OpenAI), with
                      retries and environment-based API keys
    prompts           every prompt used in the paper, organised by case study
    frame_selection   semantic observation downsampling (CLIP) + baselines
    io_utils          parsing / writing of description and trajectory records
    schemas           pydantic models for structured LLM outputs
    config            per-case-study configuration (paths, prompt keys)

API keys are always read from environment variables (see the project README
for the recommended ``~/.bashrc`` setup). No key is ever hard-coded.
"""

from . import config, io_utils, prompts, schemas  # noqa: F401

__all__ = ["config", "io_utils", "prompts", "schemas"]

__version__ = "1.0.0"
