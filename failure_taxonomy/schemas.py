"""Pydantic schemas for structured LLM outputs.

Centralising these keeps Stage 2 (taxonomy discovery) and Stage 3 (trajectory
assignment) consistent: the same ``Cluster`` definition describes a taxonomy
entry when it is produced (``convert_to_json``) and when it is consumed
(``assign_to_clusters``).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Cluster(BaseModel):
    """One failure-mode cluster in a discovered taxonomy."""

    cluster_name: str = Field(..., description="Name of the cluster")
    occurrence: str = Field(
        default="", description="Percentage or frequency information if available"
    )
    keywords: List[str] = Field(
        default_factory=list, description="Characteristic keywords for the cluster"
    )
    notes: str = Field(default="", description="Additional description of the cluster")


class ClustersResponse(BaseModel):
    """A full taxonomy: a list of clusters."""

    clusters: List[Cluster]


class TrajectoryAssignment(BaseModel):
    """Assignment of a single trajectory to one or more failure-mode clusters.

    ``assignments`` is a free-form, comma-separated list of cluster names (or a
    single name), which allows multi-label assignment and an explicit "Other"
    label for trajectories that do not fit the taxonomy.
    """

    filename: str = Field(..., description="Filename / ID of the trajectory")
    assignments: str = Field(
        ..., description="Comma-separated cluster name(s) this trajectory belongs to"
    )
