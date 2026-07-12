"""I/O helpers shared across the pipeline.

Covers three recurring needs:

* **Frame loading / encoding** -- natural-sorted listing of an image-sequence
  directory, and base64 encoding for API payloads.
* **Model-output parsing** -- extracting the ``trajectory`` and ``failure_reason``
  fields from a VLM response that follows the paper's answer format.
* **Record persistence** -- reading/writing the JSONL description records that
  flow between Stage 1, Stage 2 and Stage 3.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Iterator, Optional

from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #
def natural_sort_key(s: str) -> list:
    """Key for natural ordering of filenames (img1, img2, ..., img10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_image_files(directory: str) -> list[str]:
    """Return absolute paths to image files in ``directory``, naturally sorted."""
    names = [f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTENSIONS)]
    names.sort(key=natural_sort_key)
    return [os.path.join(directory, n) for n in names]


def load_images(directory: str, last_n: Optional[int] = None) -> list[Image.Image]:
    """Load an ordered image sequence from a directory as PIL images.

    Args:
        directory: Folder containing the frames of one trajectory.
        last_n: If given, keep only the last ``n`` frames (the paper caps the
            number of frames sent to the VLM to fit the context window).
    """
    paths = list_image_files(directory)
    if last_n is not None and len(paths) > last_n:
        paths = paths[-last_n:]
    return [Image.open(p).convert("RGB") for p in paths]


def encode_image_base64(image_path: str) -> str:
    """Base64-encode an image file on disk."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Base64-encode a PIL image in memory."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --------------------------------------------------------------------------- #
# Parsing model outputs
# --------------------------------------------------------------------------- #
def parse_trajectory_and_reason(text: str) -> tuple[str, str]:
    """Split a VLM response into ``(trajectory, failure_reason)``.

    The reasoning prompts ask the model to answer in the form::

        trajectory: <trajectory_description>
        failure_reason: <semantic_failure_reason>

    This parser is tolerant of minor formatting variation (case, markdown
    escaping like ``failure\\_reason``, and the ``failure_reason`` field
    spanning to the end of the response).
    """
    if not text:
        return "", ""

    normalized = text.replace("failure\\_reason", "failure_reason")

    reason = ""
    reason_match = re.search(r"failure_reason\s*:(.*)", normalized, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()

    trajectory = ""
    traj_match = re.search(
        r"trajectory\s*:(.*?)(?:failure_reason\s*:|$)",
        normalized,
        re.IGNORECASE | re.DOTALL,
    )
    if traj_match:
        trajectory = traj_match.group(1).strip()

    # Fallback: if the response had no explicit fields, treat the whole thing
    # as the failure reason so nothing is silently dropped.
    if not reason and not trajectory:
        reason = text.strip()

    return trajectory, reason


# --------------------------------------------------------------------------- #
# Description records (JSONL)
# --------------------------------------------------------------------------- #
def append_record(path: str, record: dict) -> None:
    """Append one JSON record as a line to ``path`` (creating parent dirs)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_records(path: str) -> list[dict]:
    """Read a JSONL file of description/assignment records."""
    records: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def processed_ids(path: str, key: str = "filename") -> set[str]:
    """Return the set of already-processed record ids in a JSONL file.

    Enables resumable runs: a script can skip trajectories whose id already
    appears in its output file.
    """
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    for record in read_records(path):
        if key in record:
            done.add(record[key])
    return done


def read_failure_reasons(path: str, include_trajectory: bool = False) -> list[str]:
    """Collect failure-reason texts from a description JSONL file.

    This is the input to Stage 2 (taxonomy discovery). When
    ``include_trajectory`` is True, each item is prefixed with its trajectory
    description for extra context.
    """
    items: list[str] = []
    for record in read_records(path):
        reason = record.get("failure_reason")
        if not reason:
            continue
        if include_trajectory and record.get("trajectory"):
            items.append(f"Trajectory description: {record['trajectory']} Failure reason: {reason}")
        else:
            items.append(reason)
    return items


def iter_trajectory_dirs(root: str) -> Iterator[tuple[str, str]]:
    """Yield ``(name, path)`` for each trajectory sub-directory under ``root``."""
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            yield name, path
