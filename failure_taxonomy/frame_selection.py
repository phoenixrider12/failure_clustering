"""Semantic observation downsampling.

A long rollout contains far more frames than a VLM can ingest, but naive
uniform subsampling discards exactly the moments that explain a failure. The
paper's *semantic observation downsampling* keeps a compact set of frames
around the failure event by walking outward from the failure frame and
retaining a frame only when it is *semantically distinct* from the last frame
kept -- measured by CLIP-embedding cosine similarity.

This module provides:

* :func:`select_frames_around_failure_clip` -- the method used in the paper
  (CLIP cosine-similarity change-point selection, bidirectional from failure).
* :func:`select_frames_around_failure_pixel` -- a pixel-difference ablation
  baseline (same walk, similarity replaced by mean absolute pixel difference).
* :func:`sample_fixed_fps` -- fixed-rate subsampling, the other ablation
  baseline (Fig. "frame sampling ablation").

All selectors return frames as a chronologically ordered list of PIL images,
ready to hand to :func:`failure_taxonomy.llm.vlm_generate`.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("failure_taxonomy.frame_selection")

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"

# Lazily-loaded, process-wide CLIP singleton. The original scripts reloaded the
# model on every call (seconds of overhead per trajectory); we load it once.
_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_CLIP_DEVICE = None


def _get_clip():
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE
    if _CLIP_MODEL is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        _CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_CLIP_DEVICE)
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        logger.info("Loaded CLIP (%s) on %s", CLIP_MODEL_NAME, _CLIP_DEVICE)
    return _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE


# --------------------------------------------------------------------------- #
# Frame reading
# --------------------------------------------------------------------------- #
def _read_window(video_path: str, start_time: float, end_time: float, failure_time: float):
    """Read frames in ``[start_time, end_time]`` and locate the failure frame.

    Returns ``(pil_frames, relative_failure_index)`` where the index is into
    ``pil_frames``. The window is clamped to the video and always contains the
    failure frame.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fail_idx = max(0, min(total - 1, int(failure_time * fps)))
    start_idx = max(0, int(start_time * fps))
    end_idx = min(total, int(end_time * fps))
    start_idx = min(start_idx, fail_idx)
    end_idx = max(end_idx, fail_idx + 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start_idx:
        # Some codecs cannot seek precisely; fall back to sequential reads.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(start_idx):
            cap.read()

    frames: list[Image.Image] = []
    for _ in range(start_idx, end_idx):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    if not frames:
        return [], 0
    rel_fail = max(0, min(len(frames) - 1, fail_idx - start_idx))
    return frames, rel_fail


def _clip_embeddings(pil_frames: list[Image.Image], batch_size: int = 32) -> np.ndarray:
    """L2-normalised CLIP image embeddings for a list of PIL frames."""
    import torch

    model, processor, device = _get_clip()
    embeds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(pil_frames), batch_size):
            batch = pil_frames[i : i + batch_size]
            inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
            # Explicit vision-tower -> visual projection -> L2 normalise. (Equivalent
            # to CLIPModel.get_image_features but robust across transformers versions.)
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            feats = model.visual_projection(vision_outputs.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeds.append(feats.cpu().numpy())
    return np.concatenate(embeds, axis=0)


def _bidirectional_select(
    n: int,
    fail_idx: int,
    is_distinct,
    min_frame_gap: int,
    max_frames: int,
) -> list[int]:
    """Walk outward from ``fail_idx`` keeping distinct frames on both sides.

    ``is_distinct(candidate_idx, last_kept_idx)`` returns True when the
    candidate frame is semantically different enough from the last frame kept.
    The failure frame is always kept; the remaining budget is split between the
    backward (history) and forward (aftermath) passes.
    """
    backward: list[int] = []
    last = fail_idx
    limit_backward = (max_frames - 1) // 2
    for i in range(fail_idx - 1, -1, -1):
        if len(backward) >= limit_backward:
            break
        if last - i < min_frame_gap:
            continue
        if is_distinct(i, last):
            backward.append(i)
            last = i

    forward: list[int] = []
    last = fail_idx
    limit_forward = max_frames - 1 - len(backward)
    for i in range(fail_idx + 1, n):
        if len(forward) >= limit_forward:
            break
        if i - last < min_frame_gap:
            continue
        if is_distinct(i, last):
            forward.append(i)
            last = i

    return sorted(backward + [fail_idx] + forward)


# --------------------------------------------------------------------------- #
# Public selectors
# --------------------------------------------------------------------------- #
def select_frames_around_failure_clip(
    video_path: str,
    failure_time: float,
    start_time: float,
    end_time: float,
    similarity_threshold: float = 0.95,
    min_frame_gap: int = 1,
    max_frames: int = 20,
) -> list[Image.Image]:
    """Paper method: CLIP change-point selection centred on the failure frame.

    Starting from the failure frame, walk backward and forward; keep a frame
    when its cosine similarity to the last kept frame drops below
    ``similarity_threshold`` (i.e. the scene has changed semantically).

    Args:
        video_path: Path to the rollout video.
        failure_time: Timestamp (seconds) of the failure.
        start_time, end_time: Window around the failure to consider.
        similarity_threshold: Keep a frame if similarity < this value.
        min_frame_gap: Minimum index gap between kept frames.
        max_frames: Cap on the number of frames returned.

    Returns:
        Chronologically ordered selected frames (PIL images).
    """
    frames, rel_fail = _read_window(video_path, start_time, end_time, failure_time)
    if not frames:
        return []

    embeds = _clip_embeddings(frames)

    def is_distinct(i: int, last: int) -> bool:
        return float(np.dot(embeds[i], embeds[last])) < similarity_threshold

    indices = _bidirectional_select(len(frames), rel_fail, is_distinct, min_frame_gap, max_frames)
    logger.info("CLIP selection: %d frames around failure @ %.2fs", len(indices), failure_time)
    return [frames[i] for i in indices]


def select_frames_around_failure_pixel(
    video_path: str,
    failure_time: float,
    start_time: float,
    end_time: float,
    diff_threshold: float = 0.05,
    min_frame_gap: int = 5,
    max_frames: int = 20,
) -> list[Image.Image]:
    """Ablation baseline: same walk as the CLIP method but using raw pixel
    difference (mean absolute difference in [0, 1]) instead of CLIP similarity.
    """
    frames, rel_fail = _read_window(video_path, start_time, end_time, failure_time)
    if not frames:
        return []

    arrays = [np.asarray(f, dtype=np.float32) / 255.0 for f in frames]

    def is_distinct(i: int, last: int) -> bool:
        return float(np.mean(np.abs(arrays[i] - arrays[last]))) > diff_threshold

    indices = _bidirectional_select(len(frames), rel_fail, is_distinct, min_frame_gap, max_frames)
    logger.info("Pixel selection: %d frames around failure @ %.2fs", len(indices), failure_time)
    return [frames[i] for i in indices]


def sample_fixed_fps(
    video_path: str,
    target_fps: float,
    failure_time: Optional[float] = None,
    pre_window: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> list[Image.Image]:
    """Ablation baseline: fixed-rate subsampling.

    Samples the video at ``target_fps`` (e.g. 1.0, 0.5, 0.25). If
    ``failure_time`` and ``pre_window`` are given, only the ``pre_window``
    seconds ending at the failure are sampled.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if failure_time is not None and pre_window is not None:
        start_idx = max(0, int((failure_time - pre_window) * fps))
        end_idx = min(total, int(failure_time * fps) + 1)
    else:
        start_idx, end_idx = 0, total

    step = max(1, int(round(fps / target_fps)))
    frames: list[Image.Image] = []
    for fidx in range(start_idx, end_idx, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    if max_frames is not None and len(frames) > max_frames:
        frames = frames[-max_frames:]
    logger.info("Fixed %.2f fps: %d frames from %s", target_fps, len(frames), video_path)
    return frames
