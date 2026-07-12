"""Stage 1: Failure reasoning.

Turn each failure trajectory into a natural-language ``(trajectory,
failure_reason)`` description using a vision-language model (Gemini 2.5 Pro by
default). One script covers all three case studies:

* ``driving`` / ``navigation`` -- the failure log is a directory of trajectory
  sub-folders, each holding an ordered image sequence. The last ``--max-frames``
  frames are sent to the VLM with a fixed chain-of-thought reasoning prompt.

* ``manipulation`` -- the failure log is a set of task videos described by a
  tasks JSON (REFLECT format). For each task, *semantic observation
  downsampling* selects frames around the known failure timestamp, and a
  task-conditioned prompt (task name, success condition, action plan) is used.

Output is a JSONL file of records::

    {"filename": ..., "trajectory": ..., "failure_reason": ...,
     "ground_truth": ...  # manipulation only, when available}

which is the input to Stage 2 (``step2_taxonomy_discovery``).

Examples
--------
    # Navigation / driving: point at a folder of trajectory sub-directories
    python get_descriptions.py --case-study navigation \\
        --input /path/to/waypointnav/failure_trajectories \\
        --output ../results/navigation/descriptions.jsonl

    # Manipulation: point at the REFLECT dataset and its tasks JSON
    python get_descriptions.py --case-study manipulation \\
        --input /path/to/reflect_dataset/real_data \\
        --tasks-json /path/to/reflect_dataset/tasks_real_world.json \\
        --output ../results/manipulation/descriptions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Make the shared library importable regardless of the working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from failure_taxonomy import config, io_utils, prompts  # noqa: E402
from failure_taxonomy.llm import vlm_generate  # noqa: E402


# --------------------------------------------------------------------------- #
# Driving / navigation: image-sequence trajectories
# --------------------------------------------------------------------------- #
def run_image_sequences(args, cfg) -> None:
    prompt = prompts.reasoning_prompt(args.case_study)
    done = io_utils.processed_ids(args.output, key="filename")

    trajectories = list(io_utils.iter_trajectory_dirs(args.input))
    if args.limit:
        trajectories = trajectories[: args.limit]

    for i, (name, path) in enumerate(trajectories):
        if name in done:
            print(f"[{i + 1}/{len(trajectories)}] skip {name} (already processed)")
            continue

        frames = io_utils.load_images(path, last_n=args.max_frames)
        if not frames:
            print(f"[{i + 1}/{len(trajectories)}] {name}: no frames, skipping")
            continue

        print(f"[{i + 1}/{len(trajectories)}] {name}: {len(frames)} frames -> {args.model}")
        text = vlm_generate(prompt, images=frames, model=args.model)
        trajectory, reason = io_utils.parse_trajectory_and_reason(text)

        io_utils.append_record(
            args.output,
            {"filename": name, "trajectory": trajectory, "failure_reason": reason},
        )
        time.sleep(args.sleep)


# --------------------------------------------------------------------------- #
# Manipulation: task videos + semantic frame selection
# --------------------------------------------------------------------------- #
def _parse_failure_time(value):
    """Parse a failure timestamp given as seconds or an ``MM:SS`` string."""
    if isinstance(value, list):
        value = value[-1]
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and ":" in value:
        minutes, seconds = value.split(":")
        return int(minutes) * 60 + int(seconds)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_manipulation_frames(video_path, failure_time, args):
    """Dispatch to the requested frame-selection strategy."""
    from failure_taxonomy import frame_selection as fs

    if args.frame_selection == "clip":
        return fs.select_frames_around_failure_clip(
            video_path, failure_time,
            start_time=max(0, failure_time - args.pre_window),
            end_time=failure_time + args.post_window,
            similarity_threshold=args.similarity_threshold,
            min_frame_gap=args.min_frame_gap, max_frames=args.max_frames,
        )
    if args.frame_selection == "pixel":
        return fs.select_frames_around_failure_pixel(
            video_path, failure_time,
            start_time=max(0, failure_time - args.pre_window),
            end_time=failure_time + args.post_window,
            min_frame_gap=args.min_frame_gap, max_frames=args.max_frames,
        )
    # fixed-fps ablation baseline
    return fs.sample_fixed_fps(
        video_path, target_fps=args.fps,
        failure_time=failure_time, pre_window=args.pre_window,
        max_frames=args.max_frames,
    )


def run_manipulation(args, cfg) -> None:
    with open(args.tasks_json, "r") as f:
        tasks = json.load(f)

    done = io_utils.processed_ids(args.output, key="filename")
    task_ids = list(tasks.keys())
    if args.limit:
        task_ids = task_ids[: args.limit]

    for i, task_id in enumerate(task_ids):
        info = tasks[task_id]
        if task_id in done:
            print(f"[{i + 1}/{len(task_ids)}] skip {task_id} (already processed)")
            continue

        folder = info.get("general_folder_name", task_id)
        video_path = os.path.join(args.input, folder, "videos/color.mp4")
        if not os.path.exists(video_path):
            print(f"[{i + 1}/{len(task_ids)}] {task_id}: missing video {video_path}")
            continue

        failure_time = _parse_failure_time(info.get("gt_failure_step"))
        if failure_time is None:
            print(f"[{i + 1}/{len(task_ids)}] {task_id}: no failure timestamp, skipping")
            continue

        frames = _select_manipulation_frames(video_path, failure_time, args)
        if not frames:
            print(f"[{i + 1}/{len(task_ids)}] {task_id}: no frames selected, skipping")
            continue

        actions = info.get("actions", [])
        action_sequence = "\n".join(f"{j + 1}. {a}" for j, a in enumerate(actions))
        prompt = prompts.reasoning_prompt(
            "manipulation",
            task_name=info["name"],
            success_condition=info["success_condition"],
            action_sequence=action_sequence,
        )

        print(f"[{i + 1}/{len(task_ids)}] {task_id}: {len(frames)} frames -> {args.model}")
        text = vlm_generate(prompt, images=frames, model=args.model)
        trajectory, reason = io_utils.parse_trajectory_and_reason(text)

        io_utils.append_record(
            args.output,
            {
                "filename": task_id,
                "trajectory": trajectory,
                "failure_reason": reason,
                "ground_truth": info.get("gt_failure_reason", ""),
            },
        )
        time.sleep(args.sleep)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case-study", required=True, choices=list(config.CASE_STUDIES),
                   help="Which system's failure log to process.")
    p.add_argument("--input", required=True,
                   help="Failure-log directory (trajectory sub-folders, or manipulation dataset root).")
    p.add_argument("--output", required=True, help="Output JSONL path for descriptions.")
    p.add_argument("--model", default=None, help="VLM model id (default: per case study).")
    p.add_argument("--max-frames", type=int, default=None, help="Max frames per trajectory.")
    p.add_argument("--limit", type=int, default=None, help="Process at most this many trajectories.")
    p.add_argument("--sleep", type=float, default=10.0, help="Seconds to sleep between calls (rate limiting).")

    # Manipulation-only options
    p.add_argument("--tasks-json", default=None, help="[manipulation] Path to the tasks JSON.")
    p.add_argument("--frame-selection", choices=["clip", "pixel", "fps"], default="clip",
                   help="[manipulation] Frame downsampling strategy (clip = paper method).")
    p.add_argument("--similarity-threshold", type=float, default=0.95,
                   help="[manipulation/clip] Keep a frame if CLIP similarity < this.")
    p.add_argument("--min-frame-gap", type=int, default=1, help="[manipulation] Min gap between kept frames.")
    p.add_argument("--pre-window", type=float, default=15.0, help="[manipulation] Seconds before failure.")
    p.add_argument("--post-window", type=float, default=10.0, help="[manipulation] Seconds after failure.")
    p.add_argument("--fps", type=float, default=1.0, help="[manipulation/fps] Fixed sampling rate.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = config.get_config(args.case_study)
    if args.model is None:
        args.model = cfg.reasoning_model
    if args.max_frames is None:
        args.max_frames = cfg.max_frames

    print(f"Case study : {cfg.display_name}")
    print(f"Model      : {args.model}")
    print(f"Input      : {args.input}")
    print(f"Output     : {args.output}")

    if args.case_study == "manipulation":
        if not args.tasks_json:
            raise SystemExit("--tasks-json is required for the manipulation case study.")
        run_manipulation(args, cfg)
    else:
        run_image_sequences(args, cfg)

    print(f"Done. Descriptions written to {args.output}")


if __name__ == "__main__":
    main()
