"""
step1_shots.py — Shot boundary detection via TransNetV2.

Detects shots for every video in VIDEO_DIR, optionally merges with a
teammate's pre-computed shot file (teammate takes priority), filters
micro-shots, computes keyframe midpoints, and saves the result to
SHOTS_FILE as JSON.

Supports resume: already-processed videos are skipped on restart.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import cv2

from .config import (
    MIN_SHOT_DURATION,
    SHOTS_FILE,
    TEAMMATE_SHOTS_FILE,
    VIDEO_DIR,
)

# ── Constants ────────────────────────────────────────────────────────────────
BATCH_SAVE_EVERY: int = 10
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mkv", ".webm", ".opus")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_video_files() -> list[Path]:
    """Return sorted list of video files in VIDEO_DIR."""
    files: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(VIDEO_DIR.glob(f"*{ext}"))
    return sorted(files)


def _get_fps(video_path: Path) -> float:
    """Read FPS from video file via OpenCV; default to 25.0 on failure."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return float(fps)


def _build_scene(
    idx: int,
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    """Build a single scene dict."""
    return {
        "scene_idx": idx,
        "start": round(start_sec, 3),
        "end": round(end_sec, 3),
        "description": None,
        "description_en": None,
        "keyframe_time": round((start_sec + end_sec) / 2, 3),
        "keyframe_path": None,
    }


def _filter_micro_shots(
    scenes: list[dict[str, Any]],
    min_duration: float,
) -> tuple[list[dict[str, Any]], int]:
    """Remove shots shorter than *min_duration* seconds.

    Returns the filtered list (with re-indexed scene_idx) and the count
    of removed micro-shots.
    """
    kept: list[dict[str, Any]] = []
    removed = 0
    for s in scenes:
        duration = s["end"] - s["start"]
        if duration >= min_duration:
            kept.append(s)
        else:
            removed += 1

    # Re-index after filtering
    for new_idx, s in enumerate(kept):
        s["scene_idx"] = new_idx

    return kept, removed


def _save_results(results: list[dict[str, Any]], path: Path) -> None:
    """Atomically write results to JSON."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _load_existing(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Load previously saved results for resume support."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            results: list[dict[str, Any]] = json.load(f)
        done = {r["video_id"] for r in results}
        return results, done
    return [], set()


def _load_teammate_shots(path: Path) -> dict[str, dict[str, Any]]:
    """Load teammate shot file and index by video_id."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
    return {entry["video_id"]: entry for entry in data}


# ── TransNetV2 detection ────────────────────────────────────────────────────

def _detect_shots_transnet(
    video_path: Path,
    model: Any,
) -> list[dict[str, Any]]:
    """Run TransNetV2 on a single video and return list of scene dicts."""
    fps = _get_fps(video_path)

    _video_frames, single_frame_predictions, _ = model.predict_video(str(video_path))
    predictions = (
        single_frame_predictions.numpy()
        if hasattr(single_frame_predictions, "numpy")
        else single_frame_predictions
    )
    scenes_frames = model.predictions_to_scenes(predictions)

    scenes: list[dict[str, Any]] = []
    for idx, (start_frame, end_frame) in enumerate(scenes_frames):
        start_sec = int(start_frame) / fps
        end_sec = int(end_frame) / fps
        scenes.append(_build_scene(idx, start_sec, end_sec))

    return scenes


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run Step 1: shot boundary detection + merge + filter."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from transnetv2_pytorch import TransNetV2

    t0 = time.time()

    # --- Resume -----------------------------------------------------------
    results, done = _load_existing(SHOTS_FILE)
    print(f"[step1] Resume: {len(done)} videos already processed")

    # --- Video list -------------------------------------------------------
    video_files = _get_video_files()
    print(f"[step1] Found {len(video_files)} video files in {VIDEO_DIR}")

    # --- Teammate shots (priority merge) ----------------------------------
    teammate: dict[str, dict[str, Any]] = _load_teammate_shots(TEAMMATE_SHOTS_FILE)
    if teammate:
        print(f"[step1] Loaded teammate shots for {len(teammate)} videos")

    # --- Model ------------------------------------------------------------
    # Only load model if there are videos to process via TransNetV2
    videos_needing_transnet = [
        vp for vp in video_files
        if vp.stem not in done and vp.stem not in teammate
    ]
    model = TransNetV2(device="cuda") if videos_needing_transnet else None

    # --- Stats accumulators -----------------------------------------------
    total_scenes = sum(len(r["scenes"]) for r in results)
    total_micro_filtered = 0
    newly_processed = 0

    # --- Process each video -----------------------------------------------
    for i, video_path in enumerate(video_files):
        video_id = video_path.stem

        if video_id in done:
            continue

        print(f"[step1] [{len(done) + 1}/{len(video_files)}] {video_id}", end=" ")

        try:
            if video_id in teammate:
                # Use teammate's shots (priority)
                entry = teammate[video_id]
                scenes_raw: list[dict[str, Any]] = entry["scenes"]
                source = "teammate"
            else:
                # Run TransNetV2
                assert model is not None
                scenes_raw = _detect_shots_transnet(video_path, model)
                source = "transnet"

            # Filter micro-shots
            scenes_filtered, n_micro = _filter_micro_shots(
                scenes_raw, MIN_SHOT_DURATION
            )
            total_micro_filtered += n_micro

            # Ensure keyframe_time is present (teammate data may lack it)
            for s in scenes_filtered:
                if s.get("keyframe_time") is None:
                    s["keyframe_time"] = round((s["start"] + s["end"]) / 2, 3)

            results.append({
                "video_id": video_id,
                "audio_key": f"videos/{video_path.name}",
                "scenes": scenes_filtered,
            })
            done.add(video_id)
            newly_processed += 1
            total_scenes += len(scenes_filtered)

            print(
                f"-> {len(scenes_filtered)} shots ({source})"
                + (f", {n_micro} micro-shots removed" if n_micro else "")
            )

        except Exception as exc:
            print(f"-> ERROR: {exc}")

        # Batch save
        if newly_processed > 0 and newly_processed % BATCH_SAVE_EVERY == 0:
            _save_results(results, SHOTS_FILE)
            print(f"  [batch save: {len(done)} videos]")

    # --- Final save -------------------------------------------------------
    _save_results(results, SHOTS_FILE)

    elapsed = time.time() - t0
    print(
        f"\n[step1] Done in {elapsed:.1f}s"
        f" | {len(results)} videos"
        f" | {total_scenes} total scenes"
        f" | {total_micro_filtered} micro-shots filtered"
        f" | saved to {SHOTS_FILE}"
    )


if __name__ == "__main__":
    main()
