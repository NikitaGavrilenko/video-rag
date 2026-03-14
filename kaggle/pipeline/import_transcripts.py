"""
import_transcripts.py — Merge teammate transcriptions into extractions.json.

Usage:
    python -m kaggle.pipeline.import_transcripts /path/to/transcripts.json

Reads shot_boundaries.json for scene timestamps, maps teammate ASR segments
to scenes using overlap logic, and updates extractions.json so the main
pipeline can skip Whisper for those videos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import ASR_OVERLAP_SEC, SHOTS_FILE, WORK_DIR

EXTRACTIONS_FILE = WORK_DIR / "extractions.json"


def _overlap_fraction(
    seg_start: float, seg_end: float, sc_start: float, sc_end: float
) -> float:
    seg_dur = seg_end - seg_start
    if seg_dur <= 0:
        return 0.0
    overlap = max(0.0, min(seg_end, sc_end) - max(seg_start, sc_start))
    return overlap / seg_dur


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m kaggle.pipeline.import_transcripts <transcripts.json>")
        sys.exit(1)

    transcripts_path = Path(sys.argv[1])

    # ── Load inputs ───────────────────────────────────────────────────────
    with open(transcripts_path, "r", encoding="utf-8") as f:
        raw: dict[str, list[dict]] = json.load(f)

    with open(SHOTS_FILE, "r", encoding="utf-8") as f:
        shots: list[dict] = json.load(f)

    extractions: dict[str, dict] = {}
    if EXTRACTIONS_FILE.exists():
        with open(EXTRACTIONS_FILE, "r", encoding="utf-8") as f:
            extractions = json.load(f)

    # ── Normalize keys: "videos/video_xxx.opus" → "video_xxx" ─────────────
    transcripts: dict[str, list[dict]] = {}
    for key, segments in raw.items():
        video_id = Path(key).stem  # "videos/video_xxx.opus" → "video_xxx"
        transcripts[video_id] = segments

    print(f"[import] Loaded {len(transcripts)} videos from {transcripts_path}")
    print(f"[import] Existing extractions: {len(extractions)}")

    # ── Map segments to scenes ────────────────────────────────────────────
    added = 0
    skipped = 0

    for video_entry in shots:
        video_id: str = video_entry["video_id"]
        if video_id not in transcripts:
            continue

        segments = transcripts[video_id]

        for scene in video_entry["scenes"]:
            scene_idx: int = scene["scene_idx"]
            key = f"{video_id}__{scene_idx}"

            sc_start: float = scene["start"]
            sc_end: float = scene["end"]
            asr_start = max(0.0, sc_start - ASR_OVERLAP_SEC)
            asr_end = sc_end + ASR_OVERLAP_SEC

            matched_texts: list[str] = []
            for seg in segments:
                if _overlap_fraction(seg["start"], seg["end"], asr_start, asr_end) > 0.5:
                    text = seg["text"].strip()
                    if text:
                        matched_texts.append(text)

            asr_text = " ".join(matched_texts)

            if key in extractions:
                extractions[key]["asr_text"] = asr_text
            else:
                extractions[key] = {"asr_text": asr_text, "keyframe_path": ""}
            added += 1

    # ── Save ──────────────────────────────────────────────────────────────
    with open(EXTRACTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(extractions, f, ensure_ascii=False, indent=2)

    matched_videos = sum(1 for v in shots if v["video_id"] in transcripts)
    print(f"[import] Updated {added} scenes from {matched_videos} videos")
    print(f"[import] Total extractions: {len(extractions)}")


if __name__ == "__main__":
    main()
