"""
step5_event_docs.py — Build Event Documents (Time-Based Sliding Window).

Groups consecutive scenes into overlapping event windows based on time
(w=110s, step=10s) and concatenates their summaries for coarse-grained retrieval.
"""

import json
from collections import defaultdict

from .config import SCENES_FILE, EVENTS_FILE, EVENT_WINDOW_SEC, EVENT_STRIDE_SEC


def _load_scenes() -> list[dict]:
    scenes: list[dict] = []
    with open(SCENES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenes.append(json.loads(line))
    return scenes


def main() -> None:
    scenes = _load_scenes()

    # Group scenes by video_id, preserving order.
    by_video: dict[str, list[dict]] = defaultdict(list)
    for scene in scenes:
        by_video[scene["video_id"]].append(scene)

    event_idx = 0
    with open(EVENTS_FILE, "w", encoding="utf-8") as out:
        for video_id, video_scenes in by_video.items():
            video_scenes.sort(key=lambda s: s["scene_idx"])
            if not video_scenes:
                continue

            video_start = video_scenes[0]["start"]
            video_end = max(s["end"] for s in video_scenes)

            # Time-based sliding window: w=EVENT_WINDOW_SEC, step=EVENT_STRIDE_SEC
            win_start = video_start
            while win_start < video_end:
                win_end = win_start + EVENT_WINDOW_SEC

                # Collect scenes that overlap with this time window
                window = [
                    s for s in video_scenes
                    if s["end"] > win_start and s["start"] < win_end
                ]

                if not window:
                    win_start += EVENT_STRIDE_SEC
                    continue

                # Build event_summary from scene_summary (bilingual: caption + ASR).
                summaries = [s["scene_summary"] for s in window if s.get("scene_summary")]
                event_summary = " ".join(summaries)
                if not event_summary:
                    win_start += EVENT_STRIDE_SEC
                    continue

                mid = len(window) // 2
                doc = {
                    "video_id": video_id,
                    "event_idx": event_idx,
                    "start": win_start,
                    "end": win_end,
                    "center_scene_idx": window[mid]["scene_idx"],
                    "scene_indices": [s["scene_idx"] for s in window],
                    "event_summary": event_summary,
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                event_idx += 1
                win_start += EVENT_STRIDE_SEC

    print(f"[step5] Wrote {event_idx} event docs to {EVENTS_FILE} "
          f"(w={EVENT_WINDOW_SEC}s, step={EVENT_STRIDE_SEC}s)")


if __name__ == "__main__":
    main()
