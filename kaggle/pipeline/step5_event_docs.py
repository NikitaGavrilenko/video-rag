"""
step5_event_docs.py — Build Event Documents (Sliding Window over Scenes).

Groups consecutive scenes into overlapping event windows and concatenates
their VLM captions into a single event_summary for coarse-grained retrieval.
"""

import json
from collections import defaultdict

from .config import SCENES_FILE, EVENTS_FILE, EVENT_WINDOW_SIZE, EVENT_WINDOW_STRIDE


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
            # Sort by scene_idx to guarantee correct order.
            video_scenes.sort(key=lambda s: s["scene_idx"])
            n = len(video_scenes)

            for win_start in range(0, n, EVENT_WINDOW_STRIDE):
                window = video_scenes[win_start : win_start + EVENT_WINDOW_SIZE]

                # Build event_summary from scene_summary (bilingual: caption + ASR).
                summaries = [s["scene_summary"] for s in window if s.get("scene_summary")]
                event_summary = " ".join(summaries)
                if not event_summary:
                    continue

                mid = len(window) // 2
                doc = {
                    "video_id": video_id,
                    "event_idx": event_idx,
                    "start": min(s["start"] for s in window),
                    "end": max(s["end"] for s in window),
                    "center_scene_idx": window[mid]["scene_idx"],
                    "scene_indices": [s["scene_idx"] for s in window],
                    "event_summary": event_summary,
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                event_idx += 1

    print(f"[step5] Wrote {event_idx} event docs to {EVENTS_FILE}")


if __name__ == "__main__":
    main()
