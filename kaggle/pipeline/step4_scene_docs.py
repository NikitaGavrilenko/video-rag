"""
step4_scene_docs.py — Build Scene Documents (JSONL).

Merges shot boundaries, ASR extractions, and VLM captions into one
scenes.jsonl file for downstream indexing and retrieval.
"""

import json
from pathlib import Path

from .config import SHOTS_FILE, WORK_DIR, SCENES_FILE


def _load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    shots: list[dict] = _load_json(SHOTS_FILE)
    extractions: dict[str, str] = _load_json(WORK_DIR / "extractions.json")
    captions: dict[str, str] = _load_json(WORK_DIR / "captions.json")

    count = 0
    with open(SCENES_FILE, "w", encoding="utf-8") as out:
        for video_entry in shots:
            video_id: str = video_entry["video_id"]
            for scene in video_entry["scenes"]:
                scene_idx: int = scene["scene_idx"]
                key = f"{video_id}__{scene_idx}"

                ext = extractions.get(key, {})
                cap = captions.get(key, {})
                asr_text: str = ext.get("asr_text", "") if isinstance(ext, dict) else ""
                vlm_caption: str = cap.get("vlm_caption", "") if isinstance(cap, dict) else ""

                # VLM caption is preferred; fall back to ASR if empty.
                scene_summary = vlm_caption if vlm_caption else asr_text

                doc = {
                    "video_id": video_id,
                    "scene_idx": scene_idx,
                    "start": scene["start"],
                    "end": scene["end"],
                    "asr_text": asr_text,
                    "vlm_caption": vlm_caption,
                    "scene_summary": scene_summary,
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1

    print(f"[step4] Wrote {count} scene docs to {SCENES_FILE}")


if __name__ == "__main__":
    main()
