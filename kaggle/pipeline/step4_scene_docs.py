"""
step4_scene_docs.py — Build Scene Documents (JSONL).

Merges shot boundaries, ASR extractions, and VLM captions into one
scenes.jsonl file for downstream indexing and retrieval.

scene_summary is bilingual: LLM/VLM caption (EN+RU) + ASR text,
optionally augmented with training queries mapped to same time range.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import SHOTS_FILE, WORK_DIR, SCENES_FILE, TRAIN_CSV


def _load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_train_augmentation() -> dict[str, list[str]]:
    """Map train queries to scenes by video_id + time overlap.

    Returns {scene_key: [query_text, ...]} for augmenting scene documents.
    """
    import pandas as pd

    if not TRAIN_CSV.exists():
        print("[step4] train_qa.csv not found, skipping train augmentation")
        return {}

    df = pd.read_csv(TRAIN_CSV)

    required = {"video_file", "start", "end"}
    if not required.issubset(set(df.columns)):
        print(f"[step4] train_qa.csv missing columns {required - set(df.columns)}, skipping")
        return {}

    # Build scene lookup from shot_boundaries.json
    shots: list[dict] = _load_json(SHOTS_FILE)
    scenes_by_video: dict[str, list[dict]] = defaultdict(list)
    for video_entry in shots:
        video_id = video_entry["video_id"]
        for scene in video_entry["scenes"]:
            scenes_by_video[video_id].append(scene)

    augmentations: dict[str, list[str]] = defaultdict(list)

    for _, row in df.iterrows():
        video_id = str(row["video_file"])
        q_start = float(row["start"])
        q_end = float(row["end"])

        # Collect query texts (support both column naming conventions)
        queries = []
        if "question_ru" in df.columns and pd.notna(row.get("question_ru")):
            queries.append(str(row["question_ru"]))
        elif "question" in df.columns and pd.notna(row.get("question")):
            queries.append(str(row["question"]))
        if "question_en" in df.columns and pd.notna(row.get("question_en")):
            q_en = str(row["question_en"])
            if not queries or q_en.lower() != queries[0].lower():
                queries.append(q_en)

        if not queries:
            continue

        # Find overlapping scenes
        for scene in scenes_by_video.get(video_id, []):
            overlap_start = max(scene["start"], q_start)
            overlap_end = min(scene["end"], q_end)
            if overlap_end > overlap_start:
                key = f"{video_id}__{scene['scene_idx']}"
                for q in queries:
                    if q not in augmentations[key]:
                        augmentations[key].append(q)

    total_aug = sum(len(v) for v in augmentations.values())
    print(f"[step4] Train augmentation: {total_aug} queries mapped to {len(augmentations)} scenes")
    return dict(augmentations)


def main() -> None:
    shots: list[dict] = _load_json(SHOTS_FILE)
    extractions: dict[str, dict] = _load_json(WORK_DIR / "extractions.json")
    captions: dict[str, dict] = _load_json(WORK_DIR / "captions.json")

    # Load train augmentation (optional)
    train_aug = _load_train_augmentation()

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
                llm_caption_en: str = cap.get("llm_caption_en", "") if isinstance(cap, dict) else ""
                llm_caption_ru: str = cap.get("llm_caption_ru", "") if isinstance(cap, dict) else ""

                # Build bilingual scene_summary:
                # priority: LLM caption (if available) > VLM caption, + ASR text
                parts = []
                if llm_caption_en:
                    parts.append(llm_caption_en)
                elif vlm_caption:
                    parts.append(vlm_caption)
                if llm_caption_ru:
                    parts.append(llm_caption_ru)
                if asr_text:
                    parts.append(asr_text)

                # Synthetic doc2query queries
                syn_queries = cap.get("synthetic_queries", []) if isinstance(cap, dict) else []
                if syn_queries:
                    parts.append(" | ".join(syn_queries))

                # Train query augmentation
                aug_queries = train_aug.get(key, [])
                if aug_queries:
                    parts.append(" | ".join(aug_queries))

                scene_summary = "\n".join(parts) if parts else ""

                doc = {
                    "video_id": video_id,
                    "scene_idx": scene_idx,
                    "start": scene["start"],
                    "end": scene["end"],
                    "asr_text": asr_text,
                    "vlm_caption": vlm_caption,
                    "llm_caption_en": llm_caption_en,
                    "llm_caption_ru": llm_caption_ru,
                    "scene_summary": scene_summary,
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1

    print(f"[step4] Wrote {count} scene docs to {SCENES_FILE}")


if __name__ == "__main__":
    main()
