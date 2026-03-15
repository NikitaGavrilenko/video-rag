"""
step6_index.py — Create BGE-M3 embeddings and build FAISS indices.

Reads scenes.jsonl and events.jsonl, encodes with BGE-M3 (dense + sparse),
builds FAISS inner-product indices (CPU), saves metadata/sparse vectors as pickle.
Also encodes train queries for train→test matching at search time.
"""

import json
import pickle
from typing import Any

import faiss
import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel

from .config import (
    BGE_BATCH_SIZE,
    BGE_MODEL,
    EVENTS_FILE,
    EVENTS_META_FILE,
    FAISS_EVENTS_INDEX,
    FAISS_SCENES_INDEX,
    SCENES_FILE,
    SCENES_META_FILE,
    SPARSE_EVENTS_FILE,
    SPARSE_SCENES_FILE,
    TRAIN_CSV,
    WORK_DIR,
)

TRAIN_INDEX_FILE = WORK_DIR / "faiss_train.index"
TRAIN_META_FILE = WORK_DIR / "train_meta.pkl"


def _load_jsonl(path: str | Any) -> list[dict]:
    docs: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def _encode(
    model: BGEM3FlagModel, texts: list[str], batch_size: int
) -> tuple[np.ndarray, list[dict[str, float]]]:
    output = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
    )
    dense_vectors: np.ndarray = output["dense_vecs"]
    sparse_vectors: list[dict[str, float]] = []
    for sparse_dict in output["lexical_weights"]:
        sparse_vectors.append({str(k): float(v) for k, v in sparse_dict.items()})
    return dense_vectors, sparse_vectors


def _build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    print(f"  Adding {vectors.shape[0]} vectors (dim={dim})...")
    index.add(vectors.astype(np.float32))
    return index


def _build_scene_metadata(docs: list[dict]) -> list[dict]:
    meta: list[dict] = []
    for doc in docs:
        meta.append({
            "video_id": doc["video_id"],
            "scene_idx": doc["scene_idx"],
            "start": doc["start"],
            "end": doc["end"],
            "asr_text": doc.get("asr_text", ""),
            "summary": doc.get("scene_summary", ""),
            "llm_caption_en": doc.get("llm_caption_en", ""),
            "llm_caption_ru": doc.get("llm_caption_ru", ""),
        })
    return meta


def _build_event_metadata(docs: list[dict]) -> list[dict]:
    meta: list[dict] = []
    for doc in docs:
        meta.append({
            "video_id": doc["video_id"],
            "event_idx": doc["event_idx"],
            "start": doc["start"],
            "end": doc["end"],
            "event_summary": doc.get("event_summary", ""),
            "center_scene_idx": doc.get("center_scene_idx", 0),
            "scene_indices": doc.get("scene_indices", []),
        })
    return meta


def _save_pickle(obj: Any, path: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {path}")


def _build_train_index(model: BGEM3FlagModel) -> None:
    """Encode train queries and build a FAISS index for train→test matching."""
    if not TRAIN_CSV.exists():
        print("[step6] train_qa.csv not found, skipping train index")
        return

    df = pd.read_csv(TRAIN_CSV)
    required = {"question", "video_file", "start", "end"}
    if not required.issubset(set(df.columns)):
        print(f"[step6] train_qa.csv missing columns {required - set(df.columns)}, skipping")
        return

    # Build train metadata + query texts
    train_meta: list[dict] = []
    train_texts: list[str] = []

    for _, row in df.iterrows():
        question = str(row["question"])
        video_file = str(row["video_file"])
        start = float(row["start"])
        end = float(row["end"])

        # Also include English translation if available
        question_en = str(row.get("question_en", "")) if pd.notna(row.get("question_en")) else ""
        combined = question
        if question_en and question_en.lower() != question.lower():
            combined = f"{question} {question_en}"

        train_meta.append({
            "video_id": video_file,
            "start": start,
            "end": end,
            "question": question,
            "question_en": question_en,
        })
        train_texts.append(combined)

    print(f"[step6] Encoding {len(train_texts)} train queries...")
    train_dense, _ = _encode(model, train_texts, BGE_BATCH_SIZE)

    train_index = _build_faiss_index(train_dense)
    faiss.write_index(train_index, str(TRAIN_INDEX_FILE))
    print(f"  Saved train FAISS index ({train_index.ntotal} vectors) -> {TRAIN_INDEX_FILE}")

    _save_pickle(train_meta, TRAIN_META_FILE)


def main() -> None:
    # ── Load documents ───────────────────────────────────────────────────────
    print("[step6] Loading scene and event documents...")
    scenes = _load_jsonl(SCENES_FILE)
    events = _load_jsonl(EVENTS_FILE)
    print(f"  Loaded {len(scenes)} scenes, {len(events)} events")

    # ── BGE-M3 embedding ─────────────────────────────────────────────────────
    print(f"[step6] Loading BGE-M3 model ({BGE_MODEL})...")
    model = BGEM3FlagModel(BGE_MODEL, use_fp16=True)

    print(f"[step6] Encoding {len(scenes)} scene summaries (batch_size={BGE_BATCH_SIZE})...")
    scene_texts = [doc.get("scene_summary", "") for doc in scenes]
    scene_dense, scene_sparse = _encode(model, scene_texts, BGE_BATCH_SIZE)
    print(f"  Scene embeddings: {scene_dense.shape[0]} dense, {len(scene_sparse)} sparse")

    print(f"[step6] Encoding {len(events)} event summaries (batch_size={BGE_BATCH_SIZE})...")
    event_texts = [doc.get("event_summary", "") for doc in events]
    event_dense, event_sparse = _encode(model, event_texts, BGE_BATCH_SIZE)
    print(f"  Event embeddings: {event_dense.shape[0]} dense, {len(event_sparse)} sparse")

    # ── Build FAISS indices ──────────────────────────────────────────────────
    print("[step6] Building FAISS scenes index...")
    scenes_index = _build_faiss_index(scene_dense)
    faiss.write_index(scenes_index, str(FAISS_SCENES_INDEX))
    print(f"  Saved FAISS scenes index ({scenes_index.ntotal} vectors) -> {FAISS_SCENES_INDEX}")

    print("[step6] Building FAISS events index...")
    events_index = _build_faiss_index(event_dense)
    faiss.write_index(events_index, str(FAISS_EVENTS_INDEX))
    print(f"  Saved FAISS events index ({events_index.ntotal} vectors) -> {FAISS_EVENTS_INDEX}")

    # ── Save metadata pickles ────────────────────────────────────────────────
    print("[step6] Saving metadata...")
    scenes_meta = _build_scene_metadata(scenes)
    _save_pickle(scenes_meta, SCENES_META_FILE)

    events_meta = _build_event_metadata(events)
    _save_pickle(events_meta, EVENTS_META_FILE)

    # ── Save sparse vectors ──────────────────────────────────────────────────
    print("[step6] Saving sparse vectors...")
    _save_pickle(scene_sparse, SPARSE_SCENES_FILE)
    _save_pickle(event_sparse, SPARSE_EVENTS_FILE)

    # ── Train query index for train→test matching ────────────────────────────
    print("[step6] Building train query index...")
    _build_train_index(model)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(
        f"[step6] Done. "
        f"{scenes_index.ntotal} scenes + {events_index.ntotal} events indexed."
    )


if __name__ == "__main__":
    main()
