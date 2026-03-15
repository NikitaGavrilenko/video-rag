"""
step6_index.py — Create BGE-M3 embeddings, build FAISS-GPU indices, and BM25.

Reads scenes.jsonl and events.jsonl, encodes with BGE-M3 (dense + sparse),
builds FAISS inner-product indices on GPU, saves metadata/sparse vectors
as pickle, and builds BM25 indices over text fields.
"""

import json
import pickle
from typing import Any

import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi

from .config import (
    BGE_BATCH_SIZE,
    BGE_MODEL,
    BM25_EVENTS_FILE,
    BM25_SCENES_FILE,
    EVENTS_FILE,
    EVENTS_META_FILE,
    FAISS_EVENTS_INDEX,
    FAISS_SCENES_INDEX,
    SCENES_FILE,
    SCENES_META_FILE,
    SPARSE_EVENTS_FILE,
    SPARSE_SCENES_FILE,
)


def _load_jsonl(path: str | Any) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
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
    """Encode texts with BGE-M3, returning dense vectors and sparse weight dicts."""
    output = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
    )
    dense_vectors: np.ndarray = output["dense_vecs"]  # already np.ndarray

    # Convert sparse weights: token_id (int) -> str key for compatibility.
    sparse_vectors: list[dict[str, float]] = []
    for sparse_dict in output["lexical_weights"]:
        sparse_vectors.append({str(k): float(v) for k, v in sparse_dict.items()})

    return dense_vectors, sparse_vectors


def _build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """Build a FAISS IndexFlatIP (CPU — fast enough for ~30K vectors)."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    print(f"  Adding {vectors.shape[0]} vectors (dim={dim})...")
    index.add(vectors.astype(np.float32))
    return index


def _build_scene_metadata(docs: list[dict]) -> list[dict]:
    """Extract metadata dicts for scenes, parallel to FAISS index order."""
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
    """Extract metadata dicts for events, parallel to FAISS index order."""
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


def _build_bm25(texts: list[str]) -> BM25Okapi:
    """Tokenise texts by whitespace and build a BM25Okapi index."""
    tokenized = [text.lower().split() for text in texts]
    return BM25Okapi(tokenized)


def _save_pickle(obj: Any, path: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {path}")


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

    # ── Build FAISS indices on GPU ───────────────────────────────────────────
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

    # ── Build and save BM25 indices ──────────────────────────────────────────
    print("[step6] Building BM25 index over scene asr_text...")
    scene_asr_texts = [doc.get("asr_text", "") for doc in scenes]
    bm25_scenes = _build_bm25(scene_asr_texts)
    _save_pickle(bm25_scenes, BM25_SCENES_FILE)

    print("[step6] Building BM25 index over event summaries...")
    event_summary_texts = [doc.get("event_summary", "") for doc in events]
    bm25_events = _build_bm25(event_summary_texts)
    _save_pickle(bm25_events, BM25_EVENTS_FILE)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(
        f"[step6] Done. "
        f"{scenes_index.ntotal} scenes + {events_index.ntotal} events indexed. "
        f"BM25 built for both."
    )


if __name__ == "__main__":
    main()
