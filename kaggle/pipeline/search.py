"""
search.py — Hybrid search with train→test matching + multi-query.

Flow:
  1. Load translated_data.csv (corrected_question + translated_question)
  2. For each test query:
     a. Check train index — if cosine sim > threshold, inject train answer
     b. Encode BOTH corrected (RU) and translated (EN) versions
     c. Search: FAISS dense + sparse × scenes/events × 2 queries
  3. RRF merge across all channels (including train matches)
  4. Dedup by (video_id, overlapping timecodes) — IoU > 50%
  5. BGE reranker — language-aware: RU query vs RU caption, EN query vs EN caption
  6. Event -> scene resolution
  7. Return top FINAL_TOP_N results
"""

from __future__ import annotations

import json
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    BGE_MODEL,
    EVENTS_META_FILE,
    FAISS_EVENTS_INDEX,
    FAISS_SCENES_INDEX,
    FINAL_TOP_N,
    RERANKER_MODEL,
    RERANKER_FINETUNED,
    RERANKER_OUTPUT_K,
    RERANKER_TOP_K,
    RRF_K,
    SCENES_FILE,
    SCENES_META_FILE,
    SEARCH_TOP_K_DENSE,
    SEARCH_TOP_K_SPARSE,
    SPARSE_SCENES_FILE,
    SPARSE_EVENTS_FILE,
    TEST_CSV,
    TRANSLATED_CSV,
    WORK_DIR,
)

# Train index paths (built by step6)
TRAIN_INDEX_FILE = WORK_DIR / "faiss_train.index"
TRAIN_META_FILE = WORK_DIR / "train_meta.pkl"

# Train→test matching thresholds
TRAIN_MATCH_HIGH = 0.85   # direct answer — inject at top with high RRF boost
TRAIN_MATCH_LOW = 0.70    # candidate — add to reranker pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RU_RE = re.compile(r"[а-яА-ЯёЁ]")


def _is_russian(text: str) -> bool:
    return bool(_RU_RE.search(text))


def _get_rerank_text(doc: dict[str, Any], lang: str) -> str:
    """Build reranker passage text matched to query language."""
    if "event_summary" in doc and doc["event_summary"]:
        return doc["event_summary"]

    parts = []
    if lang == "ru":
        caption = doc.get("llm_caption_ru", "")
        if caption:
            parts.append(caption)
        else:
            parts.append(doc.get("llm_caption_en", "") or doc.get("summary", ""))
    else:
        caption = doc.get("llm_caption_en", "")
        if caption:
            parts.append(caption)
        else:
            parts.append(doc.get("llm_caption_ru", "") or doc.get("summary", ""))

    asr = doc.get("asr_text", "")
    if asr:
        parts.append(asr)

    return " ".join(parts).strip() or doc.get("summary", "") or doc.get("question", "")


def _iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    intersection = max(0.0, inter_end - inter_start)
    union = (end_a - start_a) + (end_b - start_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _rrf_merge(
    ranked_lists: list[list[dict[str, Any]]],
    k: int,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    items: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            uid = f"{item['video_id']}_{item['start']:.3f}_{item['end']:.3f}_{item.get('source', '')}"
            scores[uid] += 1.0 / (k + rank)
            if uid not in items:
                items[uid] = item.copy()

    merged: list[dict[str, Any]] = []
    for uid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = items[uid]
        entry["rrf_score"] = score
        merged.append(entry)

    return merged


def _dedup_by_overlap(
    results: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in results:
        dominated = False
        for existing in kept:
            if candidate["video_id"] != existing["video_id"]:
                continue
            if _iou(candidate["start"], candidate["end"],
                    existing["start"], existing["end"]) > iou_threshold:
                dominated = True
                break
        if not dominated:
            kept.append(candidate)
    return kept


def _sparse_dot_search(
    query_sparse: dict[str, float],
    doc_sparse_list: list[dict[str, float]],
    top_k: int,
) -> list[tuple[int, float]]:
    scores: list[tuple[int, float]] = []
    for i, doc_sparse in enumerate(doc_sparse_list):
        score = sum(query_sparse.get(k, 0.0) * v for k, v in doc_sparse.items())
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------

class Searcher:
    """Hybrid search with train→test matching + language-aware reranking."""

    def __init__(self) -> None:
        from FlagEmbedding import BGEM3FlagModel, FlagReranker

        print("[search] Loading BGE-M3 model ...")
        self.bge_model = BGEM3FlagModel(BGE_MODEL, use_fp16=True)

        print("[search] Loading reranker ...")
        # Use fine-tuned reranker if available, else pretrained
        reranker_path = str(RERANKER_FINETUNED) if RERANKER_FINETUNED.exists() else RERANKER_MODEL
        print(f"  Using: {reranker_path}")
        self.reranker = FlagReranker(reranker_path, use_fp16=True)

        # -- Translated queries ------------------------------------------------
        self._translated: dict[int, dict[str, str]] = {}
        self._load_translated()

        # -- FAISS indices (CPU) -----------------------------------------------
        print("[search] Loading FAISS indices ...")
        self.faiss_scenes = faiss.read_index(str(FAISS_SCENES_INDEX))
        self.faiss_events = faiss.read_index(str(FAISS_EVENTS_INDEX))

        # -- Train index (optional) --------------------------------------------
        self.faiss_train: faiss.Index | None = None
        self.train_meta: list[dict[str, Any]] = []
        self._load_train_index()

        # -- Metadata ----------------------------------------------------------
        print("[search] Loading metadata ...")
        with open(SCENES_META_FILE, "rb") as f:
            self.scenes_meta: list[dict[str, Any]] = pickle.load(f)
        with open(EVENTS_META_FILE, "rb") as f:
            self.events_meta: list[dict[str, Any]] = pickle.load(f)

        # -- Sparse vectors ----------------------------------------------------
        print("[search] Loading sparse vectors ...")
        with open(SPARSE_SCENES_FILE, "rb") as f:
            self.sparse_scenes: list[dict[str, float]] = pickle.load(f)
        with open(SPARSE_EVENTS_FILE, "rb") as f:
            self.sparse_events: list[dict[str, float]] = pickle.load(f)

        # -- Scene lookup for event -> scene resolution ------------------------
        self._scene_lookup: dict[str, dict[str, Any]] = {}
        self._load_scene_lookup()

        print("[search] Searcher ready.")

    # -- Load helpers -------------------------------------------------------

    def _load_translated(self) -> None:
        csv_path = TRANSLATED_CSV
        if not csv_path.exists():
            alt = Path(__file__).parent.parent.parent / "translated_data.csv"
            if alt.exists():
                csv_path = alt
            else:
                print(f"[search] WARNING: {TRANSLATED_CSV} not found, multi-query disabled")
                return

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            qid = int(row["query_id"])
            self._translated[qid] = {
                "corrected": str(row.get("corrected_question", row.get("question", ""))),
                "translated": str(row.get("translated_question", "")),
            }
        print(f"[search] Loaded {len(self._translated)} translated queries")

    def _load_train_index(self) -> None:
        if not TRAIN_INDEX_FILE.exists() or not TRAIN_META_FILE.exists():
            print("[search] Train index not found, train→test matching disabled")
            return
        self.faiss_train = faiss.read_index(str(TRAIN_INDEX_FILE))
        with open(TRAIN_META_FILE, "rb") as f:
            self.train_meta = pickle.load(f)
        print(f"[search] Loaded train index ({self.faiss_train.ntotal} queries)")

    def _load_scene_lookup(self) -> None:
        if not SCENES_FILE.exists():
            return
        with open(SCENES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                key = f"{doc['video_id']}__{doc['scene_idx']}"
                self._scene_lookup[key] = doc

    # -- Encode ------------------------------------------------------------

    def _encode_query(self, query: str) -> dict[str, Any]:
        output = self.bge_model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vec: np.ndarray = output["dense_vecs"][0]
        raw_sparse: dict[int, float] = output["lexical_weights"][0]
        sparse_vec: dict[str, float] = {str(k): float(v) for k, v in raw_sparse.items()}
        return {"dense": dense_vec, "sparse": sparse_vec}

    # -- Train→test matching -----------------------------------------------

    def _train_match(self, query_vec: np.ndarray) -> list[dict[str, Any]]:
        """Find similar train queries and return their ground-truth answers."""
        if self.faiss_train is None:
            return []

        query_vec_2d = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self.faiss_train.search(query_vec_2d, 10)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or float(score) < TRAIN_MATCH_LOW:
                continue
            meta = self.train_meta[idx]
            is_high = float(score) >= TRAIN_MATCH_HIGH
            results.append({
                "video_id": meta["video_id"],
                "start": meta["start"],
                "end": meta["end"],
                "train_score": float(score),
                "source": "train_high" if is_high else "train_low",
                "question": meta.get("question", ""),
            })
        return results

    # -- FAISS dense search ------------------------------------------------

    def _faiss_dense_search(
        self,
        index: faiss.Index,
        meta: list[dict[str, Any]],
        query_vec: np.ndarray,
        top_k: int,
        source_label: str,
    ) -> list[dict[str, Any]]:
        query_vec_2d = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = index.search(query_vec_2d, top_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = meta[idx]
            results.append({
                **doc,
                "faiss_score": float(score),
                "source": source_label,
            })
        return results

    # -- Sparse search -----------------------------------------------------

    def _sparse_search(
        self,
        query_sparse: dict[str, float],
        doc_sparse_list: list[dict[str, float]],
        meta: list[dict[str, Any]],
        top_k: int,
        source_label: str,
    ) -> list[dict[str, Any]]:
        ranked = _sparse_dot_search(query_sparse, doc_sparse_list, top_k)

        results: list[dict[str, Any]] = []
        for idx, score in ranked:
            if score <= 0:
                break
            doc = meta[idx]
            results.append({
                **doc,
                "sparse_score": float(score),
                "source": source_label,
            })
        return results

    # -- Event -> scene resolution -----------------------------------------

    def _resolve_event_to_scene(self, result: dict[str, Any]) -> dict[str, Any]:
        center_idx = result.get("center_scene_idx")
        if center_idx is None:
            return result

        key = f"{result['video_id']}__{center_idx}"
        scene = self._scene_lookup.get(key)
        if scene:
            result = result.copy()
            result["start"] = scene["start"]
            result["end"] = scene["end"]
        return result

    # -- Main search -------------------------------------------------------

    def search(
        self,
        query: str,
        top_n: int = FINAL_TOP_N,
        corrected_query: str | None = None,
        translated_query: str | None = None,
    ) -> list[dict[str, Any]]:
        q_main = corrected_query or query
        q_en = translated_query or ""

        queries_to_search = [q_main]
        if q_en and q_en.strip().lower() != q_main.strip().lower():
            queries_to_search.append(q_en)

        encodings = [self._encode_query(q) for q in queries_to_search]

        ranked_lists: list[list[dict[str, Any]]] = []

        # Train→test matching: use main query encoding
        train_matches = self._train_match(encodings[0]["dense"])
        if train_matches:
            # High-confidence matches get injected as a separate ranked list
            # with artificially high rank to boost through RRF
            high = [m for m in train_matches if m["source"] == "train_high"]
            low = [m for m in train_matches if m["source"] == "train_low"]
            if high:
                ranked_lists.append(high)
            if low:
                ranked_lists.append(low)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {}
            for qi, enc in enumerate(encodings):
                q_tag = "main" if qi == 0 else "en"
                futures[pool.submit(
                    self._faiss_dense_search,
                    self.faiss_scenes, self.scenes_meta,
                    enc["dense"], SEARCH_TOP_K_DENSE, f"dense_scenes_{q_tag}",
                )] = f"dense_scenes_{q_tag}"

                futures[pool.submit(
                    self._faiss_dense_search,
                    self.faiss_events, self.events_meta,
                    enc["dense"], SEARCH_TOP_K_DENSE, f"dense_events_{q_tag}",
                )] = f"dense_events_{q_tag}"

                futures[pool.submit(
                    self._sparse_search,
                    enc["sparse"], self.sparse_scenes, self.scenes_meta,
                    SEARCH_TOP_K_SPARSE, f"sparse_scenes_{q_tag}",
                )] = f"sparse_scenes_{q_tag}"

                futures[pool.submit(
                    self._sparse_search,
                    enc["sparse"], self.sparse_events, self.events_meta,
                    SEARCH_TOP_K_SPARSE, f"sparse_events_{q_tag}",
                )] = f"sparse_events_{q_tag}"

            for future in as_completed(futures):
                ranked_lists.append(future.result())

        # RRF merge
        merged = _rrf_merge(ranked_lists, k=RRF_K)

        # Dedup
        deduped = _dedup_by_overlap(merged)

        # Reranker
        candidates = deduped[:RERANKER_TOP_K]

        if candidates:
            query_lang = "ru" if _is_russian(q_main) else "en"

            rerank_texts = [_get_rerank_text(c, query_lang) for c in candidates]
            pairs_main = [[q_main, t] for t in rerank_texts]
            scores_main = self.reranker.compute_score(pairs_main)
            if isinstance(scores_main, (int, float)):
                scores_main = [scores_main]

            if q_en and q_en.strip().lower() != q_main.strip().lower():
                rerank_texts_en = [_get_rerank_text(c, "en") for c in candidates]
                pairs_en = [[q_en, t] for t in rerank_texts_en]
                scores_en = self.reranker.compute_score(pairs_en)
                if isinstance(scores_en, (int, float)):
                    scores_en = [scores_en]
                final_scores = [max(s1, s2) for s1, s2 in zip(scores_main, scores_en)]
            else:
                final_scores = list(scores_main) if not isinstance(scores_main, list) else scores_main

            # Boost train_high matches in reranker score
            for cand, score in zip(candidates, final_scores):
                if cand.get("source") == "train_high":
                    score = max(score, score + 5.0)  # significant boost
                cand["reranker_score"] = float(score)

            candidates.sort(key=lambda x: x["reranker_score"], reverse=True)
            candidates = candidates[:RERANKER_OUTPUT_K]

        # Event -> scene resolution
        resolved = [self._resolve_event_to_scene(c) for c in candidates]

        # Final dedup and trim
        resolved = _dedup_by_overlap(resolved)
        final = resolved[:top_n]

        return [
            {
                "video_file": r["video_id"],
                "start": round(r["start"], 3),
                "end": round(r["end"], 3),
                "score": r.get("reranker_score", r.get("rrf_score", 0.0)),
            }
            for r in final
        ]

    # -- Submission generation ---------------------------------------------

    def generate_submission(
        self,
        test_csv: Path = TEST_CSV,
        output_csv: Path = WORK_DIR / "submission.csv",
    ) -> Path:
        test_df = pd.read_csv(test_csv)
        print(f"[search] Processing {len(test_df)} queries from {test_csv}")

        rows: list[dict[str, Any]] = []

        for i, (_, row) in tqdm(enumerate(test_df.iterrows()), total=len(test_df), desc="[search]"):
            qid = int(row["query_id"])
            query_text = str(row["question"])

            translated = self._translated.get(qid, {})
            corrected = translated.get("corrected", query_text)
            translated_en = translated.get("translated", "")

            results = self.search(
                query=query_text,
                top_n=FINAL_TOP_N,
                corrected_query=corrected,
                translated_query=translated_en,
            )

            out: dict[str, Any] = {"query_id": qid}
            for rank in range(1, FINAL_TOP_N + 1):
                if rank <= len(results):
                    r = results[rank - 1]
                    out[f"video_file_{rank}"] = r["video_file"]
                    out[f"start_{rank}"] = r["start"]
                    out[f"end_{rank}"] = r["end"]
                else:
                    out[f"video_file_{rank}"] = ""
                    out[f"start_{rank}"] = 0.0
                    out[f"end_{rank}"] = 0.0
            rows.append(out)

        cols = ["query_id"]
        for rank in range(1, FINAL_TOP_N + 1):
            cols += [f"video_file_{rank}", f"start_{rank}", f"end_{rank}"]

        submission_df = pd.DataFrame(rows)[cols]
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        submission_df.to_csv(output_csv, index=False)

        print(f"[search] Saved {len(submission_df)} rows to {output_csv}")
        return output_csv


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    searcher = Searcher()
    output = searcher.generate_submission()
    print(f"[search] Done. Submission at {output}")


if __name__ == "__main__":
    main()
