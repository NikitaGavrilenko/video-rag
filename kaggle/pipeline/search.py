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
TRAIN_MATCH_HIGH = 0.92   # direct answer — inject at top with reranker boost
TRAIN_MATCH_LOW = 0.80    # candidate — add to reranker pool


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

    def _encode_queries_batch(self, queries: list[str], batch_size: int = 256) -> list[dict[str, Any]]:
        """Batch encode all queries at once — much faster than one-by-one."""
        output = self.bge_model.encode(
            queries,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        results = []
        for i in range(len(queries)):
            dense_vec = output["dense_vecs"][i]
            raw_sparse = output["lexical_weights"][i]
            sparse_vec = {str(k): float(v) for k, v in raw_sparse.items()}
            results.append({"dense": dense_vec, "sparse": sparse_vec})
        return results

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

    # -- Timecode expansion ±55s, capped at 180s -----------------------------

    EXPAND_SEC = 55.0       # ±55s around center → 110s window
    MAX_DURATION = 180.0    # hard cap — never exceed 180s

    def _expand_timecodes(self, result: dict[str, Any]) -> dict[str, Any]:
        """Expand short segments to ±55s (110s window). Cap at 180s max."""
        if result.get("source", "").startswith("train"):
            return result

        result = result.copy()
        start = result["start"]
        end = result["end"]
        duration = end - start

        if duration >= self.MAX_DURATION:
            # Already too long — trim to MAX_DURATION centered
            center = (start + end) / 2.0
            result["start"] = max(0.0, center - self.MAX_DURATION / 2.0)
            result["end"] = center + self.MAX_DURATION / 2.0
            return result

        # Expand ±55s from center
        center = (start + end) / 2.0
        new_start = max(0.0, center - self.EXPAND_SEC)
        new_end = center + self.EXPAND_SEC

        # Cap at MAX_DURATION
        if (new_end - new_start) > self.MAX_DURATION:
            new_start = max(0.0, center - self.MAX_DURATION / 2.0)
            new_end = center + self.MAX_DURATION / 2.0

        result["start"] = new_start
        result["end"] = new_end
        return result

    # -- Video-level clustering -----------------------------------------------

    def _video_cluster(self, results: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        """Prefer multiple fragments from top-scoring videos.

        Strategy: rank videos by sum of reranker scores.
        Take top-2 videos, fill top_n slots from them.
        If not enough, fill remaining from other videos.
        """
        if len(results) <= top_n:
            return results

        # Score each video by sum of its candidate scores
        video_scores: dict[str, float] = defaultdict(float)
        video_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in results:
            vid = r["video_id"]
            video_scores[vid] += r.get("reranker_score", r.get("rrf_score", 0.0))
            video_results[vid].append(r)

        # Rank videos by aggregate score
        ranked_videos = sorted(video_scores.keys(), key=lambda v: video_scores[v], reverse=True)

        # Fill from top-2 videos first, then others
        final: list[dict[str, Any]] = []
        top_videos = ranked_videos[:2]

        for vid in top_videos:
            for r in video_results[vid]:
                if len(final) >= top_n:
                    break
                final.append(r)

        # Fill remaining from other videos
        if len(final) < top_n:
            for vid in ranked_videos[2:]:
                for r in video_results[vid]:
                    if len(final) >= top_n:
                        break
                    final.append(r)
                if len(final) >= top_n:
                    break

        return final

    # -- Retrieval (no reranker) --------------------------------------------

    def _retrieve(
        self,
        query: str,
        corrected_query: str | None = None,
        translated_query: str | None = None,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """Run retrieval channels + RRF + dedup. Returns (q_main, q_en, candidates)."""
        q_main = corrected_query or query
        q_en = translated_query or ""

        queries_to_search = [q_main]
        if q_en and q_en.strip().lower() != q_main.strip().lower():
            queries_to_search.append(q_en)

        encodings = [self._encode_query(q) for q in queries_to_search]

        ranked_lists: list[list[dict[str, Any]]] = []

        # Train→test matching
        train_matches = self._train_match(encodings[0]["dense"])
        if train_matches:
            high = [m for m in train_matches if m["source"] == "train_high"]
            low = [m for m in train_matches if m["source"] == "train_low"]
            if high:
                ranked_lists.append(high)
            if low:
                ranked_lists.append(low)

        for qi, enc in enumerate(encodings):
            q_tag = "main" if qi == 0 else "en"
            ranked_lists.append(self._faiss_dense_search(
                self.faiss_scenes, self.scenes_meta,
                enc["dense"], SEARCH_TOP_K_DENSE, f"dense_scenes_{q_tag}",
            ))
            ranked_lists.append(self._faiss_dense_search(
                self.faiss_events, self.events_meta,
                enc["dense"], SEARCH_TOP_K_DENSE, f"dense_events_{q_tag}",
            ))
            ranked_lists.append(self._sparse_search(
                enc["sparse"], self.sparse_scenes, self.scenes_meta,
                SEARCH_TOP_K_SPARSE, f"sparse_scenes_{q_tag}",
            ))
            ranked_lists.append(self._sparse_search(
                enc["sparse"], self.sparse_events, self.events_meta,
                SEARCH_TOP_K_SPARSE, f"sparse_events_{q_tag}",
            ))

        merged = _rrf_merge(ranked_lists, k=RRF_K)
        deduped = _dedup_by_overlap(merged, iou_threshold=0.3)
        candidates = deduped[:RERANKER_TOP_K]

        return q_main, q_en, candidates

    # -- Batch reranking ---------------------------------------------------

    def _batch_rerank(
        self,
        query_candidates: list[tuple[str, str, list[dict[str, Any]]]],
    ) -> list[list[dict[str, Any]]]:
        """Rerank candidates for multiple queries in one big GPU batch.

        Input: [(q_main, q_en, candidates), ...]
        Output: [reranked_candidates, ...] per query
        """
        # Build flat pair lists with index tracking
        all_pairs_main: list[list[str]] = []
        all_pairs_en: list[list[str]] = []
        index_map: list[tuple[int, int]] = []  # (query_idx, candidate_idx)
        en_index_map: list[tuple[int, int]] = []  # same but only for EN pairs

        for qi, (q_main, q_en, candidates) in enumerate(query_candidates):
            query_lang = "ru" if _is_russian(q_main) else "en"
            use_en = bool(q_en and q_en.strip().lower() != q_main.strip().lower())

            for ci, cand in enumerate(candidates):
                text_main = _get_rerank_text(cand, query_lang)
                all_pairs_main.append([q_main, text_main])
                index_map.append((qi, ci))
                if use_en:
                    text_en = _get_rerank_text(cand, "en")
                    all_pairs_en.append([q_en, text_en])
                    en_index_map.append((qi, ci))

        if not all_pairs_main:
            return [[] for _ in query_candidates]

        # One big reranker call for all pairs
        scores_main = self.reranker.compute_score(all_pairs_main)
        if isinstance(scores_main, (int, float)):
            scores_main = [scores_main]

        # Build EN scores lookup: (qi, ci) -> score
        en_scores: dict[tuple[int, int], float] = {}
        if all_pairs_en:
            scores_en_raw = self.reranker.compute_score(all_pairs_en)
            if isinstance(scores_en_raw, (int, float)):
                scores_en_raw = [scores_en_raw]
            for (qi, ci), s in zip(en_index_map, scores_en_raw):
                en_scores[(qi, ci)] = float(s)

        # Distribute scores back to queries
        for flat_idx, (qi, ci) in enumerate(index_map):
            score = float(scores_main[flat_idx])
            en_score = en_scores.get((qi, ci))
            if en_score is not None:
                score = max(score, en_score)
            cand = query_candidates[qi][2][ci]
            if cand.get("source") == "train_high":
                score = score + 2.0
            cand["reranker_score"] = score

        # Sort, expand, dedup, cluster per query
        results: list[list[dict[str, Any]]] = []
        for qi, (q_main, q_en, candidates) in enumerate(query_candidates):
            candidates.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)
            top = candidates[:RERANKER_OUTPUT_K]
            expanded = [self._expand_timecodes(c) for c in top]
            expanded = _dedup_by_overlap(expanded, iou_threshold=0.3)
            # Video-level clustering: prefer multiple fragments from top videos
            final = self._video_cluster(expanded, FINAL_TOP_N)
            results.append(final)

        return results

    # -- Submission generation ---------------------------------------------

    def generate_submission(
        self,
        test_csv: Path = TEST_CSV,
        output_csv: Path = WORK_DIR / "submission.csv",
        rerank_batch_size: int = 32,
    ) -> Path:
        """Two-phase submission: batch retrieval, then batch reranking."""
        test_df = pd.read_csv(test_csv)
        print(f"[search] Processing {len(test_df)} queries from {test_csv}")

        # Collect all queries
        query_data: list[tuple[int, str, str, str]] = []  # (qid, q_main, q_en, raw)
        all_texts_to_encode: list[str] = []
        text_index_map: list[tuple[int, str]] = []  # (query_idx, "main"/"en")

        for _, row in test_df.iterrows():
            qid = int(row["query_id"])
            query_text = str(row["question"])
            translated = self._translated.get(qid, {})
            corrected = translated.get("corrected", query_text)
            translated_en = translated.get("translated", "")

            q_main = corrected or query_text
            q_en = translated_en or ""
            qi = len(query_data)
            query_data.append((qid, q_main, q_en, query_text))

            all_texts_to_encode.append(q_main)
            text_index_map.append((qi, "main"))
            if q_en and q_en.strip().lower() != q_main.strip().lower():
                all_texts_to_encode.append(q_en)
                text_index_map.append((qi, "en"))

        # Phase 1a: Batch encode all queries on GPU
        print(f"[search] Phase 1a: Batch encoding {len(all_texts_to_encode)} query texts...")
        all_encodings = self._encode_queries_batch(all_texts_to_encode)

        # Map encodings back to queries
        query_encodings: list[list[dict[str, Any]]] = [[] for _ in query_data]
        for enc, (qi, tag) in zip(all_encodings, text_index_map):
            query_encodings[qi].append(enc)

        # Phase 1b: Retrieval (FAISS + sparse + RRF) — with cache
        retrieval_cache = WORK_DIR / "retrieval_cache.pkl"
        retrieval_results: list[tuple[str, str, list[dict[str, Any]]]] = []

        if retrieval_cache.exists():
            print(f"[search] Loading retrieval cache from {retrieval_cache}...")
            with open(retrieval_cache, "rb") as f:
                retrieval_results = pickle.load(f)
            print(f"  Loaded {len(retrieval_results)} cached results")
        else:
            print("[search] Phase 1b: Retrieval...")
            train_high_count = 0
            train_low_count = 0
            for qi, (qid, q_main, q_en, _) in tqdm(enumerate(query_data), total=len(query_data), desc="[retrieval]"):
                encodings = query_encodings[qi]
                ranked_lists: list[list[dict[str, Any]]] = []

                # Train→test matching
                train_matches = self._train_match(encodings[0]["dense"])
                if train_matches:
                    high = [m for m in train_matches if m["source"] == "train_high"]
                    low = [m for m in train_matches if m["source"] == "train_low"]
                    if high:
                        ranked_lists.append(high)
                        train_high_count += 1
                    if low:
                        ranked_lists.append(low)
                        train_low_count += 1

                for ei, enc in enumerate(encodings):
                    q_tag = "main" if ei == 0 else "en"
                    ranked_lists.append(self._faiss_dense_search(
                        self.faiss_scenes, self.scenes_meta,
                        enc["dense"], SEARCH_TOP_K_DENSE, f"dense_scenes_{q_tag}",
                    ))
                    ranked_lists.append(self._faiss_dense_search(
                        self.faiss_events, self.events_meta,
                        enc["dense"], SEARCH_TOP_K_DENSE, f"dense_events_{q_tag}",
                    ))
                    ranked_lists.append(self._sparse_search(
                        enc["sparse"], self.sparse_scenes, self.scenes_meta,
                        SEARCH_TOP_K_SPARSE, f"sparse_scenes_{q_tag}",
                    ))
                    ranked_lists.append(self._sparse_search(
                        enc["sparse"], self.sparse_events, self.events_meta,
                        SEARCH_TOP_K_SPARSE, f"sparse_events_{q_tag}",
                    ))

                merged = _rrf_merge(ranked_lists, k=RRF_K)
                deduped = _dedup_by_overlap(merged, iou_threshold=0.3)
                candidates = deduped[:RERANKER_TOP_K]
                retrieval_results.append((q_main, q_en, candidates))

            print(f"  Train matches: {train_high_count} high (>{TRAIN_MATCH_HIGH}), {train_low_count} low (>{TRAIN_MATCH_LOW})")

            # Save cache
            with open(retrieval_cache, "wb") as f:
                pickle.dump(retrieval_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Saved retrieval cache to {retrieval_cache}")

        # Phase 2: Batch reranking — GPU heavy, batched for throughput
        print(f"[search] Phase 2: Batch reranking ({len(retrieval_results)} queries)...")
        all_reranked: list[list[dict[str, Any]]] = []

        for batch_start in tqdm(range(0, len(retrieval_results), rerank_batch_size),
                                desc="[rerank]"):
            batch = retrieval_results[batch_start : batch_start + rerank_batch_size]
            reranked = self._batch_rerank(batch)
            all_reranked.extend(reranked)

        # Build submission — fill missing slots with last valid result
        rows: list[dict[str, Any]] = []
        for (qid, _, _, _), results in zip(query_data, all_reranked):
            out: dict[str, Any] = {"query_id": qid}
            for rank in range(1, FINAL_TOP_N + 1):
                if rank <= len(results):
                    r = results[rank - 1]
                    out[f"video_file_{rank}"] = r["video_id"]
                    out[f"start_{rank}"] = round(r["start"], 3)
                    out[f"end_{rank}"] = round(r["end"], 3)
                else:
                    # Duplicate last valid result to avoid nulls
                    last = results[-1] if results else {"video_id": "video_00000000", "start": 0.0, "end": 30.0}
                    out[f"video_file_{rank}"] = last["video_id"]
                    out[f"start_{rank}"] = round(last["start"], 3)
                    out[f"end_{rank}"] = round(last["end"], 3)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="Delete retrieval cache and recompute")
    args, _ = parser.parse_known_args()

    if args.no_cache:
        cache = WORK_DIR / "retrieval_cache.pkl"
        if cache.exists():
            cache.unlink()
            print("[search] Deleted retrieval cache")

    searcher = Searcher()
    output = searcher.generate_submission()
    print(f"[search] Done. Submission at {output}")


if __name__ == "__main__":
    main()
