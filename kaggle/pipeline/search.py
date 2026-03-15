"""
search.py — Hybrid search with multi-query (corrected + translated).

Flow:
  1. Load translated_data.csv (corrected_question + translated_question)
  2. For each query: encode BOTH corrected (RU) and translated (EN) versions
  3. Parallel search across channels:
     - FAISS dense scenes × 2 queries
     - FAISS dense events × 2 queries
     - Sparse scenes × 2 queries
     - Sparse events × 2 queries
  4. RRF merge across all channels
  5. Dedup by (video_id, overlapping timecodes) — IoU > 50%
  6. BGE reranker — language-aware: RU query vs RU caption, EN query vs EN caption
  7. Event -> scene resolution
  8. Return top FINAL_TOP_N results
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

from .config import (
    BGE_MODEL,
    EVENTS_META_FILE,
    FAISS_EVENTS_INDEX,
    FAISS_SCENES_INDEX,
    FINAL_TOP_N,
    RERANKER_MODEL,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RU_RE = re.compile(r"[а-яА-ЯёЁ]")


def _is_russian(text: str) -> bool:
    """Heuristic: contains Cyrillic characters."""
    return bool(_RU_RE.search(text))


def _get_rerank_text(doc: dict[str, Any], lang: str) -> str:
    """Build reranker passage text matched to query language.

    For scenes: use llm_caption in matching language + ASR.
    For events: use event_summary (already bilingual).
    """
    # Event docs
    if "event_summary" in doc and doc["event_summary"]:
        return doc["event_summary"]

    # Scene docs — prefer language-matched LLM caption
    parts = []
    if lang == "ru":
        caption = doc.get("llm_caption_ru", "")
        if caption:
            parts.append(caption)
        else:
            # fallback to EN or raw summary
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

    return " ".join(parts).strip() or doc.get("summary", "")


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
    """Reciprocal Rank Fusion across multiple ranked result lists."""
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
    """Hybrid search engine: multi-query dense + sparse, language-aware reranking."""

    def __init__(self) -> None:
        from FlagEmbedding import BGEM3FlagModel, FlagReranker

        print("[search] Loading BGE-M3 model ...")
        self.bge_model = BGEM3FlagModel(BGE_MODEL, use_fp16=True)

        print("[search] Loading reranker ...")
        self.reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)

        # -- Translated queries (pre-corrected + English) ----------------------
        self._translated: dict[int, dict[str, str]] = {}
        self._load_translated()

        # -- FAISS indices (CPU) -----------------------------------------------
        print("[search] Loading FAISS indices ...")
        self.faiss_scenes = faiss.read_index(str(FAISS_SCENES_INDEX))
        self.faiss_events = faiss.read_index(str(FAISS_EVENTS_INDEX))

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

    # -- Translated queries -------------------------------------------------

    def _load_translated(self) -> None:
        """Load translated_data.csv for corrected + English translations."""
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

    # -- Scene lookup -------------------------------------------------------

    def _load_scene_lookup(self) -> None:
        if not SCENES_FILE.exists():
            print(f"[search] WARNING: {SCENES_FILE} not found")
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

    # -- Sparse vector search on doc list ----------------------------------

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
        """Run multi-query hybrid search with language-aware reranking."""
        q_main = corrected_query or query
        q_en = translated_query or ""

        # Determine unique queries to encode
        queries_to_search = [q_main]
        if q_en and q_en.strip().lower() != q_main.strip().lower():
            queries_to_search.append(q_en)

        # Encode all queries
        encodings = [self._encode_query(q) for q in queries_to_search]

        # Search across all channels × all queries
        ranked_lists: list[list[dict[str, Any]]] = []

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

        # Dedup by overlapping timecodes
        deduped = _dedup_by_overlap(merged)

        # Reranker: language-aware — RU query vs RU caption, EN query vs EN caption
        candidates = deduped[:RERANKER_TOP_K]

        if candidates:
            # Detect query language for reranker passage matching
            query_lang = "ru" if _is_russian(q_main) else "en"

            # Double reranking: score against both RU and EN queries
            # Take max score to handle bilingual content
            rerank_texts = [_get_rerank_text(c, query_lang) for c in candidates]

            # Primary: rerank with corrected (original language) query
            pairs_main = [[q_main, t] for t in rerank_texts]
            scores_main = self.reranker.compute_score(pairs_main)
            if isinstance(scores_main, (int, float)):
                scores_main = [scores_main]

            # Secondary: if EN translation available, also rerank with it
            if q_en and q_en.strip().lower() != q_main.strip().lower():
                rerank_texts_en = [_get_rerank_text(c, "en") for c in candidates]
                pairs_en = [[q_en, t] for t in rerank_texts_en]
                scores_en = self.reranker.compute_score(pairs_en)
                if isinstance(scores_en, (int, float)):
                    scores_en = [scores_en]

                # Take max of both language scores
                final_scores = [max(s1, s2) for s1, s2 in zip(scores_main, scores_en)]
            else:
                final_scores = scores_main

            for cand, score in zip(candidates, final_scores):
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
        """Read test queries, run multi-query search, write submission.csv."""
        test_df = pd.read_csv(test_csv)
        print(f"[search] Processing {len(test_df)} queries from {test_csv}")

        rows: list[dict[str, Any]] = []

        for _, row in test_df.iterrows():
            qid = int(row["query_id"])
            query_text = str(row["question"])

            # Use translated data if available
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
