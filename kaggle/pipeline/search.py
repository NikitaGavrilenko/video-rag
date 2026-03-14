"""
search.py — Online search / inference for Video RAG submission generation.

Uses FAISS-GPU for dense retrieval, in-memory sparse vector matching,
and BM25 (rank_bm25) instead of Elasticsearch.

Flow:
  1. Query -> BGE-M3 encode (dense + sparse)
  2. Parallel search across 6 channels:
     a. FAISS scenes dense
     b. FAISS events dense
     c. BM25 scenes (asr_text)
     d. BM25 events (event_summary)
     e. Sparse vector matching on scenes
     f. Sparse vector matching on events
  3. RRF merge across all channels
  4. Dedup by (video_id, overlapping timecodes) — IoU > 50 %
  5. BGE reranker: rerank top RERANKER_TOP_K -> keep RERANKER_OUTPUT_K
  6. Event -> scene resolution (center_scene_idx for exact timecodes)
  7. Return top FINAL_TOP_N results
"""

from __future__ import annotations

import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from .config import (
    BGE_MODEL,
    BM25_EVENTS_FILE,
    BM25_SCENES_FILE,
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
    WORK_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Compute Intersection-over-Union for two time intervals."""
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
    """Reciprocal Rank Fusion across multiple ranked result lists.

    Each result dict must contain at least: video_id, start, end, source.
    Returns merged list sorted by RRF score (descending).
    """
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
    """Remove duplicates where same video_id has overlapping timecodes (IoU > threshold).

    Keeps the higher-scored result.
    """
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
    """Compute dot product between query sparse vector and each document sparse vector.

    Returns list of (doc_index, score) sorted by score descending.
    """
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
    """Hybrid search engine combining FAISS-GPU dense, BM25, and sparse retrieval with reranking."""

    def __init__(self) -> None:
        from FlagEmbedding import BGEM3FlagModel, FlagReranker

        print("[search] Loading BGE-M3 model ...")
        self.bge_model = BGEM3FlagModel(BGE_MODEL, use_fp16=True)

        print("[search] Loading reranker ...")
        self.reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)

        # -- FAISS GPU indices ------------------------------------------------
        print("[search] Loading FAISS indices to GPU ...")
        res = faiss.StandardGpuResources()
        self.faiss_scenes = faiss.index_cpu_to_gpu(
            res, 0, faiss.read_index(str(FAISS_SCENES_INDEX)),
        )
        self.faiss_events = faiss.index_cpu_to_gpu(
            res, 0, faiss.read_index(str(FAISS_EVENTS_INDEX)),
        )

        # -- Metadata ---------------------------------------------------------
        print("[search] Loading metadata ...")
        with open(SCENES_META_FILE, "rb") as f:
            self.scenes_meta: list[dict[str, Any]] = pickle.load(f)
        with open(EVENTS_META_FILE, "rb") as f:
            self.events_meta: list[dict[str, Any]] = pickle.load(f)

        # -- Sparse vectors ---------------------------------------------------
        print("[search] Loading sparse vectors ...")
        with open(SPARSE_SCENES_FILE, "rb") as f:
            self.sparse_scenes: list[dict[str, float]] = pickle.load(f)
        with open(SPARSE_EVENTS_FILE, "rb") as f:
            self.sparse_events: list[dict[str, float]] = pickle.load(f)

        # -- BM25 indices -----------------------------------------------------
        print("[search] Loading BM25 indices ...")
        with open(BM25_SCENES_FILE, "rb") as f:
            self.bm25_scenes = pickle.load(f)  # BM25Okapi
        with open(BM25_EVENTS_FILE, "rb") as f:
            self.bm25_events = pickle.load(f)  # BM25Okapi

        # -- Scene lookup for event -> scene resolution -----------------------
        self._scene_lookup: dict[str, dict[str, Any]] = {}
        self._load_scene_lookup()

        print("[search] Searcher ready.")

    # -- Scene lookup -------------------------------------------------------

    def _load_scene_lookup(self) -> None:
        """Build lookup: (video_id, scene_idx) -> scene doc."""
        if not SCENES_FILE.exists():
            print(f"[search] WARNING: {SCENES_FILE} not found, event resolution disabled")
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
        """Encode query via BGE-M3 into dense and sparse representations."""
        output = self.bge_model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vec: np.ndarray = output["dense_vecs"][0]

        # Sparse: BGE-M3 returns dict of {token_id: weight}
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
        """Search a FAISS-GPU index and map results back to metadata."""
        query_vec_2d = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = index.search(query_vec_2d, top_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = meta[idx]
            results.append({
                "video_id": doc["video_id"],
                "start": doc.get("start", 0.0),
                "end": doc.get("end", 0.0),
                "faiss_score": float(score),
                "source": source_label,
                "center_scene_idx": doc.get("center_scene_idx"),
                "scene_idx": doc.get("scene_idx"),
                "text": doc.get("asr_text", "") or doc.get("summary", "") or doc.get("event_summary", ""),
            })
        return results

    # -- BM25 search -------------------------------------------------------

    def _bm25_search(
        self,
        bm25,
        meta: list[dict[str, Any]],
        query: str,
        top_k: int,
        source_label: str,
    ) -> list[dict[str, Any]]:
        """Search with a BM25Okapi index and map results back to metadata."""
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            doc = meta[idx]
            results.append({
                "video_id": doc["video_id"],
                "start": doc.get("start", 0.0),
                "end": doc.get("end", 0.0),
                "bm25_score": float(scores[idx]),
                "source": source_label,
                "center_scene_idx": doc.get("center_scene_idx"),
                "scene_idx": doc.get("scene_idx"),
                "text": doc.get("asr_text", "") or doc.get("summary", "") or doc.get("event_summary", ""),
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
        """In-memory sparse dot-product search."""
        ranked = _sparse_dot_search(query_sparse, doc_sparse_list, top_k)

        results: list[dict[str, Any]] = []
        for idx, score in ranked:
            if score <= 0:
                break
            doc = meta[idx]
            results.append({
                "video_id": doc["video_id"],
                "start": doc.get("start", 0.0),
                "end": doc.get("end", 0.0),
                "sparse_score": float(score),
                "source": source_label,
                "center_scene_idx": doc.get("center_scene_idx"),
                "scene_idx": doc.get("scene_idx"),
                "text": doc.get("asr_text", "") or doc.get("summary", "") or doc.get("event_summary", ""),
            })
        return results

    # -- Event -> scene resolution -----------------------------------------

    def _resolve_event_to_scene(self, result: dict[str, Any]) -> dict[str, Any]:
        """For event results, resolve center_scene_idx to exact scene timecodes."""
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

    def search(self, query: str, top_n: int = FINAL_TOP_N) -> list[dict[str, Any]]:
        """Run hybrid search and return top_n results.

        Returns list of {video_file, start, end, score}.
        """
        # 1. Encode query
        encoded = self._encode_query(query)
        dense_vec = encoded["dense"]
        sparse_vec = encoded["sparse"]

        # 2. Parallel search across 6 channels
        ranked_lists: list[list[dict[str, Any]]] = []

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(
                    self._faiss_dense_search,
                    self.faiss_scenes, self.scenes_meta,
                    dense_vec, SEARCH_TOP_K_DENSE, "dense_scenes",
                ): "scenes_dense",
                pool.submit(
                    self._faiss_dense_search,
                    self.faiss_events, self.events_meta,
                    dense_vec, SEARCH_TOP_K_DENSE, "dense_events",
                ): "events_dense",
                pool.submit(
                    self._bm25_search,
                    self.bm25_scenes, self.scenes_meta,
                    query, SEARCH_TOP_K_SPARSE, "bm25_scenes",
                ): "scenes_bm25",
                pool.submit(
                    self._bm25_search,
                    self.bm25_events, self.events_meta,
                    query, SEARCH_TOP_K_SPARSE, "bm25_events",
                ): "events_bm25",
                pool.submit(
                    self._sparse_search,
                    sparse_vec, self.sparse_scenes, self.scenes_meta,
                    SEARCH_TOP_K_SPARSE, "sparse_scenes",
                ): "scenes_sparse",
                pool.submit(
                    self._sparse_search,
                    sparse_vec, self.sparse_events, self.events_meta,
                    SEARCH_TOP_K_SPARSE, "sparse_events",
                ): "events_sparse",
            }
            for future in as_completed(futures):
                ranked_lists.append(future.result())

        # 3. RRF merge
        merged = _rrf_merge(ranked_lists, k=RRF_K)

        # 4. Dedup by overlapping timecodes
        deduped = _dedup_by_overlap(merged)

        # 5. Reranker: take top RERANKER_TOP_K candidates, rerank
        candidates = deduped[:RERANKER_TOP_K]

        if candidates:
            pairs = [[query, c.get("text", "")] for c in candidates]
            reranker_scores = self.reranker.compute_score(pairs)
            # compute_score returns a single float if only one pair
            if isinstance(reranker_scores, (int, float)):
                reranker_scores = [reranker_scores]

            for cand, score in zip(candidates, reranker_scores):
                cand["reranker_score"] = float(score)

            candidates.sort(key=lambda x: x["reranker_score"], reverse=True)
            candidates = candidates[:RERANKER_OUTPUT_K]

        # 6. Event -> scene resolution
        resolved = [self._resolve_event_to_scene(c) for c in candidates]

        # 7. Final dedup after resolution and trim to top_n
        resolved = _dedup_by_overlap(resolved)
        final = resolved[:top_n]

        # Format output
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
        """Read test queries, run search for each, and write submission.csv."""
        test_df = pd.read_csv(test_csv)
        print(f"[search] Processing {len(test_df)} queries from {test_csv}")

        rows: list[dict[str, Any]] = []

        for _, row in test_df.iterrows():
            qid = row["query_id"]
            query_text = row["question"]
            results = self.search(query_text, top_n=FINAL_TOP_N)

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

        # Build DataFrame with correct column order
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
    """Run search and generate submission.csv."""
    searcher = Searcher()
    output = searcher.generate_submission()
    print(f"[search] Done. Submission at {output}")


if __name__ == "__main__":
    main()
