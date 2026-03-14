"""
demo/api.py — FastAPI сервер для демо-стенда.

Загружает индексы, собранные kaggle/pipeline/step6_index.py, и отвечает
на текстовые поисковые запросы через гибридный поиск + BGE reranker.

Эндпоинты:
  POST /search          — текстовый поиск, возвращает топ-5 сцен
  GET  /keyframes/{video_id}/{scene_idx}  — отдаёт JPEG кадр
  GET  /health          — статус

Запуск:
  cd demo && uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from pydantic import BaseModel

# ── sys.path: чтобы импортировать preproc из корня репо ──────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from preproc.query_preprocessor import QueryPreprocessor

# ── Пути к данным (скачанным с сервера) ──────────────────────────────────────
DATA_DIR      = ROOT / "data"
INDEXES_DIR   = DATA_DIR / "indexes"
KEYFRAMES_DIR = DATA_DIR / "keyframes"

FAISS_SCENES  = INDEXES_DIR / "faiss_scenes.index"
FAISS_EVENTS  = INDEXES_DIR / "faiss_events.index"
SCENES_META   = INDEXES_DIR / "scenes_meta.pkl"
EVENTS_META   = INDEXES_DIR / "events_meta.pkl"
SPARSE_SCENES = INDEXES_DIR / "sparse_scenes.pkl"
SPARSE_EVENTS = INDEXES_DIR / "sparse_events.pkl"
BM25_SCENES   = INDEXES_DIR / "bm25_scenes.pkl"
BM25_EVENTS   = INDEXES_DIR / "bm25_events.pkl"
SCENES_JSONL  = DATA_DIR / "scenes.jsonl"

# ── Параметры поиска ──────────────────────────────────────────────────────────
TOP_K_DENSE    = 50
TOP_K_SPARSE   = 50
RRF_K          = 60
RERANKER_TOP_K = 100
FINAL_TOP_N    = 5

BGE_MODEL      = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _iou(s1: float, e1: float, s2: float, e2: float) -> float:
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def _rrf_merge(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    scores: dict[str, float] = defaultdict(float)
    items: dict[str, dict] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, 1):
            uid = f"{item['video_id']}_{item['start']:.3f}_{item['end']:.3f}"
            scores[uid] += 1.0 / (k + rank)
            items.setdefault(uid, item)
    merged = [items[uid] for uid in sorted(scores, key=scores.__getitem__, reverse=True)]
    for item, uid in zip(merged, sorted(scores, key=scores.__getitem__, reverse=True)):
        item["rrf_score"] = scores[uid]
    return merged


def _dedup(results: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    kept: list[dict] = []
    for cand in results:
        dominated = any(
            cand["video_id"] == ex["video_id"] and
            _iou(cand["start"], cand["end"], ex["start"], ex["end"]) > iou_threshold
            for ex in kept
        )
        if not dominated:
            kept.append(cand)
    return kept


def _sparse_dot(query_sparse: dict[str, float], doc_list: list[dict], top_k: int) -> list[tuple[int, float]]:
    scores = [(i, sum(query_sparse.get(k, 0.0) * v for k, v in doc.items()))
              for i, doc in enumerate(doc_list)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ── Загрузка индексов ─────────────────────────────────────────────────────────

print("[startup] загружаем индексы...")
faiss_scenes = faiss.read_index(str(FAISS_SCENES))
faiss_events = faiss.read_index(str(FAISS_EVENTS))
scenes_meta: list[dict] = _load_pickle(SCENES_META)
events_meta: list[dict] = _load_pickle(EVENTS_META)
sparse_scenes: list[dict] = _load_pickle(SPARSE_SCENES)
sparse_events: list[dict] = _load_pickle(SPARSE_EVENTS)
bm25_scenes = _load_pickle(BM25_SCENES)
bm25_events = _load_pickle(BM25_EVENTS)

# scene lookup для резолвинга event → scene
print("[startup] строим scene lookup...")
scene_lookup: dict[str, dict] = {}
if SCENES_JSONL.exists():
    import json
    with open(SCENES_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                doc = json.loads(line)
                scene_lookup[f"{doc['video_id']}__{doc['scene_idx']}"] = doc

print("[startup] загружаем BGE-M3...")
bge_model = BGEM3FlagModel(BGE_MODEL, use_fp16=True)

print("[startup] загружаем reranker...")
reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)

print("[startup] загружаем query preprocessor...")
train_csv = ROOT / "data" / "train_qa.csv"
qp = QueryPreprocessor(str(train_csv) if train_csv.exists() else None, use_sage=False)

print("[startup] готово.")


# ── Поиск ─────────────────────────────────────────────────────────────────────

def _encode_query(query: str) -> dict:
    out = bge_model.encode([query], return_dense=True, return_sparse=True)
    dense = out["dense_vecs"][0]
    sparse = {str(k): float(v) for k, v in out["lexical_weights"][0].items()}
    return {"dense": dense, "sparse": sparse}


def _faiss_search(index, meta: list[dict], vec: np.ndarray, top_k: int, label: str) -> list[dict]:
    scores, indices = index.search(vec.reshape(1, -1).astype(np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc = meta[idx]
        results.append({
            "video_id": doc["video_id"],
            "start": doc.get("start", 0.0),
            "end": doc.get("end", 0.0),
            "scene_idx": doc.get("scene_idx"),
            "center_scene_idx": doc.get("center_scene_idx"),
            "text": doc.get("summary", "") or doc.get("asr_text", "") or doc.get("event_summary", ""),
            "source": label,
            "faiss_score": float(score),
        })
    return results


def _bm25_search(bm25, meta: list[dict], query: str, top_k: int, label: str) -> list[dict]:
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            break
        doc = meta[idx]
        results.append({
            "video_id": doc["video_id"],
            "start": doc.get("start", 0.0),
            "end": doc.get("end", 0.0),
            "scene_idx": doc.get("scene_idx"),
            "center_scene_idx": doc.get("center_scene_idx"),
            "text": doc.get("asr_text", "") or doc.get("summary", "") or doc.get("event_summary", ""),
            "source": label,
            "bm25_score": float(scores[idx]),
        })
    return results


def _sparse_search(query_sparse: dict, doc_list: list[dict], meta: list[dict], top_k: int, label: str) -> list[dict]:
    ranked = _sparse_dot(query_sparse, doc_list, top_k)
    results = []
    for idx, score in ranked:
        if score <= 0:
            break
        doc = meta[idx]
        results.append({
            "video_id": doc["video_id"],
            "start": doc.get("start", 0.0),
            "end": doc.get("end", 0.0),
            "scene_idx": doc.get("scene_idx"),
            "center_scene_idx": doc.get("center_scene_idx"),
            "text": doc.get("asr_text", "") or doc.get("summary", "") or doc.get("event_summary", ""),
            "source": label,
            "sparse_score": float(score),
        })
    return results


def _resolve_event(result: dict) -> dict:
    center = result.get("center_scene_idx")
    if center is None:
        return result
    key = f"{result['video_id']}__{center}"
    scene = scene_lookup.get(key)
    if scene:
        result = result.copy()
        result["start"] = scene["start"]
        result["end"] = scene["end"]
        result["scene_idx"] = scene["scene_idx"]
    return result


def search(query: str, top_n: int = FINAL_TOP_N) -> list[dict]:
    clean_query = qp(query)
    encoded = _encode_query(clean_query)
    dense_vec = encoded["dense"]
    sparse_vec = encoded["sparse"]

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(_faiss_search, faiss_scenes, scenes_meta, dense_vec, TOP_K_DENSE, "dense_scenes"),
            pool.submit(_faiss_search, faiss_events, events_meta, dense_vec, TOP_K_DENSE, "dense_events"),
            pool.submit(_bm25_search, bm25_scenes, scenes_meta, clean_query, TOP_K_SPARSE, "bm25_scenes"),
            pool.submit(_bm25_search, bm25_events, events_meta, clean_query, TOP_K_SPARSE, "bm25_events"),
            pool.submit(_sparse_search, sparse_vec, sparse_scenes, scenes_meta, TOP_K_SPARSE, "sparse_scenes"),
            pool.submit(_sparse_search, sparse_vec, sparse_events, events_meta, TOP_K_SPARSE, "sparse_events"),
        ]
        ranked_lists = [f.result() for f in as_completed(futures)]

    merged = _rrf_merge(ranked_lists)
    deduped = _dedup(merged)
    candidates = deduped[:RERANKER_TOP_K]

    if candidates:
        pairs = [[clean_query, c.get("text", "")] for c in candidates]
        rscores = reranker.compute_score(pairs)
        if isinstance(rscores, float):
            rscores = [rscores]
        for c, s in zip(candidates, rscores):
            c["reranker_score"] = float(s)
        candidates.sort(key=lambda x: x["reranker_score"], reverse=True)

    resolved = [_resolve_event(c) for c in candidates[:top_n]]
    return resolved


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Video RAG Demo API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = FINAL_TOP_N


@app.post("/search")
def search_endpoint(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Пустой запрос")
    results = search(req.query, top_n=req.top_k)
    return {
        "query": req.query,
        "results": [
            {
                "video_id": r["video_id"],
                "scene_idx": r.get("scene_idx"),
                "start": r["start"],
                "end": r["end"],
                "timecode": f"{int(r['start']//60):02d}:{int(r['start']%60):02d}",
                "score": round(r.get("reranker_score", r.get("rrf_score", 0.0)), 4),
                "summary": r.get("text", ""),
                "keyframe_url": f"/keyframes/{r['video_id']}/{r.get('scene_idx', 0)}",
            }
            for r in results
        ],
    }


@app.get("/keyframes/{video_id}/{scene_idx}")
def get_keyframe(video_id: str, scene_idx: int):
    path = KEYFRAMES_DIR / video_id / f"scene_{scene_idx:04d}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Кадр не найден: {path}")
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "scenes": len(scenes_meta),
        "events": len(events_meta),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)