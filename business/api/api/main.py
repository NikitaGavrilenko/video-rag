"""
api/main.py — FastAPI сервер.

Эндпоинты:
  POST /search        — текстовый поиск (CLIP + e5, мёрдж)
  POST /search/image  — поиск по скриншоту (CLIP image)
  POST /qa            — вопрос о сюжете (RAG → Groq)
  GET  /video/{slug}  — видеофайл
  GET  /movies        — список фильмов
  GET  /health

Запуск:
    cd api && uvicorn main:app --reload --port 8000
"""

import io
import os
from pathlib import Path
from typing import Optional

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from PIL import Image
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import requests as _requests

load_dotenv()

ROOT = Path(__file__).parent.parent
FRAMES_DIR = ROOT / "data" / "frames"
VIDEOS_DIR = ROOT / "data" / "videos"
CHROMA_DIR = ROOT / "data" / "chroma_db"

# clip-ViT-B-32 — image encoder (для индексации кадров и поиска по скриншоту)
# clip-ViT-B-32-multilingual-v1 — text encoder с поддержкой русского
# Оба в одном CLIP пространстве → cross-modal поиск работает
CLIP_IMAGE_MODEL_NAME = "clip-ViT-B-32"
CLIP_TEXT_MODEL_NAME  = "clip-ViT-B-32-multilingual-v1"
E5_MODEL_NAME = "intfloat/multilingual-e5-base"

CLIP_COLLECTION = "clip_visual"
TEXT_VISUAL_COLLECTION = "text_visual"
TEXT_SUBS_COLLECTION = "text_subtitles"

DEDUP_WINDOW_SEC = 3.0   # считаем дублями сцены в ±3с
QA_TOP_K = 8             # сцен в контекст для QA


# ─── Инициализация ────────────────────────────────────────────────────────────
print("[startup] загружаем CLIP image модель ...")
clip_image_model = SentenceTransformer(CLIP_IMAGE_MODEL_NAME)

print("[startup] загружаем CLIP text (multilingual) модель ...")
clip_text_model = SentenceTransformer(CLIP_TEXT_MODEL_NAME)

print("[startup] загружаем e5 модель ...")
e5_model = SentenceTransformer(E5_MODEL_NAME)

print("[startup] подключаемся к ChromaDB ...")
chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
clip_col = chroma.get_or_create_collection(CLIP_COLLECTION, metadata={"hnsw:space": "cosine"})
tv_col = chroma.get_or_create_collection(TEXT_VISUAL_COLLECTION, metadata={"hnsw:space": "cosine"})
ts_col = chroma.get_or_create_collection(TEXT_SUBS_COLLECTION, metadata={"hnsw:space": "cosine"})


print(f"[startup] готово. CLIP: {clip_col.count()}, text_visual: {tv_col.count()}, subtitles: {ts_col.count()}")

app = FastAPI(title="Video RAG API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if FRAMES_DIR.exists():
    app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def frame_url(frame_path: str) -> str:
    if not frame_path:
        return ""
    try:
        rel = Path(frame_path).relative_to(FRAMES_DIR)
        return f"/frames/{rel.as_posix()}"
    except ValueError:
        return ""


def dedup_by_timecode(scenes: list, window: float = DEDUP_WINDOW_SEC) -> list:
    """Убираем дубли: из сцен в пределах ±window оставляем с лучшим скором."""
    scenes.sort(key=lambda x: -x["match_score"])
    result = []
    for s in scenes:
        t = s["timecode_seconds"]
        if not any(abs(t - r["timecode_seconds"]) <= window for r in result):
            result.append(s)
    return result


def query_collection(collection, embedding: list, n: int, movie_filter: Optional[str]) -> list:
    if collection.count() == 0:
        return []
    where = {"movie_slug": movie_filter} if movie_filter else None
    kwargs = dict(query_embeddings=[embedding], n_results=min(n, collection.count()), include=["metadatas", "distances"])
    if where:
        kwargs["where"] = where
    res = collection.query(**kwargs)
    out = []
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        out.append({**meta, "match_score": round(1 - dist / 2, 4)})
    return out


def scene_to_dict(meta: dict) -> dict:
    return {
        "movie_title": meta.get("movie_title", ""),
        "movie_slug": meta.get("movie_slug", ""),
        "timecode_str": meta.get("timecode_str", ""),
        "timecode_seconds": meta.get("timecode_seconds", 0),
        "visual_description": meta.get("visual_description", ""),
        "subtitle_text": meta.get("subtitle_text", meta.get("subtitle_context", "")),
        "frame_url": frame_url(meta.get("frame_path", "")),
        "match_score": meta.get("match_score", 0),
        "scene_type": meta.get("scene_type", "visual"),
    }


# ─── Models ──────────────────────────────────────────────────────────────────

class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 6
    movie_filter: Optional[str] = None


class QARequest(BaseModel):
    question: str
    movie_filter: Optional[str] = None


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "clip_scenes": clip_col.count(),
        "text_visual": tv_col.count(),
        "text_subtitles": ts_col.count(),
    }


@app.get("/movies")
def list_movies():
    movies = {}
    for col in (clip_col, tv_col, ts_col):
        if col.count() == 0:
            continue
        for meta in col.get(include=["metadatas"])["metadatas"]:
            slug = meta.get("movie_slug", "")
            if slug and slug not in movies:
                movies[slug] = meta.get("movie_title", slug)
    return {"movies": movies}


@app.post("/search")
def text_search(req: TextSearchRequest):
    """
    Текстовый поиск: запрос → CLIP text encoder + e5 → мёрдж топ-N из каждого.
    """
    n = req.top_k * 2  # берём с запасом, потом дедупим

    # CLIP: multilingual text encoder
    clip_emb = clip_text_model.encode(req.query, convert_to_numpy=True).tolist()
    clip_res = query_collection(clip_col, clip_emb, n, req.movie_filter)

    # e5: prefix "query: " для поисковых запросов
    e5_emb = e5_model.encode("query: " + req.query, normalize_embeddings=True, convert_to_numpy=True).tolist()
    tv_res = query_collection(tv_col, e5_emb, n, req.movie_filter)
    ts_res = query_collection(ts_col, e5_emb, n, req.movie_filter)

    # Мёрдж: собираем всё, дедупликация по таймкоду
    all_scenes = [scene_to_dict(s) for s in clip_res + tv_res + ts_res]
    merged = dedup_by_timecode(all_scenes)[:req.top_k]

    return {
        "query": req.query,
        "results": merged,
        "sources": {
            "clip": len(clip_res),
            "text_visual": len(tv_res),
            "text_subtitles": len(ts_res),
        },
    }


@app.post("/search/image")
async def image_search(file: UploadFile = File(...), top_k: int = 6, movie_filter: Optional[str] = None):
    """
    Поиск по скриншоту: изображение → CLIP image encoder → похожие кадры.
    """
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    clip_emb = clip_image_model.encode(image, convert_to_numpy=True).tolist()
    results = query_collection(clip_col, clip_emb, top_k * 2, movie_filter)
    merged = dedup_by_timecode([scene_to_dict(s) for s in results])[:top_k]

    return {"results": merged, "search_type": "image"}


@app.post("/qa")
def qa(req: QARequest):
    """
    QA: вопрос → найти релевантные сцены → Groq llama-4 → текстовый ответ + ссылки.
    """
    print(f"[qa] вопрос: {req.question}")
    # Находим релевантные сцены через e5
    # Расширяем запрос
    expanded = expand_query(req.question)
    print(f"[qa] expanded: {expanded}")
    search_text = req.question + "\n" + expanded
    print(f"[qa] expanded queries:\n{expanded}")

    e5_emb = e5_model.encode("query: " + search_text, normalize_embeddings=True, convert_to_numpy=True).tolist()
    print(f"[qa] embedding готов")
    tv_res = query_collection(tv_col, e5_emb, QA_TOP_K, req.movie_filter)
    ts_res = query_collection(ts_col, e5_emb, QA_TOP_K, req.movie_filter)
    all_scenes = dedup_by_timecode([scene_to_dict(s) for s in tv_res + ts_res])[:QA_TOP_K]

    if not all_scenes:
        return {"question": req.question, "answer": "Не найдено релевантных сцен в базе.", "sources": []}

    # Строим контекст для LLM
    context_parts = []
    for i, s in enumerate(all_scenes):
        parts = [f"[Сцена {i+1}] {s['movie_title']} • {s['timecode_str']}"]
        if s["visual_description"]:
            parts.append(f"Визуал: {s['visual_description']}")
        if s["subtitle_text"]:
            parts.append(f"Диалог: \"{s['subtitle_text']}\"")
        context_parts.append("\n".join(parts))

    context = "\n\n".join(context_parts)

    system_prompt = (
        "Ты — интеллектуальный помощник по фильмам. "
        "Тебе дан контекст из конкретных сцен фильма (визуальные описания и диалоги). "
        "Отвечай на вопрос пользователя на русском языке, опираясь только на этот контекст. "
        "Если в контексте недостаточно информации — так и скажи. "
        "В конце ответа укажи: [Источники: Сцена N, Сцена M, ...]"
    )

    resp = _requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": "arcee-ai/trinity-large-preview:free",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {req.question}"},
            ],
            "max_tokens": 600,
            "temperature": 0.3,
        },
        timeout=60,
    )
    print(resp.json())
    answer = resp.json()["choices"][0]["message"]["content"].strip()

    return {
        "question": req.question,
        "answer": answer,
        "sources": all_scenes,
    }

def expand_query(question: str) -> str:
    """LLM переформулирует вопрос в поисковые ключевые слова."""
    resp = _requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": "arcee-ai/trinity-large-preview:free",
            "messages": [{"role": "user", "content": 
                f"""Вопрос о фильме: «{question}»
                Что спрашивает пользователь? Перефразируй это как 3-4 коротких поисковых запроса.
                Запросы должны описывать то, что можно УВИДЕТЬ на экране или УСЛЫШАТЬ в диалоге.

                Пример:
                Вопрос: «Почему Джейк согласился на миссию?»
                Запросы:
                тебе предлагают работу
                сядь в кресло
                ты нужен нам
                деньги операция ноги

                Теперь для вопроса выше. Только запросы, каждый с новой строки."""
            }],
            "max_tokens": 200,
        },
        timeout=30,
    )
    print(resp.json())
    return resp.json()["choices"][0]["message"]["content"].strip()

@app.get("/video/{slug}")
def get_video(slug: str):
    for ext in (".mp4", ".mkv", ".avi", ".MP4", ".MKV", ".AVI"):
        path = VIDEOS_DIR / f"{slug}{ext}"
        if path.exists():
            return FileResponse(str(path), media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    if VIDEOS_DIR.exists():
        for f in VIDEOS_DIR.iterdir():
            if f.stem.lower().replace(" ", "_") == slug.lower():
                return FileResponse(str(f), media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    raise HTTPException(status_code=404, detail=f"Видео не найдено: {slug}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)