"""
-----------
FastAPI сервер с одним endpoint: POST /search

Запуск:
    uvicorn main:app --reload --port 8000

Пример запроса:
    curl -X POST http://localhost:8000/search \
         -H "Content-Type: application/json" \
         -d '{"query": "герой стоит под дождём и плачет", "top_k": 5}'
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chromadb
from sentence_transformers import SentenceTransformer

# ─── Пути ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CHROMA_DIR = ROOT / "data" / "chroma_db"
FRONTEND_DIR = ROOT / "frontend"

# ─── Настройки ────────────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "video_scenes"

# ─── Инициализация (один раз при старте) ─────────────────────────────────────
print("[startup] загружаем модель эмбеддингов ...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("[startup] подключаемся к ChromaDB ...")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print(f"[startup] в БД {collection.count()} сцен")

# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Okko Video RAG", version="1.0")

# Разрешаем CORS — чтобы фронт мог обращаться к API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Раздаём статику (кадры фильмов) — фронт показывает превью сцены
frames_dir = ROOT / "data" / "frames"
if frames_dir.exists():
    app.mount("/frames", StaticFiles(directory=str(frames_dir)), name="frames")


# ─── Схемы запросов/ответов ───────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    movie_filter: str | None = None  # фильтр по конкретному фильму (опционально)


class SceneResult(BaseModel):
    movie_title: str
    timecode_str: str
    timecode_seconds: float
    visual_description: str
    subtitle_text: str
    frame_url: str        # URL превью кадра
    match_score: float    # 0..1, чем выше — тем лучше совпадение


class SearchResponse(BaseModel):
    query: str
    results: list[SceneResult]
    total_scenes_in_db: int


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "scenes_in_db": collection.count()}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    # Создаём эмбеддинг запроса
    query_embedding = embed_model.encode(req.query).tolist()

    # Фильтр по фильму (если задан)
    where = None
    if req.movie_filter:
        where = {"movie_slug": req.movie_filter}

    # Ищем в ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(req.top_k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Формируем ответ
    scenes = []
    if results["ids"][0]:
        for i, scene_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            # ChromaDB возвращает косинусное расстояние (0=идеально, 2=противоположно)
            # Конвертируем в score 0..1
            match_score = round(1 - distance / 2, 3)

            # Строим URL к превью кадра
            frame_path = meta.get("frame_path", "")
            frame_url = ""
            if frame_path:
                # Превращаем абсолютный путь в URL: /frames/slug/frame_00042.jpg
                try:
                    rel = Path(frame_path).relative_to(frames_dir)
                    frame_url = f"/frames/{rel}"
                except ValueError:
                    frame_url = ""

            scenes.append(SceneResult(
                movie_title=meta.get("movie_title", ""),
                timecode_str=meta.get("timecode_str", ""),
                timecode_seconds=meta.get("timecode_seconds", 0),
                visual_description=meta.get("visual_description", ""),
                subtitle_text=meta.get("subtitle_text", ""),
                frame_url=frame_url,
                match_score=match_score,
            ))

    return SearchResponse(
        query=req.query,
        results=scenes,
        total_scenes_in_db=collection.count(),
    )


@app.get("/movies")
def list_movies():
    """Список всех проиндексированных фильмов."""
    all_meta = collection.get(include=["metadatas"])
    movies = {}
    for meta in all_meta["metadatas"]:
        slug = meta.get("movie_slug", "")
        if slug and slug not in movies:
            movies[slug] = meta.get("movie_title", slug)
    return {"movies": movies}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)