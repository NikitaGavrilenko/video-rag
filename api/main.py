"""
api/main.py
-----------
FastAPI сервер.

Запуск:
    uvicorn main:app --reload --port 8000
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
CHROMA_DIR = ROOT / "data" / "chroma_db"
FRAMES_DIR = ROOT / "data" / "frames"
VIDEOS_DIR = ROOT / "data" / "videos"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "video_scenes"

print("[startup] загружаем модель эмбеддингов ...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("[startup] подключаемся к ChromaDB ...")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print(f"[startup] в БД {collection.count()} сцен")

app = FastAPI(title="Okko Video RAG", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRAMES_DIR.exists():
    app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    movie_filter: str | None = None


class SceneResult(BaseModel):
    movie_title: str
    movie_slug: str
    timecode_str: str
    timecode_seconds: float
    visual_description: str
    subtitle_text: str
    frame_url: str
    match_score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SceneResult]
    total_scenes_in_db: int


@app.get("/health")
def health():
    return {"status": "ok", "scenes_in_db": collection.count()}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    query_embedding = embed_model.encode(req.query).tolist()

    where = None
    if req.movie_filter:
        where = {"movie_slug": req.movie_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(req.top_k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    scenes = []
    if results["ids"][0]:
        for i, scene_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            match_score = round(1 - distance / 2, 3)

            frame_path = meta.get("frame_path", "")
            frame_url = ""
            if frame_path:
                try:
                    rel = Path(frame_path).relative_to(FRAMES_DIR)
                    frame_url = f"/frames/{rel.as_posix()}"
                except ValueError:
                    frame_url = ""

            scenes.append(SceneResult(
                movie_title=meta.get("movie_title", ""),
                movie_slug=meta.get("movie_slug", ""),
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
    all_meta = collection.get(include=["metadatas"])
    movies = {}
    for meta in all_meta["metadatas"]:
        slug = meta.get("movie_slug", "")
        if slug and slug not in movies:
            movies[slug] = meta.get("movie_title", slug)
    return {"movies": movies}


@app.get("/video/{slug}")
def get_video(slug: str):
    """Отдаёт видеофайл для воспроизведения в браузере."""
    for ext in (".mp4", ".mkv", ".avi", ".MP4", ".MKV", ".AVI"):
        path = VIDEOS_DIR / f"{slug}{ext}"
        if path.exists():
            return FileResponse(
                str(path),
                media_type="video/mp4",
                headers={"Accept-Ranges": "bytes"},
            )
    # Ищем файл с похожим именем (на случай расхождения slug)
    if VIDEOS_DIR.exists():
        for f in VIDEOS_DIR.iterdir():
            if f.stem.lower().replace(" ", "_") == slug.lower():
                return FileResponse(
                    str(f),
                    media_type="video/mp4",
                    headers={"Accept-Ranges": "bytes"},
                )
    raise HTTPException(status_code=404, detail=f"Видео не найдено: {slug}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)