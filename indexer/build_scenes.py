"""
Шаг 4: Объединяет описания кадров + субтитры в "сцены",
        создаёт эмбеддинги и сохраняет в ChromaDB.

Два типа сцен:
  - visual: один кадр = одна сцена (поиск по визуалу)
  - subtitle: одна реплика = одна сцена (поиск по цитатам, точный таймкод)

Использование:
    python build_scenes.py --title "Амели"
"""

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─── Пути ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
FRAMES_DIR = ROOT / "data" / "frames"
CHROMA_DIR = ROOT / "data" / "chroma_db"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SUBTITLE_WINDOW_SEC = 30
COLLECTION_NAME = "video_scenes"


def slugify(title: str) -> str:
    s = title.lower()
    for c in ' :/_\\,.':
        s = s.replace(c, '_')
    return s


def get_subtitles_near(subtitles: list, timecode_sec: float, window: float = SUBTITLE_WINDOW_SEC) -> str:
    texts = []
    for sub in subtitles:
        if sub["start"] >= timecode_sec - window and sub["end"] <= timecode_sec + window:
            texts.append(sub["text"])
    return " ".join(texts).strip()


def build_and_store(title: str):
    slug = slugify(title)
    movie_dir = FRAMES_DIR / slug
    descriptions_path = movie_dir / "descriptions.json"
    subtitles_path = movie_dir / "subtitles.json"

    if not descriptions_path.exists():
        raise FileNotFoundError(
            f"Описания не найдены: {descriptions_path}\n"
            f'Сначала запусти: python describe_frames.py --title "{title}"'
        )
    with open(descriptions_path, encoding="utf-8") as f:
        descriptions = json.load(f)

    subtitles = []
    if subtitles_path.exists():
        with open(subtitles_path, encoding="utf-8") as f:
            subtitles = json.load(f)
        print(f"  [info] субтитры загружены: {len(subtitles)} фраз")
    else:
        print(f"  [warn] субтитры не найдены, только визуальные сцены")

    print(f"  [chroma] подключаемся к БД: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    existing = collection.get(where={"movie_slug": slug})
    if existing["ids"]:
        print(f"  [skip] {title} уже в БД: {len(existing['ids'])} сцен")
        return

    print(f"  [embed] загружаем модель '{EMBED_MODEL}' ...")
    model = SentenceTransformer(EMBED_MODEL)

    scenes = []

    # ── Тип 1: визуальные сцены (кадр + контекстные субтитры) ────────────────
    for fname, frame_info in descriptions.items():
        timecode_sec = frame_info["timecode_seconds"]
        visual_desc = frame_info.get("visual_description", "")
        subtitle_text = get_subtitles_near(subtitles, timecode_sec)

        parts = []
        if visual_desc:
            parts.append(f"[визуал] {visual_desc}")
        if subtitle_text:
            parts.append(f"[диалог] {subtitle_text}")
        combined_text = " ".join(parts)

        if not combined_text.strip():
            continue

        scenes.append({
            "id": f"{slug}_visual_{fname.replace('.jpg', '')}",
            "text": combined_text,
            "metadata": {
                "scene_type": "visual",
                "movie_title": title,
                "movie_slug": slug,
                "timecode_str": frame_info["timecode_str"],
                "timecode_seconds": timecode_sec,
                "frame_path": frame_info["frame_path"],
                "visual_description": visual_desc,
                "subtitle_text": subtitle_text,
            }
        })

    # ── Тип 2: субтитры как отдельные сцены (точный таймкод реплики) ─────────
    for i, sub in enumerate(subtitles):
        text = sub["text"].strip()
        if not text:
            continue

        # Ближайший кадр — для показа превью
        nearest_frame_path = ""
        nearest_frame_meta = None
        min_diff = float("inf")
        for fname, frame_info in descriptions.items():
            diff = abs(frame_info["timecode_seconds"] - sub["start"])
            if diff < min_diff:
                min_diff = diff
                nearest_frame_path = frame_info["frame_path"]
                nearest_frame_meta = frame_info

        scenes.append({
            "id": f"{slug}_sub_{i:05d}",
            "text": f"[диалог] {text}",
            "metadata": {
                "scene_type": "subtitle",
                "movie_title": title,
                "movie_slug": slug,
                "timecode_str": sub["start_str"],
                "timecode_seconds": sub["start"],  # точный таймкод реплики
                "frame_path": nearest_frame_path,
                "visual_description": nearest_frame_meta.get("visual_description", "") if nearest_frame_meta else "",
                "subtitle_text": text,
            }
        })

    print(f"  [embed] создаём эмбеддинги для {len(scenes)} сцен "
          f"({len(descriptions)} визуальных + {len(subtitles)} субтитров) ...")

    BATCH_SIZE = 50
    for i in tqdm(range(0, len(scenes), BATCH_SIZE), desc="Индексация"):
        batch = scenes[i:i + BATCH_SIZE]
        embeddings = model.encode([s["text"] for s in batch], show_progress_bar=False).tolist()
        collection.add(
            ids=[s["id"] for s in batch],
            embeddings=embeddings,
            documents=[s["text"] for s in batch],
            metadatas=[s["metadata"] for s in batch],
        )

    print(f"  [ok] добавлено {len(scenes)} сцен в ChromaDB")
    print(f"  Всего в коллекции: {collection.count()} сцен")


def main():
    parser = argparse.ArgumentParser(description="Шаг 4: собрать сцены и загрузить в ChromaDB")
    parser.add_argument("--title", required=True, help="Название фильма")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Индексируем: {args.title}")
    print(f"{'='*50}")

    build_and_store(args.title)

    print(f"\n  Готово!")
    print(f"  Следующий шаг: cd api && uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()