"""
---------------
Шаг 4: Объединяет описания кадров + субтитры в "сцены",
        создаёт эмбеддинги и сохраняет в ChromaDB.

Логика: каждый кадр = одна сцена.
К кадру прикрепляются субтитры из ±30 сек вокруг него.

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

# ─── Настройки ────────────────────────────────────────────────────────────────
# Мультиязычная модель — понимает русский запрос
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Сколько секунд субтитров брать вокруг кадра
SUBTITLE_WINDOW_SEC = 30

COLLECTION_NAME = "video_scenes"


def slugify(title: str) -> str:
    return title.lower().replace(" ", "_").replace(":", "").replace("/", "_")


def get_subtitles_near(subtitles: list, timecode_sec: float, window: float = SUBTITLE_WINDOW_SEC) -> str:
    """Возвращает текст субтитров в окне [timecode - window, timecode + window]."""
    texts = []
    for sub in subtitles:
        if sub["start"] >= timecode_sec - window and sub["end"] <= timecode_sec + window:
            texts.append(sub["text"])
    return " ".join(texts).strip()


def build_combined_text(visual_desc: str, subtitle_text: str) -> str:
    """Собирает итоговый текст сцены для эмбеддинга."""
    parts = []
    if visual_desc:
        parts.append(f"[визуал] {visual_desc}")
    if subtitle_text:
        parts.append(f"[диалог] {subtitle_text}")
    return " ".join(parts)


def build_and_store(title: str):
    slug = slugify(title)
    movie_dir = FRAMES_DIR / slug
    descriptions_path = movie_dir / "descriptions.json"
    subtitles_path = movie_dir / "subtitles.json"

    # Загружаем описания кадров (обязательно)
    if not descriptions_path.exists():
        raise FileNotFoundError(
            f"Описания не найдены: {descriptions_path}\n"
            f"Сначала запусти: python describe_frames.py --title \"{title}\""
        )
    with open(descriptions_path, encoding="utf-8") as f:
        descriptions = json.load(f)

    # Субтитры — опционально (могут отсутствовать)
    subtitles = []
    if subtitles_path.exists():
        with open(subtitles_path, encoding="utf-8") as f:
            subtitles = json.load(f)
        print(f"  [info] субтитры загружены: {len(subtitles)} фраз")
    else:
        print(f"  [warn] субтитры не найдены, используем только визуальные описания")

    # Инициализируем ChromaDB
    print(f"  [chroma] подключаемся к БД: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # косинусное расстояние для текстов
    )

    # Проверяем — не индексировали ли уже этот фильм
    existing = collection.get(where={"movie_slug": slug})
    if existing["ids"]:
        print(f"  [skip] {title} уже в БД: {len(existing['ids'])} сцен")
        return

    # Загружаем модель эмбеддингов
    print(f"  [embed] загружаем модель '{EMBED_MODEL}' ...")
    model = SentenceTransformer(EMBED_MODEL)

    # Собираем сцены
    scenes = []
    for fname, frame_info in descriptions.items():
        timecode_sec = frame_info["timecode_seconds"]
        visual_desc = frame_info.get("visual_description", "")
        subtitle_text = get_subtitles_near(subtitles, timecode_sec)
        combined_text = build_combined_text(visual_desc, subtitle_text)

        if not combined_text.strip():
            continue  # пропускаем пустые сцены

        scene_id = f"{slug}_{fname.replace('.jpg', '')}"
        scenes.append({
            "id": scene_id,
            "text": combined_text,
            "metadata": {
                "movie_title": title,
                "movie_slug": slug,
                "timecode_str": frame_info["timecode_str"],
                "timecode_seconds": timecode_sec,
                "frame_path": frame_info["frame_path"],
                "visual_description": visual_desc,
                "subtitle_text": subtitle_text,
            }
        })

    print(f"  [embed] создаём эмбеддинги для {len(scenes)} сцен ...")

    # Батчами добавляем в ChromaDB
    BATCH_SIZE = 50
    for i in tqdm(range(0, len(scenes), BATCH_SIZE), desc="Индексация"):
        batch = scenes[i:i + BATCH_SIZE]

        texts = [s["text"] for s in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=[s["id"] for s in batch],
            embeddings=embeddings,
            documents=texts,
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
    print(f"  Следующий шаг: python ../api/main.py")


if __name__ == "__main__":
    main()