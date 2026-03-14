"""
build_scenes.py — Шаг 5: e5 embeddings для текстовых описаний и субтитров.

Создаёт две коллекции в ChromaDB:
  - text_visual:    описание кадра + соседние субтитры (768d, e5)
  - text_subtitles: каждая реплика отдельно (768d, e5)

Использование:
    python indexer/build_scenes.py --title "Амели"
"""

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
FRAMES_DIR = ROOT / "data" / "frames"
CHROMA_DIR = ROOT / "data" / "chroma_db"

E5_MODEL = "intfloat/multilingual-e5-base"
TEXT_VISUAL_COLLECTION = "text_visual"
TEXT_SUBS_COLLECTION = "text_subtitles"
BATCH_SIZE = 64
SUB_WINDOW_SEC = 30  # субтитры в ±30с от кадра идут в контекст


def slugify(title: str) -> str:
    s = title.lower()
    for c in " :/_\\,.":
        s = s.replace(c, "_")
    return s


def subs_near(subtitles: list, t: float, window: float = SUB_WINDOW_SEC) -> str:
    """Субтитры в окне вокруг таймкода — для контекста визуальной сцены."""
    texts = [s["text"] for s in subtitles if t - window <= s["start"] <= t + window]
    return " ".join(texts).strip()


def index_batch(collection, ids, texts, metadatas, model, prefix="passage: "):
    """e5 требует префикс 'passage: ' для индексируемых документов."""
    prefixed = [prefix + t for t in texts]
    embeddings = model.encode(prefixed, show_progress_bar=False, normalize_embeddings=True).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def build_text_scenes(title: str):
    slug = slugify(title)
    movie_dir = FRAMES_DIR / slug
    desc_path = movie_dir / "descriptions.json"
    subs_path = movie_dir / "subtitles.json"

    if not desc_path.exists():
        print(f"  [error] нет descriptions.json. Сначала: describe_frames.py")
        return

    descriptions = json.loads(desc_path.read_text(encoding="utf-8"))
    subtitles = json.loads(subs_path.read_text(encoding="utf-8")) if subs_path.exists() else []
    if subtitles:
        print(f"  [info] субтитры: {len(subtitles)} реплик")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # ── Коллекция 1: text_visual ──────────────────────────────────────────────
    tv_col = client.get_or_create_collection(TEXT_VISUAL_COLLECTION, metadata={"hnsw:space": "cosine"})
    existing = tv_col.get(where={"movie_slug": slug})

    if existing["ids"]:
        print(f"  [skip] text_visual уже есть: {len(existing['ids'])} сцен")
    else:
        print(f"  [e5] загружаем модель '{E5_MODEL}' ...")
        model = SentenceTransformer(E5_MODEL)

        scenes = []
        for fname, info in descriptions.items():
            vis = info.get("visual_description", "").strip()
            sub_ctx = subs_near(subtitles, info["timecode_seconds"])

            parts = []
            if vis:
                parts.append(vis)
            if sub_ctx:
                parts.append(f"Диалог: {sub_ctx}")
            text = " ".join(parts)
            if not text:
                continue

            scenes.append({
                "id": f"{slug}_tv_{fname.replace('.jpg', '')}",
                "text": text,
                "meta": {
                    "movie_title": title,
                    "movie_slug": slug,
                    "timecode_seconds": info["timecode_seconds"],
                    "timecode_str": info["timecode_str"],
                    "frame_path": info["frame_path"],
                    "visual_description": vis,
                    "subtitle_context": sub_ctx,
                    "scene_type": "visual",
                },
            })

        print(f"  [e5] text_visual: кодируем {len(scenes)} сцен ...")
        for i in tqdm(range(0, len(scenes), BATCH_SIZE), desc="text_visual"):
            b = scenes[i:i + BATCH_SIZE]
            index_batch(tv_col, [s["id"] for s in b], [s["text"] for s in b], [s["meta"] for s in b], model)
        print(f"  [ok] {len(scenes)} сцен → '{TEXT_VISUAL_COLLECTION}'")

    # ── Коллекция 2: text_subtitles ───────────────────────────────────────────
    ts_col = client.get_or_create_collection(TEXT_SUBS_COLLECTION, metadata={"hnsw:space": "cosine"})
    existing_subs = ts_col.get(where={"movie_slug": slug})

    if existing_subs["ids"]:
        print(f"  [skip] text_subtitles уже есть: {len(existing_subs['ids'])} реплик")
    elif subtitles:
        if "model" not in dir():
            model = SentenceTransformer(E5_MODEL)

        # Для каждой реплики находим ближайший кадр (для превью)
        frame_list = [(fn, info) for fn, info in descriptions.items()]

        sub_scenes = []
        for i, sub in enumerate(subtitles):
            text = sub["text"].strip()
            if not text:
                continue

            nearest = min(frame_list, key=lambda x: abs(x[1]["timecode_seconds"] - sub["start"]))
            nearest_info = nearest[1]

            sub_scenes.append({
                "id": f"{slug}_sub_{i:05d}",
                "text": text,
                "meta": {
                    "movie_title": title,
                    "movie_slug": slug,
                    "timecode_seconds": sub["start"],
                    "timecode_str": sub["start_str"],
                    "frame_path": nearest_info["frame_path"],
                    "visual_description": nearest_info.get("visual_description", ""),
                    "subtitle_text": text,
                    "scene_type": "subtitle",
                },
            })

        print(f"  [e5] text_subtitles: кодируем {len(sub_scenes)} реплик ...")
        for i in tqdm(range(0, len(sub_scenes), BATCH_SIZE), desc="text_subtitles"):
            b = sub_scenes[i:i + BATCH_SIZE]
            index_batch(ts_col, [s["id"] for s in b], [s["text"] for s in b], [s["meta"] for s in b], model)
        print(f"  [ok] {len(sub_scenes)} реплик → '{TEXT_SUBS_COLLECTION}'")
    else:
        print(f"  [warn] субтитры не найдены, коллекция text_subtitles пуста")

    print(f"\n  Итого в БД:")
    print(f"    text_visual:    {tv_col.count()}")
    print(f"    text_subtitles: {ts_col.count()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    print(f"\n{'='*50}\n  build_scenes: {args.title}\n{'='*50}")
    build_text_scenes(args.title)
    print(f"\n  Готово! Запусти API: cd api && uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()
