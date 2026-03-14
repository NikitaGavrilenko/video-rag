"""
embed_clip.py — Шаг 4: CLIP image embeddings для каждого кадра.

Использует sentence-transformers/clip-ViT-B-32-multilingual-v1 —
модель, которая умеет сравнивать РУССКИЙ текст и картинки
в одном векторном пространстве (512d).

Использование:
    python indexer/embed_clip.py --title "Амели"
"""

import argparse
import json
from pathlib import Path

import chromadb
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
FRAMES_DIR = ROOT / "data" / "frames"
CHROMA_DIR = ROOT / "data" / "chroma_db"

CLIP_MODEL = "clip-ViT-B-32"  # image encoder; для текстовых запросов используем multilingual вариант
CLIP_COLLECTION = "clip_visual"
BATCH_SIZE = 32


def slugify(title: str) -> str:
    s = title.lower()
    for c in " :/_\\,.":
        s = s.replace(c, "_")
    return s


def embed_clip_frames(title: str):
    slug = slugify(title)
    movie_dir = FRAMES_DIR / slug
    meta_path = movie_dir / "metadata.json"

    if not meta_path.exists():
        print(f"  [error] нет metadata.json. Сначала: extract_frames.py")
        return

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=CLIP_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.get(where={"movie_slug": slug})
    if existing["ids"]:
        print(f"  [skip] CLIP embeddings уже есть: {len(existing['ids'])} кадров")
        return

    print(f"  [clip] загружаем модель '{CLIP_MODEL}' ...")
    model = SentenceTransformer(CLIP_MODEL)

    frames = sorted(metadata.keys())
    print(f"  [clip] кодируем {len(frames)} кадров ...")

    for i in tqdm(range(0, len(frames), BATCH_SIZE), desc="CLIP embed"):
        batch_names = frames[i:i + BATCH_SIZE]
        images = []
        valid_names = []

        for fname in batch_names:
            path = Path(metadata[fname]["frame_path"])
            if path.exists():
                images.append(Image.open(path).convert("RGB"))
                valid_names.append(fname)

        if not images:
            continue

        embeddings = model.encode(images, show_progress_bar=False, convert_to_numpy=True, batch_size=BATCH_SIZE).tolist()

        collection.add(
            ids=[f"{slug}_clip_{n.replace('.jpg', '')}" for n in valid_names],
            embeddings=embeddings,
            metadatas=[{
                "movie_title": title,
                "movie_slug": slug,
                "frame_name": n,
                "timecode_seconds": metadata[n]["timecode_seconds"],
                "timecode_str": metadata[n]["timecode_str"],
                "frame_path": metadata[n]["frame_path"],
            } for n in valid_names],
            documents=[f"{title} frame {metadata[n]['timecode_str']}" for n in valid_names],
        )

    print(f"  [ok] {len(frames)} CLIP embeddings → коллекция '{CLIP_COLLECTION}'")
    print(f"  Всего в коллекции: {collection.count()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    print(f"\n{'='*50}\n  embed_clip: {args.title}\n{'='*50}")
    embed_clip_frames(args.title)


if __name__ == "__main__":
    main()