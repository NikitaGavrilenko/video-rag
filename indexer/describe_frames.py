"""
describe_frames.py
------------------
Шаг 2: Для каждого кадра просит LLaVA (через Ollama) написать
        текстовое описание сцены на русском языке.

Использование:
    python indexer\describe_frames.py --title "Аватар 3"
"""

import argparse
import json
import time
from pathlib import Path

import ollama
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
FRAMES_DIR = ROOT / "data" / "frames"

BATCH_SAVE_EVERY = 10

PROMPT = """Опиши эту сцену из фильма на русском языке в 2-3 предложениях.
Укажи: что происходит, где (интерьер/экстерьер, время суток, погода),
кто на экране (пол, возраст, эмоции, действия), ключевые объекты.
Пиши кратко и информативно, без лишних слов."""


def slugify(title: str) -> str:
    s = title.lower()
    for c in ' :/_\\,.':
        s = s.replace(c, '_')
    return s


def describe_frame(image_path: str, model: str, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": PROMPT,
                    "images": [image_path],
                }],
                options={"temperature": 0.1, "num_predict": 150}
            )
            return response["message"]["content"].strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
            else:
                print(f"\n    [error] не удалось описать кадр: {e}")
                return ""


def describe_all_frames(title: str, model: str):
    slug = slugify(title)
    movie_dir = FRAMES_DIR / slug
    meta_path = movie_dir / "metadata.json"
    descriptions_path = movie_dir / "descriptions.json"

    if not meta_path.exists():
        print(f"  [error] Метаданные не найдены: {meta_path}")
        print(f"  Сначала запусти: python indexer\\extract_frames.py --title \"{title}\"")
        return {}

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    descriptions = {}
    if descriptions_path.exists():
        with open(descriptions_path, encoding="utf-8") as f:
            descriptions = json.load(f)
        print(f"  [resume] найдено {len(descriptions)} готовых описаний")

    frames_to_process = [f for f in metadata if f not in descriptions]

    if not frames_to_process:
        print(f"  [skip] все кадры уже описаны!")
        return descriptions

    print(f"  [llava] описываем {len(frames_to_process)} кадров (модель: {model})")
    print(f"  Примерное время: ~{len(frames_to_process) * 8 // 60} мин на CPU\n")

    for i, fname in enumerate(tqdm(frames_to_process, desc="Описание кадров")):
        frame_info = metadata[fname]
        description = describe_frame(frame_info["frame_path"], model)
        descriptions[fname] = {**frame_info, "visual_description": description}

        if (i + 1) % BATCH_SAVE_EVERY == 0:
            with open(descriptions_path, "w", encoding="utf-8") as f:
                json.dump(descriptions, f, ensure_ascii=False, indent=2)

    with open(descriptions_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)

    print(f"\n  [ok] описания сохранены → {descriptions_path.name}")
    return descriptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True, help="Название фильма")
    parser.add_argument("--model", default="moondream", help="Ollama модель (default: llava)")
    args = parser.parse_args()

    try:
        ollama.list()
    except Exception:
        print("  [error] Ollama не запущена!")
        print("  Запусти в отдельном терминале: ollama serve")
        return

    print(f"\n{'='*50}")
    print(f"  Описываем кадры: {args.title}")
    print(f"  Модель: {args.model}")
    print(f"{'='*50}")

    descriptions = describe_all_frames(args.title, args.model)

    if descriptions:
        print(f"\n  Готово! {len(descriptions)} описаний.")
        print(f'  Следующий шаг: python indexer\\transcribe.py --title "{args.title}"')


if __name__ == "__main__":
    main()