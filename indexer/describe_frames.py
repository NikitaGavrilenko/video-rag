"""
describe_frames.py
------------------
Шаг 2: Описывает кадры через Groq API (llama-3.2-11b-vision).

Установка:
    pip install groq

Получить API ключ (бесплатно, доступен из России):
    https://console.groq.com → API Keys

Использование:
    set GROQ_API_KEY=ваш_ключ        # Windows
    export GROQ_API_KEY=ваш_ключ     # Linux/Mac

    python indexer/describe_frames.py --title "Аватар 3"
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).parent.parent
FRAMES_DIR = ROOT / "data" / "frames"

BATCH_SAVE_EVERY = 20

# meta-llama/llama-4-scout-17b-16e-instruct — бесплатно, хорошее качество
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Бесплатный план: 30 req/min, 1000 req/day
RATE_LIMIT_DELAY = 2.1  # сек между запросами (чуть больше 60/30)

PROMPT = """Опиши кадр из фильма на русском языке. Только 2-3 предложения, без вступлений.
Формат: что происходит + где (интерьер/экстерьер, время суток) + кто на экране (пол, возраст, эмоции, действия) + ключевые объекты.
Не упоминай название фильма. Не используй заголовки, списки, markdown. Только сплошной текст."""


def slugify(title: str) -> str:
    s = title.lower()
    for c in ' :/_\\,.':
        s = s.replace(c, '_')
    return s


def describe_frame(image_path: str, client: Groq, retries: int = 3) -> str:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }],
                max_tokens=200,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 60
                print(f"\n    [rate limit] ждём {wait} сек ...")
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(3)
            else:
                print(f"\n    [error] {e}")
                return ""
    return ""


def describe_all_frames(title: str):
    slug = slugify(title)
    movie_dir = FRAMES_DIR / slug
    meta_path = movie_dir / "metadata.json"
    descriptions_path = movie_dir / "descriptions.json"

    if not meta_path.exists():
        print(f"  [error] Метаданные не найдены: {meta_path}")
        print(f'  Сначала запусти: python indexer/extract_frames.py --title "{title}"')
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
        print("  [skip] все кадры уже описаны!")
        return descriptions

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("  [error] Нет GROQ_API_KEY!")
        print("  1. Зарегистрируйся: https://console.groq.com")
        print("  2. API Keys → Create API Key")
        print("  3. set GROQ_API_KEY=ваш_ключ")
        return {}

    client = Groq(api_key="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")

    total = len(frames_to_process)
    est_min = int(total * RATE_LIMIT_DELAY // 60)
    print(f"  [groq] описываем {total} кадров (модель: {MODEL})")
    print(f"  Примерное время: ~{est_min} мин\n")

    for i, fname in enumerate(tqdm(frames_to_process, desc="Описание кадров")):
        frame_info = metadata[fname]
        description = describe_frame(frame_info["frame_path"], client)
        descriptions[fname] = {**frame_info, "visual_description": description}

        time.sleep(RATE_LIMIT_DELAY)

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
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Описываем кадры: {args.title}")
    print(f"  Модель: {MODEL}")
    print(f"{'='*50}")

    descriptions = describe_all_frames(args.title)

    if descriptions:
        print(f"\n  Готово! {len(descriptions)} описаний.")
        print(f'  Следующий шаг: python indexer/transcribe.py --title "{args.title}"')


if __name__ == "__main__":
    main()