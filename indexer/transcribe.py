"""
transcribe.py — Шаг 3: транскрипция аудио через faster-whisper.

Использование:
    python indexer\transcribe.py --title "Аватар"
"""

import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
VIDEOS_DIR = ROOT / "data" / "videos"
FRAMES_DIR = ROOT / "data" / "frames"

DEVICE = "cpu"
COMPUTE_TYPE = "int8"


def slugify(title: str) -> str:
    s = title.lower()
    for c in ' :/_\\,.':
        s = s.replace(c, '_')
    return s


def seconds_to_timecode(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def find_video(title: str) -> Path:
    """Ищет видеофайл по названию — с учётом регистра и расширений."""
    slug = slugify(title)
    for ext in (".mp4", ".mkv", ".avi"):
        # Пробуем slug
        p = VIDEOS_DIR / f"{slug}{ext}"
        if p.exists():
            return p
    # Ищем любой файл, в имени которого есть slug
    if VIDEOS_DIR.exists():
        for f in VIDEOS_DIR.iterdir():
            if slugify(f.stem) == slug:
                return f
    raise FileNotFoundError(
        f"Видео не найдено в {VIDEOS_DIR}\n"
        f"Ожидалось имя файла похожее на: {slug}.mp4"
    )


def transcribe_video(title: str, model_name: str):
    slug = slugify(title)
    movie_dir = FRAMES_DIR / slug
    subtitles_path = movie_dir / "subtitles.json"

    if subtitles_path.exists():
        print(f"  [skip] субтитры уже есть: {subtitles_path.name}")
        with open(subtitles_path, encoding="utf-8") as f:
            return json.load(f)

    video_path = find_video(title)
    print(f"  [whisper] загружаем модель '{model_name}' ...")
    model = WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE)

    print(f"  [whisper] транскрибируем '{video_path.name}' ...")
    segments, info = model.transcribe(
        str(video_path),
        beam_size=5,
        language=None,
        vad_filter=True,
        word_timestamps=False,
    )
    print(f"  Язык: {info.language} (уверенность: {info.language_probability:.0%})")

    subtitles = []
    for segment in tqdm(segments, desc="Транскрипция"):
        subtitles.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "start_str": seconds_to_timecode(segment.start),
            "end_str": seconds_to_timecode(segment.end),
            "text": segment.text.strip(),
        })

    movie_dir.mkdir(parents=True, exist_ok=True)
    with open(subtitles_path, "w", encoding="utf-8") as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=2)

    print(f"  [ok] {len(subtitles)} фраз → {subtitles_path.name}")
    return subtitles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--model", default="small", help="tiny/small/medium/large-v2")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Транскрибируем: {args.title}")
    print(f"  Модель Whisper: {args.model}")
    print(f"{'='*50}")

    subtitles = transcribe_video(args.title, args.model)
    print(f"\n  Готово! {len(subtitles)} реплик.")
    print(f'  Следующий шаг: python indexer\\build_scenes.py --title "{args.title}"')


if __name__ == "__main__":
    main()