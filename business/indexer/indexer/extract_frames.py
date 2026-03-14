"""
extract_frames.py
-----------------
Шаг 1: Принимает локальный mp4/mkv/avi файл и нарезает кадры через FFmpeg.

Использование:
    python indexer\extract_frames.py --file "data\videos\Аватар.mp4" --title "Аватар 3"
"""

import argparse
import json
import subprocess
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent  # indexer/ -> project root
FRAMES_DIR = ROOT / "data" / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INTERVAL_SEC = 15


def find_ffmpeg() -> str:
    """Ищет ffmpeg в PATH и в типичных папках установки на Windows."""
    # Сначала ищем в PATH
    found = shutil.which("ffmpeg")
    if found:
        return found

    # Типичные места установки ffmpeg на Windows
    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        str(Path.home() / "ffmpeg" / "bin" / "ffmpeg.exe"),
        str(Path.home() / "ffmpeg" / "ffmpeg.exe"),
    ]
    for path in candidates:
        if Path(path).exists():
            return path

    # Не нашли
    print("\n  [error] ffmpeg не найден!")
    print("  Скачай ffmpeg: https://www.gyan.dev/ffmpeg/builds/")
    print("  Выбери 'ffmpeg-release-essentials.zip', распакуй в C:\\ffmpeg\\")
    print("  Папка должна выглядеть так: C:\\ffmpeg\\bin\\ffmpeg.exe")
    sys.exit(1)


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


def extract_frames(video_path: Path, title: str, interval_sec: int = DEFAULT_INTERVAL_SEC) -> dict:
    slug = slugify(title)
    movie_dir = FRAMES_DIR / slug
    movie_dir.mkdir(parents=True, exist_ok=True)

    existing = list(movie_dir.glob("frame_*.jpg"))
    meta_path = movie_dir / "metadata.json"
    if existing and meta_path.exists():
        print(f"  [skip] кадры уже есть: {len(existing)} шт.")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    ffmpeg = find_ffmpeg()
    print(f"  [ffmpeg] {ffmpeg}")
    print(f"  [ffmpeg] нарезаем кадры каждые {interval_sec} сек ...")

    cmd = [
        ffmpeg,
        "-i", str(video_path),
        "-vf", f"fps=1/{interval_sec}",
        "-q:v", "2",
        "-vsync", "vfr",
        str(movie_dir / "frame_%05d.jpg"),
        "-y",
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [error] FFmpeg вернул ошибку:\n{result.stderr}")
        sys.exit(1)

    frames = sorted(movie_dir.glob("frame_*.jpg"))
    if not frames:
        print("  [error] FFmpeg не создал ни одного кадра. Проверь что файл не повреждён.")
        sys.exit(1)

    metadata = {}
    for frame in frames:
        num = int(frame.stem.split("_")[1])
        t = (num - 1) * interval_sec
        metadata[frame.name] = {
            "timecode_seconds": t,
            "timecode_str": seconds_to_timecode(t),
            "frame_path": str(frame),
            "movie_title": title,
            "movie_slug": slug,
        }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"  [ok] {len(frames)} кадров → data/frames/{slug}/")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Путь к mp4/mkv/avi файлу")
    parser.add_argument("--title", required=True, help="Название фильма")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC,
                        help=f"Секунд между кадрами (default: {DEFAULT_INTERVAL_SEC})")
    args = parser.parse_args()

    # Превращаем в абсолютный путь
    video_path = Path(args.file).resolve()
    if not video_path.exists():
        print(f"\n  [error] Файл не найден: {video_path}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Обрабатываем: {args.title}")
    print(f"  Файл: {video_path}")
    print(f"{'='*50}")

    metadata = extract_frames(video_path, args.title, args.interval)

    print(f"\n  Готово! {len(metadata)} кадров.")
    print(f'  Следующий шаг: python indexer\\describe_frames.py --title "{args.title}"')


if __name__ == "__main__":
    main()