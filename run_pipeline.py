"""
run_pipeline.py
---------------
Запускает весь пайплайн для одного фильма.

Использование:
    python run_pipeline.py --file "C:/фильмы/аватар.mp4" --title "Аватар 3"
    python run_pipeline.py --file "D:/movies/amelie.mkv" --title "Амели"

Флаги для пропуска шагов (если уже выполнены):
    --skip-describe     пропустить описание кадров LLaVA
    --skip-transcribe   пропустить транскрипцию Whisper
"""

import argparse
import subprocess
import sys
from pathlib import Path

INDEXER = Path(__file__).parent / "indexer"


def run(script: str, args: list):
    cmd = [sys.executable, str(INDEXER / script)] + args
    print(f"\n▶ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[error] Шаг завершился с ошибкой.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Путь к mp4/mkv/avi файлу")
    parser.add_argument("--title", required=True, help="Название фильма")
    parser.add_argument("--interval", type=int, default=15,
                        help="Секунд между кадрами (default: 15)")
    parser.add_argument("--skip-describe", action="store_true")
    parser.add_argument("--skip-transcribe", action="store_true")
    parser.add_argument("--whisper-model", default="small",
                        help="tiny/small/medium/large-v2")
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"[error] Файл не найден: {args.file}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Video RAG Pipeline: {args.title}")
    print(f"{'='*60}")

    # Шаг 1: кадры
    run("extract_frames.py", ["--file", args.file, "--title", args.title,
                               "--interval", str(args.interval)])

    # Шаг 2: описания LLaVA
    if not args.skip_describe:
        run("describe_frames.py", ["--title", args.title])

    # Шаг 3: субтитры Whisper
    if not args.skip_transcribe:
        run("transcribe.py", ["--title", args.title, "--model", args.whisper_model])

    # Шаг 4: сборка сцен + ChromaDB
    run("build_scenes.py", ["--title", args.title])

    print(f"\n{'='*60}")
    print(f"  ✓ Готово: {args.title}")
    print(f"{'='*60}")
    print(f"\n  Запусти API:")
    print(f"  cd api && uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()