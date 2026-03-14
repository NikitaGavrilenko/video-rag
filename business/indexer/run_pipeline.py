"""
run_pipeline.py — запускает весь пайплайн для одного фильма.

Использование:
    python run_pipeline.py --file "data/videos/amelie.mp4" --title "Амели"

Флаги для пропуска шагов:
    --skip-describe    пропустить Groq описания (если уже есть)
    --skip-transcribe  пропустить Whisper (если уже есть)
    --skip-clip        пропустить CLIP embeddings (если уже есть)
"""

import argparse
import subprocess
import sys
from pathlib import Path

INDEXER = Path(__file__).parent / "indexer"


def run(script: str, args: list):
    cmd = [sys.executable, str(INDEXER / script)] + args
    print(f"\n▶  {' '.join(str(a) for a in cmd)}\n{'─'*60}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"\n[error] шаг завершился с ошибкой: {script}")
        sys.exit(r.returncode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--interval", type=int, default=15)
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--skip-describe", action="store_true")
    p.add_argument("--skip-transcribe", action="store_true")
    p.add_argument("--skip-clip", action="store_true")
    args = p.parse_args()

    if not Path(args.file).exists():
        print(f"[error] файл не найден: {args.file}")
        sys.exit(1)

    print(f"\n{'='*60}\n  Video RAG Pipeline: {args.title}\n{'='*60}")

    run("extract_frames.py",  ["--file", args.file, "--title", args.title, "--interval", str(args.interval)])

    if not args.skip_describe:
        run("describe_frames.py", ["--title", args.title])

    if not args.skip_transcribe:
        run("transcribe.py", ["--title", args.title, "--model", args.whisper_model])

    if not args.skip_clip:
        run("embed_clip.py", ["--title", args.title])

    run("build_scenes.py", ["--title", args.title])

    print(f"\n{'='*60}\n  ✓ Готово: {args.title}\n{'='*60}")
    print("\n  Запусти API:")
    print("  cd api && uvicorn main:app --reload --port 8000\n")


if __name__ == "__main__":
    main()
