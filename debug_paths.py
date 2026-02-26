"""
debug_paths.py — запусти это и покажи вывод
"""
import sys
import os
from pathlib import Path

print("=== ДИАГНОСТИКА ПУТЕЙ ===\n")
print(f"Python: {sys.executable}")
print(f"Текущая папка: {os.getcwd()}")
print(f"Этот файл: {Path(__file__).resolve()}")
print(f"Родитель файла: {Path(__file__).parent.resolve()}")

# Проверяем где ищем видео
ROOT = Path(__file__).parent  # debug_paths.py лежит в корне проекта
print(f"\nROOT = {ROOT.resolve()}")
print(f"data/frames = {(ROOT / 'data' / 'frames').resolve()}")

# Проверяем содержимое папки
print(f"\n=== Содержимое текущей папки ===")
for f in sorted(Path(os.getcwd()).iterdir()):
    print(f"  {'[папка]' if f.is_dir() else '[файл] '} {f.name}")

# Если передали аргумент — проверяем этот путь
if len(sys.argv) > 1:
    test_path = Path(sys.argv[1])
    print(f"\n=== Проверяем путь: {sys.argv[1]} ===")
    print(f"  Абсолютный: {test_path.resolve()}")
    print(f"  Существует: {test_path.exists()}")
    print(f"  Это файл:   {test_path.is_file()}")