Понял, читаю контекст чата. Вот новый README:

```markdown
# Video RAG — Семантический поиск по видеофрагментам

Система для поиска точных временных отрезков в видео по текстовому запросу.
Разработана в рамках Master Hackathon ML 2026, кейс Okko/Sber.

## Соревнование

Kaggle: MultiLingual Video Fragment Retrieval Challenge
Метрика: Composite Recall Score (AvgSR + AvgVR) — IoU ≥ 0.5

## Архитектура

### Офлайн (предобработка на сервере)
1. **TransNetV2** — детекция границ шотов по видеоряду
2. **Whisper Large-v3** — транскрипция аудио с точными таймкодами
3. **Shot-aligned сегментация** — границы шота = границы чанка
4. **e5-large** → FAISS index (семантический поиск)
5. **BM25** index (точные цитаты и ключевые слова)
6. **BGE-M3 reranker** — cross-encoder для финального ранжирования

### Инференс
```
Запрос → Query expansion (LLM) → FAISS + BM25 параллельно
→ Merge (RRF) → BGE-M3 rerank top-20 → Top-5 сегментов
```

### Две версии системы

| | Kaggle | Бизнес (Okko) |
|---|---|---|
| Цель | Максимальная метрика | Инференс < 1 сек |
| Транскрипция | Whisper Large-v3 | faster-whisper |
| Reranker | BGE-M3 cross-encoder | нет |
| Query expansion | LLM (Groq) | нет |
| Индекс | FAISS + BM25 | FAISS + BM25 |

## Структура репозитория

```
video-rag/
├── kaggle/
│   ├── 1_transnetv2_shots.ipynb     # границы шотов из видео
│   ├── 2_whisper_transcribe.ipynb   # транскрипция Whisper Large-v3
│   ├── 3_describe_scenes.ipynb      # описания сцен через LLM (опц.)
│   ├── 4_build_index.ipynb          # e5-large + BM25 → FAISS
│   └── 5_inference.ipynb            # поиск + submission.csv
│
├── business/
│   ├── indexer/
│   │   ├── extract_frames.py        # FFmpeg, кадры каждые 15 сек
│   │   ├── transcribe.py            # faster-whisper
│   │   ├── build_index.py           # e5 + BM25 → FAISS
│   │   └── run_pipeline.py          # запуск всего пайплайна
│   ├── api/
│   │   └── main.py                  # FastAPI: /search, /search/image, /qa
│   ├── frontend/
│   │   └── okko-demo.html           # UI в стиле Okko
│   └── requirements.txt
│
├── preproc/                         # Предобработка пользовательских данных
│
└── data/
    └── shot_boundaries_schema.json  # Контракт формата между участниками
```

## Формат данных между участниками

Файл `shot_boundaries.json` — передаётся через Google Drive:

```json
[
  {
    "video_id": "video_abc123",
    "audio_key": "videos/video_abc123.mp4",
    "scenes": [
      {
        "scene_idx": 0,
        "start": 0.0,
        "end": 18.4,
        "description": null,
        "description_en": null,
        "keyframe_time": 9.2,
        "keyframe_path": null
      }
    ]
  }
]
```

## Разделение задач

| Участник | Задача |
|---|---|
|  | TransNetV2 — границы шотов |
| ? | Whisper Large-v3 — транскрипция |
| ? | Описания сцен через LLM (Groq Vision) |
| ? | Индексация + инференс + сабмит |

## Установка (бизнес-версия)

```bash
pip install -r business/requirements.txt
```

Создай `.env`:
```
GROQ_API_KEY=ваш_ключ_с_console.groq.com
```

Запуск пайплайна:
```bash
python business/indexer/run_pipeline.py --file "data/videos/film.mp4" --title "Название"
```

Запуск API:
```bash
cd business/api && uvicorn main:app --reload --port 8000
```

## Ключевые решения

- **Сегменты размером ~3-9 сек** — соответствует ground truth, IoU ≥ 0.5
- **Shot-aligned chunking** — TransNetV2 даёт семантически чистые границы
- **Два индекса** — FAISS для смысла, BM25 для точных цитат
- **Запросы RU/EN + опечатки** — multilingual-e5 устойчив к этому
- **Внешние API только для запросов** — видео не покидают сервер (требование соревнования)

## .gitignore

```
data/videos/
data/audio/
*.pkl
shot_boundaries.json
__pycache__/
.venv/
.env
```
