# Video RAG — Семантический поиск по видеофрагментам

Система поиска точных временных отрезков в видео по текстовому описанию сцены.
Разработана в рамках **Master Hackathon ML 2026**, кейс Okko/Sber.

**Соревнование:** [MultiLingual Video Fragment Retrieval Challenge](https://www.kaggle.com/competitions/multi-lingual-video-fragment-retrieval-challenge)
**Метрика:** Composite Recall Score = avg(SuccessRate@{1,3,5}, VideoRecall@{1,3,5}), IoU ≥ 0.5

---

## Общий пайплайн

```
┌─────────────────────────────────────────────────────────────┐
│                        H100 сервер                          │
│                                                             │
│  Видео → TransNetV2 → ~20-25K сцен                         │
│             ├── ffmpeg keyframes (CPU) ──┐                  │
│             └── Whisper large-v3 (GPU) ──┤→ Qwen3-VL-8B    │
│                                          │   (caption)      │
│                                          ↓                  │
│                                   scenes.jsonl              │
│                                   events.jsonl              │
│                                          ↓                  │
│                              BGE-M3 → FAISS + BM25          │
│                                   (indexes.tar.gz)          │
└─────────────────────────────────────────────────────────────┘
              ↓ scp / Kaggle Dataset
┌─────────────────────────────────────────────────────────────┐
│                   Kaggle ноутбук (6_submit.ipynb)           │
│                                                             │
│  Запрос → препроцессинг → BGE-M3 embed                     │
│           → 6 каналов (FAISS + BM25 + sparse) → RRF        │
│           → bge-reranker-v2-m3 → топ-5 → submission.csv    │
└─────────────────────────────────────────────────────────────┘
              ↓ параллельно
┌─────────────────────────────────────────────────────────────┐
│                    Демо-стенд (demo/)                       │
│                                                             │
│  те же индексы → demo/api.py (FastAPI)                     │
│  → business/frontend/okko-demo.html                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Структура репозитория

```
video-rag/
│
├── kaggle/
│   ├── pipeline/               # Офлайн пайплайн (запускается на H100)
│   │   ├── config.py           # Пути и константы
│   │   ├── step1_shots.py      # TransNetV2 shot detection
│   │   ├── step2_extract.py    # ffmpeg keyframes + Whisper ASR (параллельно)
│   │   ├── step2_3_stream.py   # Streaming вариант: шаги 2+3 слиты в producer-consumer
│   │   ├── step3_vlm_caption.py# Qwen3-VL-8B описание сцены (image + ASR → text)
│   │   ├── step4_scene_docs.py # Сборка scenes.jsonl
│   │   ├── step5_event_docs.py # Sliding window events.jsonl (окно 5 сцен)
│   │   ├── step6_index.py      # BGE-M3 → FAISS-GPU + BM25
│   │   ├── search.py           # Гибридный поиск + reranker → submission.csv
│   │   └── run_pipeline.py     # Оркестратор всех шагов
│   └── 6_submit.ipynb          # Kaggle ноутбук: индексы → submission.csv
│
├── demo/
│   ├── api.py                  # FastAPI сервер для демо (те же индексы)
│   └── requirements.txt
│
├── business/
│   └── frontend/
│       └── okko-demo.html      # UI в стиле Okko (подключается к demo/api.py)
│
├── preproc/
│   ├── query_preprocessor.py   # Препроцессинг запросов (rules → SymSpell → SAGE)
│   └── query_preprocessor_doc.md
│
├── data/
│   └── shot_boundaries_schema.json  # Контракт формата между участниками
│
└── setup_h100.sh               # Скрипт настройки сервера
```

---

## Шаг 1 — Запуск пайплайна на H100

```bash
# Настроить сервер (один раз)
bash setup_h100.sh

# Перейти в репо
cd /kaggle/working/video-rag

# Полный пайплайн (~45 мин на H100)
python -m kaggle.pipeline.run_pipeline

# Или streaming-режим (шаги 2+3 параллельно, быстрее)
python -m kaggle.pipeline.run_pipeline --stream

# Пропустить уже выполненные шаги
python -m kaggle.pipeline.run_pipeline --skip-shots --skip-extract
```

**Что создаётся в `/kaggle/working/`:**

| Файл | Описание |
|---|---|
| `shot_boundaries.json` | Границы сцен по всем видео |
| `extractions.json` | Путь к keyframe + ASR текст на каждую сцену |
| `captions.json` | VLM описание каждой сцены |
| `scenes.jsonl` | Финальные scene documents |
| `events.jsonl` | Event documents (sliding window) |
| `faiss_scenes.index` | Dense FAISS индекс сцен |
| `faiss_events.index` | Dense FAISS индекс событий |
| `scenes_meta.pkl` | Метаданные сцен (parallel to FAISS) |
| `events_meta.pkl` | Метаданные событий |
| `sparse_scenes.pkl` | BGE-M3 sparse vectors сцен |
| `sparse_events.pkl` | BGE-M3 sparse vectors событий |
| `bm25_scenes.pkl` | BM25 индекс по ASR текстам |
| `bm25_events.pkl` | BM25 индекс по event summary |
| `keyframes/` | JPEG кадры сцен |

---

## Шаг 2 — Упаковка и загрузка индексов

На сервере:
```bash
tar czf indexes.tar.gz \
  /kaggle/working/faiss_scenes.index \
  /kaggle/working/faiss_events.index \
  /kaggle/working/scenes_meta.pkl \
  /kaggle/working/events_meta.pkl \
  /kaggle/working/sparse_scenes.pkl \
  /kaggle/working/sparse_events.pkl \
  /kaggle/working/bm25_scenes.pkl \
  /kaggle/working/bm25_events.pkl \
  /kaggle/working/scenes.jsonl \
  /kaggle/working/events.jsonl
```

Загрузить `indexes.tar.gz` как Kaggle Dataset с именем **`video-rag-indexes`**.

Кадры для демо (опционально):
```bash
scp -r ubuntu@<IP>:/kaggle/working/keyframes/ ./data/keyframes/
```

---

## Шаг 3 — Сабмит на Kaggle

В ноутбуке `kaggle/6_submit.ipynb`:
1. Add Data → подключить датасет `video-rag-indexes`
2. Run All
3. Submit `submission.csv`

**Формат submission.csv:**
```
query_id, video_file_1, start_1, end_1, ..., video_file_5, start_5, end_5
```
`video_file` — stem имени файла без расширения, например `video_abc123`.

---

## Шаг 4 — Демо-стенд (локально)

```bash
# Разложить индексы:
# data/indexes/faiss_scenes.index, *.pkl ...
# data/keyframes/{video_id}/scene_XXXX.jpg
# data/scenes.jsonl

pip install -r demo/requirements.txt
cd demo && uvicorn api:app --port 8000

# Открыть в браузере:
# business/frontend/okko-demo.html
```

---

## Поиск: как работает инференс

```
Запрос
  → QueryPreprocessor (rules → SymSpell → transliterate)
  → BGE-M3 encode (dense 1024d + sparse lexical weights)
  → 6 каналов параллельно:
      ├── FAISS scenes  (dense top-50)
      ├── FAISS events  (dense top-50)
      ├── BM25 scenes   (sparse top-50)
      ├── BM25 events   (sparse top-50)
      ├── sparse dot scenes (top-50)
      └── sparse dot events (top-50)
  → RRF merge
  → dedup (IoU > 0.5)
  → bge-reranker-v2-m3 (top-100 → top-10)
  → top-5
```

---

## Модели

| Назначение | Модель |
|---|---|
| Shot detection | TransNetV2 |
| ASR | faster-whisper large-v3 |
| VLM caption | Qwen/Qwen3-VL-8B (vLLM, BF16) |
| Embedding | BAAI/bge-m3 (dense 1024d + sparse) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Query correction RU | ai-forever/sage-fredt5-distilled-95m |

## GPU бюджет (H100, ~21GB VRAM)

| Шаг | Модель | VRAM | Время |
|---|---|---|---|
| 2 | faster-whisper large-v3 | ~3 GB | ~8-10 мин |
| 3 | Qwen3-VL-8B (batch 64) | ~16 GB | ~25-40 мин |
| 6 | BGE-M3 encode | ~2 GB | ~3-5 мин |

---

## Формат обмена данными между участниками

Файл `shot_boundaries.json` (передаётся через Google Drive):

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
        "keyframe_time": 9.2
      }
    ]
  }
]
```

Положить в `/kaggle/working/teammate_shots.json` — пайплайн использует его приоритетно вместо TransNetV2.