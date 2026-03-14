# Video RAG — Семантический поиск по видеофрагментам

Система для поиска точных временных отрезков в видео по текстовому запросу.
Разработана в рамках Master Hackathon ML 2026, кейс Okko/Sber.

## Соревнование

Kaggle: MultiLingual Video Fragment Retrieval Challenge
Метрика: Composite Recall Score (AvgSR + AvgVR) — IoU ≥ 0.5

## Архитектура

### Офлайн пайплайн (Kaggle, H100)

```
Video → TransNetV2 (shot detection) → ~20-25K сцен
         ├── ffmpeg keyframes (CPU)       ─┐
         └── faster-whisper large-v3 (GPU) ─┤→ Qwen3-VL-8B (vLLM, мультимодальный caption)
                                            ↓
                                     Scene Documents (JSONL)
                                            ↓
                                     Event Documents (sliding window 5 сцен)
                                            ↓
                                     BGE-M3 (dense + sparse) → FAISS-GPU + BM25
```

### Инференс

```
Запрос → BGE-M3 embed → 4 канала параллельно:
  ├── scenes: BM25 (asr_text) top-50
  ├── scenes: dense vector top-50
  ├── events: BM25 (summary) top-50
  └── events: dense vector top-50
→ RRF merge → dedup (IoU > 0.5) → bge-reranker-v2-m3 (top-100 → top-10) → top-5
```

### Две версии системы

| | Kaggle (pipeline/) | Бизнес (business/) |
|---|---|---|
| Цель | Максимальная метрика | Инференс < 1 сек |
| Транскрипция | Whisper Large-v3 | faster-whisper small |
| VLM | Qwen3-VL-8B (vLLM) | Groq Vision API |
| Embedding | BGE-M3 (dense+sparse) | multilingual-e5 |
| Индекс | FAISS-GPU + BM25 in-memory | ChromaDB |
| Reranker | bge-reranker-v2-m3 | нет |
| Search | 4-channel hybrid + RRF | CLIP + e5 merge |

## Структура репозитория

```
video-rag/
├── kaggle/
│   ├── pipeline/                       # Production pipeline (H100)
│   │   ├── config.py                   # пути, константы, параметры моделей
│   │   ├── step1_shots.py              # TransNetV2 + merge teammate JSON + фильтр микрошотов
│   │   ├── step2_extract.py            # параллельно: ffmpeg keyframes + faster-whisper ASR
│   │   ├── step3_vlm_caption.py        # Qwen3-VL-8B через vLLM (мультимодальный caption)
│   │   ├── step4_scene_docs.py         # сборка scene documents (JSONL)
│   │   ├── step5_event_docs.py         # sliding window events
│   │   ├── step6_index.py              # BGE-M3 embedding → FAISS-GPU + BM25
│   │   ├── search.py                   # hybrid search + reranker → submission.csv
│   │   └── run_pipeline.py             # оркестратор всех шагов
│   ├── 1_transnetv2_shots.ipynb        # legacy notebook (shot detection)
│   ├── 2_whisper_transcribe.ipynb      # legacy notebook (ASR)
│   └── 5_inference.ipynb               # legacy notebook (e5 + BM25 baseline)
│
├── business/
│   ├── api/api/main.py                 # FastAPI: /search, /search/image, /qa
│   ├── indexer/run_pipeline.py         # пайплайн индексации для бизнес-версии
│   ├── frontend/okko-demo.html         # UI в стиле Okko
│   └── requirements.txt
│
├── preproc/
│   └── query_preprocessor.py           # 3-stage query cleaning (rules → SymSpell → SAGE)
│
└── data/
    └── shot_boundaries_schema.json     # контракт формата между участниками
```

## Запуск Kaggle Pipeline (H100)

```bash
# Полный пайплайн
python -m kaggle.pipeline.run_pipeline

# С пропуском шагов (если данные уже есть)
python -m kaggle.pipeline.run_pipeline --skip-shots --skip-extract

# Только поиск (офлайн шаги уже выполнены)
python -m kaggle.pipeline.run_pipeline --search-only
```

### Бюджет GPU (H100, ~21GB / 96GB)

| Модель | VRAM | Время |
|---|---|---|
| faster-whisper large-v3 | ~3GB | ~8-10 мин |
| Qwen3-VL-8B (vLLM, batch 64) | ~16GB | ~25-40 мин |
| BGE-M3 embeddings | ~2GB | ~3-5 мин |

## Запуск бизнес-версии

```bash
pip install -r business/requirements.txt

# Индексация видео
python business/indexer/run_pipeline.py --file "data/videos/film.mp4" --title "Название"

# API сервер
cd business/api && uvicorn api.main:app --reload --port 8000
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
        "keyframe_time": 9.2
      }
    ]
  }
]
```

## Ключевые решения

- **VLM мультимодальный caption** — Qwen3-VL видит кадр + читает ASR одновременно, синтезируя оба сигнала
- **Dual-level retrieval** — scenes (точные фрагменты) + events (sliding window для сюжетных запросов)
- **BGE-M3 hybrid** — dense + sparse vectors в одном encode, FAISS-GPU для dense, in-memory dot-product для sparse
- **Сегменты ~3-9 сек** — соответствует ground truth, IoU ≥ 0.5
- **Shot-aligned chunking** — TransNetV2 даёт семантически чистые границы
- **Внешние API только для запросов** — видео не покидают сервер (требование соревнования)

## Environment Variables

- `GROQ_API_KEY` — Groq API (бизнес-версия: описания сцен)
- `OPENROUTER_API_KEY` — OpenRouter API (бизнес-версия: QA endpoint)
