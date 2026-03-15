# Video RAG — поиск сцен в видео по текстовому описанию

Решение для соревнования [MultiLingual Video Fragment Retrieval Challenge](https://www.kaggle.com/competitions/multi-lingual-video-fragment-retrieval-challenge), Master Hackathon ML 2026, кейс Okko/Sber.

**Метрика:** Composite Recall Score = avg(SuccessRate@{1,3,5}, VideoRecall@{1,3,5}), IoU >= 0.5

---

## Как работает система

Два этапа: предобработка видео (один раз) и инференс (для каждого запроса).

---

### Этап 1 — Предобработка видео

Берём каждое видео и превращаем его в базу данных сцен.

```
Видео
  → TransNetV2: нарезаем на сцены по смене кадра
  → для каждой сцены параллельно:
      ├── ffmpeg (CPU, 16 потоков): вырезаем кадр из середины сцены
      └── Whisper large-v3-turbo / transcripts.pkl: транскрибируем речь
  → Qwen3-VL-8B (vLLM): смотрит на кадр + читает транскрипт
                        → текстовое описание сцены на английском
  → Gemini Flash Lite (API, без GPU):
      ├── step3b: улучшение описаний (EN + RU, query-friendly)
      └── step3c: генерация 4 синтетических поисковых запросов (doc2query)
  → BGE-M3: кодируем описание в вектор (dense 1024d + sparse + ColBERT)
  → сохраняем в FAISS индекс (CPU) + ColBERT vectors (fp16)
  → (опционально) LoRA fine-tune bge-reranker-v2-m3 на train данных
```

**Запуск:**
```bash
# Полный pipeline (H100)
python -m kaggle.pipeline.run_pipeline

# LLM пост-обработка (без GPU)
PROXY_API_KEY=... python -m kaggle.pipeline.step3b_fix_captions --workers 20
PROXY_API_KEY=... python -m kaggle.pipeline.step3c_doc2query --workers 10

# Пересборка + индексация (GPU для step6+)
python -m kaggle.pipeline.step4_scene_docs
python -m kaggle.pipeline.step5_event_docs
python -m kaggle.pipeline.step6_index
python -m kaggle.pipeline.step6b_finetune_reranker  # опционально, ~10 мин
```

---

### Этап 2 — Инференс

Берём тестовый датасет с запросами и для каждого запроса находим 5 наиболее подходящих сцен.

```
Запрос пользователя
  → Препроцессинг:
      - исправление опечаток (SymSpell + SAGE T5)
      - перевод: corrected_question (RU) + translated_question (EN)
        из translated_data.csv
  → Batch encode: BGE-M3 кодирует все запросы разом (GPU, ~10 сек)
  → Поиск (10 каналов × 2 языка):
      ├── FAISS dense — сцены (RU + EN)
      ├── FAISS dense — события (RU + EN)
      ├── sparse dot — сцены (RU + EN)
      ├── sparse dot — события (RU + EN)
      └── ColBERT MaxSim — сцены (RU + EN)
  → Train→test matching: если test запрос похож на train (cosine > 0.92),
    инжектим ground-truth ответ из train
  → RRF: объединяем все каналы (cross-channel fusion)
  → Dedup по IoU > 0.3 → топ-50 кандидатов
  → bge-reranker-v2-m3 (language-aware):
      - RU запрос × RU caption + EN запрос × EN caption
      - берём max(score_ru, score_en)
      - train matches получают +2.0 к score
  → Расширение таймкодов: ±55с от центра (110с окно, cap 180с)
  → Video clustering: топ-2 видео по сумме score заполняют топ-5
  → submission.csv
```

**Запуск:**
```bash
python -m kaggle.pipeline.search           # с кэшем retrieval
python -m kaggle.pipeline.search --no-cache # пересчитать retrieval
```

---

## Демо

Помимо Kaggle-сабмита, систему можно попробовать вживую на сайте — введи текстовое описание сцены и получи топ-5 результатов с превью кадра и таймкодом.

Фронтенд: `business/frontend/okko-demo.html`
API: `demo/api.py` (FastAPI, те же индексы)

```bash
pip install -r demo/requirements.txt
cd demo && uvicorn api:app --port 8000
# открыть business/frontend/okko-demo.html в браузере
```

---

## Структура репозитория

```
video-rag/
├── kaggle/
│   ├── pipeline/                    # предобработка видео
│   │   ├── run_pipeline.py          # точка входа
│   │   ├── config.py                # пути и константы (auto-detect local/server)
│   │   ├── step1_shots.py           # нарезка на сцены (TransNetV2)
│   │   ├── step2_extract.py         # keyframe + ASR (параллельно)
│   │   ├── step2_3_stream.py        # streaming pipeline (steps 2+3)
│   │   ├── step3_vlm_caption.py     # описание сцены (Qwen3-VL-8B)
│   │   ├── step3b_fix_captions.py   # LLM пост-обработка (Gemini API)
│   │   ├── step3c_doc2query.py      # синтетические запросы (Gemini API)
│   │   ├── step4_scene_docs.py      # сборка scenes.jsonl
│   │   ├── step5_event_docs.py      # группировка в events.jsonl
│   │   ├── step6_index.py           # BGE-M3 → FAISS + train index
│   │   ├── step6b_finetune_reranker.py  # LoRA fine-tune reranker
│   │   ├── search.py                # поиск + batch reranker
│   │   └── retry_failed.py          # пере-обработка пропущенных сцен
│   └── 6_submit.ipynb               # индексы → submission.csv
│
├── demo/
│   └── api.py                       # FastAPI для демо-стенда
│
├── business/
│   └── frontend/
│       └── okko-demo.html           # UI в стиле Okko
│
├── preproc/
│   └── query_preprocessor.py       # препроцессинг запросов
│
└── setup_h100.sh                    # настройка окружения
```

---

## Воспроизведение результата

**1. Запустить предобработку (H100):**
```bash
python -m kaggle.pipeline.run_pipeline
```

**2. LLM пост-обработка (без GPU, нужен PROXY_API_KEY):**
```bash
PROXY_API_KEY=... python -m kaggle.pipeline.step3b_fix_captions --workers 20
PROXY_API_KEY=... python -m kaggle.pipeline.step3c_doc2query --workers 10
```

**3. Пересобрать docs + индексы (H100):**
```bash
python -m kaggle.pipeline.step4_scene_docs
python -m kaggle.pipeline.step5_event_docs
python -m kaggle.pipeline.step6_index
python -m kaggle.pipeline.step6b_finetune_reranker  # опционально
python -m kaggle.pipeline.search
```

**4. Забрать результат:**
```bash
scp ubuntu@server:/kaggle/working/submission.csv .
```
