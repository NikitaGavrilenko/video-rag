# Video RAG — поиск видеофрагментов по текстовому описанию

Решение для [MultiLingual Video Fragment Retrieval Challenge](https://www.kaggle.com/competitions/multi-lingual-video-fragment-retrieval-challenge) (Master Hackathon ML 2026, кейс Okko/Sber).

**Метрика:** Composite Recall Score = avg(SuccessRate@{1,3,5}, VideoRecall@{1,3,5}), IoU ≥ 0.5

---

## Архитектура

### Предобработка видео (один раз, H100)

```
Видео
  → step1: TransNetV2 — нарезка на сцены по смене кадра
  → step2: параллельно:
      ├── ffmpeg (CPU, 16 потоков) — кадр из середины сцены
      └── faster-whisper large-v3-turbo (GPU) — транскрипция речи
              или transcripts.pkl (предвычисленные)
  → step3: Qwen3-VL-8B (vLLM, BF16) — multimodal caption: кадр + ASR → EN описание
  → step4: сборка scenes.jsonl (caption + ASR + train augmentation)
  → step5: sliding window (w=110s, step=10s) → events.jsonl
  → step6: BGE-M3 (dense 1024d + sparse + ColBERT) → FAISS CPU IndexFlatIP
  → step6b: LoRA fine-tune bge-reranker-v2-m3 на train данных (опционально)
```

### Инференс (на тестовых запросах)

```
Запрос
  → препроцессинг: нормализация + SymSpell (EN) + transliteration
  → batch encode: BGE-M3 кодирует все запросы разом
  → 10 каналов поиска × 2 языка (RU corrected + EN translated):
      ├── FAISS dense — сцены
      ├── FAISS dense — события
      ├── sparse dot  — сцены
      ├── sparse dot  — события
      └── ColBERT MaxSim — сцены
  → train→test matching: cosine > 0.92 → inject ground-truth с boost +5.0
  → RRF fusion → dedup (IoU > 0.3) → топ-50 кандидатов
  → bge-reranker-v2-m3 (language-aware): max(score_ru, score_en)
  → расширение таймкодов ±55s (cap 180s) → video clustering → топ-5
  → submission.csv
```

---

## Структура репозитория

```
video-rag/
├── kaggle/
│   ├── pipeline/
│   │   ├── config.py               # пути, константы, auto-detect local/server
│   │   ├── run_pipeline.py         # оркестратор
│   │   ├── step1_shots.py          # TransNetV2 shot detection
│   │   ├── step2_extract.py        # keyframe + ASR (параллельно)
│   │   ├── step2_3_stream.py       # streaming pipeline (steps 2+3)
│   │   ├── step3_vlm_caption.py    # Qwen3-VL-8B caption
│   │   ├── step4_scene_docs.py     # сборка scenes.jsonl
│   │   ├── step5_event_docs.py     # сборка events.jsonl
│   │   ├── step6_index.py          # BGE-M3 → FAISS + train index
│   │   ├── step6b_finetune_reranker.py  # LoRA fine-tune reranker
│   │   ├── search.py               # поиск + submission
│   │   ├── retry_failed.py         # повтор упавших сцен
│   │   └── import_transcripts.py   # merge teammate ASR
│   └── submit.ipynb                # загрузка индексов → submission.csv
├── preproc/
│   └── query_preprocessor.py       # нормализация + SymSpell + SAGE
├── .gitignore
├── requirements.txt
├── setup_h100.sh                   # настройка окружения на сервере
└── README.md
```

---

## Воспроизведение результата

### 1. Настройка сервера

```bash
bash setup_h100.sh
cd /kaggle/working/video-rag
```

### 2. Предобработка видео (H100, ~несколько часов)

```bash
# Полный pipeline
python -m kaggle.pipeline.run_pipeline

# Или streaming-режим (шаги 2+3 параллельно, быстрее)
python -m kaggle.pipeline.run_pipeline --stream

# Пересборка документов и индексов без повторного запуска VLM/Whisper
python -m kaggle.pipeline.step4_scene_docs
python -m kaggle.pipeline.step5_event_docs
python -m kaggle.pipeline.step6_index
python -m kaggle.pipeline.step6b_finetune_reranker  # опционально, ~10 мин
```

### 3. Генерация submission.csv

```bash
python -m kaggle.pipeline.search           # с кэшем retrieval
python -m kaggle.pipeline.search --no-cache  # пересчитать retrieval
```

### 4. Kaggle Notebook

Упаковать индексы, загрузить как Dataset, запустить `kaggle/submit.ipynb`.

```bash
tar czf indexes.tar.gz /kaggle/working/*.index /kaggle/working/*.pkl \
    /kaggle/working/scenes.jsonl /kaggle/working/events.jsonl
```

---

## Переменные окружения

| Переменная | Назначение |
|---|---|
| `PROXY_API_KEY` | ProxyAPI (Gemini) — не требуется для базового pipeline |

---

## Ключевые технические детали

- **BGE-M3:** dense 1024d + sparse lexical + ColBERT per-token (fp16). FAISS CPU `IndexFlatIP`
- **Sparse поиск:** dot-product по `lexical_weights` без внешних сервисов
- **ColBERT:** MaxSim re-scoring FAISS-кандидатов
- **RRF uid:** `{video_id}_{start}_{end}` — один документ из нескольких каналов аккумулирует score
- **Train→test:** FAISS cosine ≥ 0.92 → inject GT с reranker boost +5.0; ≥ 0.80 → в пул кандидатов
- **Таймкоды:** ±55s от центра сцены (окно 110s, cap 180s)
- **Video clustering:** топ-2 видео по сумме score заполняют топ-5 слотов