# Video RAG — семантический поиск видеофрагментов

**Команда МаLышки** · Master Hackathon ML 2026 · кейс Okko/Sber  
[MultiLingual Video Fragment Retrieval Challenge](https://www.kaggle.com/competitions/multi-lingual-video-fragment-retrieval-challenge)

---

## Задача

Онлайн-кинотеатр Okko содержит тысячи фильмов и сериалов. Пользователи хотят находить не просто фильм, а конкретный момент внутри него: «где герой произносит эту фразу», «покажи сцену с погоней», «найди эпизод с этим фактом». Современный поиск по метаданным с этим не справляется.

Мы построили систему, которая по текстовому запросу на русском или английском находит точные временные отрезки в видео — с точностью до секунды.

**Метрика:** `Composite Recall Score = avg(SuccessRate@{1,3,5}, VideoRecall@{1,3,5})`, IoU ≥ 0.5  
**Инференс:** < 1 секунды на запрос

---

## Как работает система

### Этап 1 — Офлайн-предобработка (один раз на корпус)

```
Видеокорпус
  │
  ├─ [шаг 1]  TransNetV2          → границы сцен по каждому видео
  │
  ├─ [шаг 2]  ffmpeg (CPU)        → ключевой кадр из середины сцены
  │            faster-whisper      → транскрипция речи (large-v3-turbo)
  │              └─ или transcripts.pkl (предвычисленные, пропускает Whisper)
  │
  ├─ [шаг 3]  Qwen3-VL-8B (vLLM) → мультимодальный caption: кадр + ASR → EN описание
  │
  ├─ [шаг 4]  scenes.jsonl        ← caption + ASR + аугментация train-запросами
  │
  ├─ [шаг 5]  events.jsonl        ← скользящее окно (w=110s, step=10s)
  │                                  для крупнозернистого поиска
  │
  └─ [шаг 6]  BGE-M3              → dense (1024d) + sparse (lexical) + ColBERT
               FAISS IndexFlatIP  → индекс сцен + индекс событий + индекс train-запросов
               [шаг 6b] LoRA fine-tune bge-reranker-v2-m3 на train-триплетах (опц.)
```

### Этап 2 — Онлайн-инференс (на каждый запрос)

```
Запрос: "герой плачет под дождём"
  │
  ├─ Препроцессинг   нормализация + SymSpell (EN) + транслитерация
  ├─ Переводы        corrected RU + translated EN (из translated_data.csv)
  ├─ Кодирование     BGE-M3 batch encode всех запросов (dense + sparse)
  │
  ├─ Поиск  ×10 каналов (2 языка × 5 типов индексов):
  │   ├─ FAISS dense    — сцены   (RU + EN)
  │   ├─ FAISS dense    — события (RU + EN)
  │   ├─ sparse dot     — сцены   (RU + EN)
  │   ├─ sparse dot     — события (RU + EN)
  │   └─ ColBERT MaxSim — сцены   (RU + EN)
  │
  ├─ Train→test   cosine ≥ 0.92 → inject GT-ответ (boost +5.0 к reranker score)
  ├─ RRF fusion   объединение всех каналов через Reciprocal Rank Fusion
  ├─ Dedup        IoU > 0.3 → убираем дубли → топ-50 кандидатов
  │
  ├─ Reranker     bge-reranker-v2-m3, language-aware: max(score_ru, score_en)
  ├─ Таймкоды     расширение ±55s от центра сцены (окно 110s, cap 180s)
  └─ Кластеризация предпочитать фрагменты из топ-2 видео → топ-5

  → submission.csv
```

---

## Структура репозитория

```
video-rag/
├── kaggle/
│   ├── pipeline/
│   │   ├── config.py                    # пути и константы, auto-detect local/server
│   │   ├── run_pipeline.py              # оркестратор (точка входа)
│   │   ├── step1_shots.py               # TransNetV2 shot detection
│   │   ├── step2_extract.py             # keyframes + ASR (параллельно)
│   │   ├── step2_3_stream.py            # streaming pipeline (шаги 2+3 слиты)
│   │   ├── step3_vlm_caption.py         # Qwen3-VL-8B captioning
│   │   ├── step4_scene_docs.py          # сборка scenes.jsonl
│   │   ├── step5_event_docs.py          # сборка events.jsonl
│   │   ├── step6_index.py               # BGE-M3 → FAISS + train index
│   │   ├── step6b_finetune_reranker.py  # LoRA fine-tune reranker (опционально)
│   │   ├── search.py                    # retrieval + reranking → submission.csv
│   │   ├── retry_failed.py              # повтор упавших сцен
│   │   └── import_transcripts.py        # merge ASR от тиммейтов
│   └── submit.ipynb                     # Kaggle notebook: индексы → сабмит
├── preproc/
│   └── query_preprocessor.py            # нормализация + SymSpell + SAGE + транслитерация
├── setup_h100.sh                        # настройка сервера (запустить один раз)
├── requirements.txt
└── README.md
```

---

## Воспроизведение результата

### 1. Настройка сервера

```bash
bash setup_h100.sh
cd /kaggle/working/video-rag
```

### 2. Офлайн-предобработка (Kaggle, 2×T4, несколько часов)

```bash
# Полный pipeline
python -m kaggle.pipeline.run_pipeline

# Streaming-режим — шаги 2+3 параллельно (быстрее на 2×T4)
python -m kaggle.pipeline.run_pipeline --stream

# С fine-tune reranker (~+10 мин)
python -m kaggle.pipeline.run_pipeline --stream --finetune
```

Флаги для частичного перезапуска:

```bash
python -m kaggle.pipeline.run_pipeline --skip-shots --skip-extract --skip-vlm
```

### 3. Пересборка индекса без повтора VLM/Whisper

```bash
python -m kaggle.pipeline.step4_scene_docs
python -m kaggle.pipeline.step5_event_docs
python -m kaggle.pipeline.step6_index
```

### 4. Генерация submission.csv

```bash
python -m kaggle.pipeline.search            # с кэшем retrieval
python -m kaggle.pipeline.search --no-cache  # пересчитать retrieval
```

### 5. Сабмит через Kaggle Notebook

```bash
# Упаковать индексы на сервере
tar czf indexes.tar.gz \
    /kaggle/working/*.index \
    /kaggle/working/*.pkl \
    /kaggle/working/scenes.jsonl \
    /kaggle/working/events.jsonl
# Загрузить как Kaggle Dataset → подключить к ноутбуку → запустить kaggle/submit.ipynb
```

---

## Технологический стек

| Компонент | Инструмент |
|---|---|
| Детекция сцен | TransNetV2 |
| Извлечение кадров | ffmpeg (CPU, 16 потоков) |
| Распознавание речи | faster-whisper `large-v3-turbo` |
| Визуальный caption | Qwen3-VL-8B via vLLM (BF16) |
| Эмбеддинги | BGE-M3 — dense 1024d + sparse + ColBERT |
| Векторный поиск | FAISS `IndexFlatIP` (CPU) |
| Реранкер | bge-reranker-v2-m3 (+ LoRA fine-tune) |
| Препроцессинг запросов | SymSpell (EN) + SAGE T5 (RU, опц.) |
| GPU | Kaggle 2×T4 (Tesla T4, 15 GB каждая) |

---

## Важно: ограничения на данные

Видео и аудио из датасета соревнования нельзя передавать за пределы сервера. Все компоненты pipeline работают локально:

- `ffmpeg`, `faster-whisper`, `Qwen3-VL-8B` — запускаются на сервере, видео не покидает машину
- `BGE-M3`, `bge-reranker-v2-m3` — локальный инференс, нет внешних API-вызовов
- Поиск на инференсе — только текстовые запросы, видео не участвуют

---

## Технические детали

<details>
<summary>BGE-M3: кодирование</summary>

```python
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
output = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=True)
# output['dense_vecs']      — np.ndarray (N, 1024)
# output['lexical_weights'] — list of {int_token_id: float}
# output['colbert_vecs']    — list of np.ndarray (n_tokens, 1024), fp16
```
</details>

<details>
<summary>Sparse-поиск без внешних сервисов</summary>

```python
score = sum(query_sparse.get(k, 0.0) * v for k, v in doc_sparse.items())
```
</details>

<details>
<summary>RRF fusion</summary>

UID = `{video_id}_{start}_{end}` — один фрагмент из нескольких каналов **аккумулирует** score:
```python
score[uid] += 1.0 / (RRF_K + rank)
```
</details>

<details>
<summary>Train→test matching</summary>

- cosine ≥ 0.92 → inject GT-ответ, reranker boost **+5.0**
- cosine ≥ 0.80 → добавить в пул кандидатов (конкурирует наравне)
</details>

<details>
<summary>Расширение таймкодов</summary>

Короткие сцены расширяются до **±55s** от центра (окно 110s). Жёсткий cap: 180s.  
Train-матчи сохраняют оригинальные GT-таймкоды без изменений.
</details>