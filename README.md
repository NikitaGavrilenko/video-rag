# Video RAG — поиск сцен в видео по текстовому описанию

Решение для соревнования [MultiLingual Video Fragment Retrieval Challenge](https://www.kaggle.com/competitions/multi-lingual-video-fragment-retrieval-challenge), Master Hackathon ML 2026, кейс Okko/Sber.

**Метрика:** Composite Recall Score = avg(SuccessRate@{1,3,5}, VideoRecall@{1,3,5}), IoU ≥ 0.5

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
      ├── ffmpeg: вырезаем кадр из середины сцены
      └── Whisper large-v3: транскрибируем речь в текст
  → Qwen3-VL-8B: смотрит на кадр + читает транскрипт
                 → текстовое описание сцены на английском
  → BGE-M3: кодируем описание в вектор (dense 1024d + sparse)
  → сохраняем в FAISS индекс + BM25 индекс
```

**Запуск:**
```bash
python -m kaggle.pipeline.run_pipeline
```

---

### Этап 2 — Инференс

Берём тестовый датасет с запросами и для каждого запроса находим 5 наиболее подходящих сцен.

```
Запрос пользователя
  → Препроцессинг:
      - исправление опечаток и случайных заглавных букв (SymSpell + SAGE T5)
      - нормализация смешанных раскладок (латиница внутри кириллицы и наоборот)
      - транслитерация арабских/китайских символов
      - [TODO] перевод на второй язык (RU→EN или EN→RU),
               чтобы искать одновременно в русско- и англоязычных описаниях
  → Поиск (параллельно по 6 каналам):
      ├── FAISS dense — поиск по векторному сходству (сцены)
      ├── FAISS dense — то же по событиям (группы из 5 сцен)
      ├── BM25 — полнотекстовый поиск по ASR транскриптам (сцены)
      ├── BM25 — то же по событиям
      ├── sparse dot — поиск по разреженным BGE-M3 весам (сцены)
      └── sparse dot — то же по событиям
  → RRF: объединяем результаты всех 6 каналов → топ-50
  → bge-reranker-v2-m3: переранжируем топ-50 → финальные топ-5
  → submission.csv
```

**Запуск:** открыть `kaggle/6_submit.ipynb`, подключить датасет с индексами, Run All.

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
│   ├── pipeline/                # предобработка видео
│   │   ├── run_pipeline.py      # точка входа
│   │   ├── step1_shots.py       # нарезка на сцены (TransNetV2)
│   │   ├── step2_extract.py     # keyframe + ASR (параллельно)
│   │   ├── step3_vlm_caption.py # описание сцены (Qwen3-VL-8B)
│   │   ├── step4_scene_docs.py  # сборка scenes.jsonl
│   │   ├── step5_event_docs.py  # группировка в events.jsonl
│   │   ├── step6_index.py       # BGE-M3 → FAISS + BM25
│   │   ├── search.py            # поиск + reranker
│   │   └── config.py            # пути и константы
│   └── 6_submit.ipynb           # индексы → submission.csv
│
├── demo/
│   └── api.py                   # FastAPI для демо-стенда
│
├── business/
│   └── frontend/
│       └── okko-demo.html       # UI в стиле Okko
│
├── preproc/
│   └── query_preprocessor.py   # препроцессинг запросов
│
└── setup_h100.sh                # настройка окружения
```

---

## Воспроизведение результата

**1. Запустить предобработку:**
```bash
python -m kaggle.pipeline.run_pipeline
```

**2. Упаковать индексы:**
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
Загрузить как Kaggle Dataset с именем `video-rag-indexes`.

**3. Сабмит:**
`kaggle/6_submit.ipynb` → Add Data → `video-rag-indexes` → Run All → Submit.