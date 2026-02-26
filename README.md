# Video RAG — Поиск фильмов по сцене (для Okko)

## Установка

```bash
pip install -r requirements.txt
ollama pull llava        # ~4GB, один раз
```

## Быстрый старт

### 1. Проиндексировать фильм

```bash
# Из YouTube (трейлер или полный фильм)
python run_pipeline.py --url "https://youtu.be/XXXX" --title "Амели"

# Из локального файла
python run_pipeline.py --file /path/to/amelie.mp4 --title "Амели"
```

Пайплайн по шагам:
1. `extract_frames.py` — скачивает видео, нарезает кадры (1 / 15 сек)
2. `describe_frames.py` — LLaVA описывает каждый кадр на русском
3. `transcribe.py`      — Whisper транскрибирует аудио
4. `build_scenes.py`    — объединяет всё, создаёт эмбеддинги → ChromaDB

### 2. Запустить API

```bash
cd api
uvicorn main:app --reload --port 8000
```

### 3. Открыть демо

Открыть `frontend/okko-demo.html` в браузере.

## Структура данных

```
data/
├── videos/
│   └── амели.mp4
├── frames/
│   └── амели/
│       ├── frame_00001.jpg
│       ├── metadata.json      # таймкоды кадров
│       ├── descriptions.json  # описания от LLaVA
│       └── subtitles.json     # субтитры от Whisper
└── chroma_db/                 # векторная БД
```

## API

```
POST /search
{ "query": "герой стоит под дождём", "top_k": 5 }

GET  /movies   — список проиндексированных фильмов
GET  /health   — статус и количество сцен в БД
```

## Советы для хакатона

- Индексируй заранее, LLaVA ~8 сек/кадр на CPU
- Для 10 трейлеров (5 мин каждый) ≈ 200 кадров ≈ 30 мин
- Если нет GPU — используй `--whisper-model tiny` для скорости
- `--skip-describe` если хочешь проверить поиск только по субтитрам