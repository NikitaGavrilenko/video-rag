# Video RAG — Поиск по сценам

Три режима поиска:
- **По описанию** — CLIP (мультиязычный) + multilingual-e5, мёрдж результатов
- **По скриншоту** — CLIP image encoder → похожие кадры
- **Вопрос о фильме** — RAG: e5 → находим сцены → Groq llama-4 → текстовый ответ

## Установка

```bash
pip install -r requirements.txt
```

Создай `.env`:
```
GROQ_API_KEY=ваш_ключ_с_console.groq.com
```

## Пайплайн (один раз на фильм)

```bash
python run_pipeline.py --file "data/videos/amelie.mp4" --title "Амели"
```

Шаги:
1. `extract_frames.py`  — кадры через FFmpeg (каждые 15 сек)
2. `describe_frames.py` — описания через Groq Vision (llama-4)
3. `transcribe.py`      — субтитры через faster-whisper
4. `embed_clip.py`      — CLIP image embeddings → коллекция `clip_visual`
5. `build_scenes.py`    — e5 text embeddings → коллекции `text_visual`, `text_subtitles`

Флаги для пропуска уже выполненных шагов:
```bash
python run_pipeline.py --file "..." --title "..." --skip-describe --skip-transcribe
```

## Запуск API

```bash
cd api && uvicorn main:app --reload --port 8000
```

## Фронтенд

Открыть `frontend/okko-demo.html` в браузере.

## API эндпоинты

```
POST /search         { query, top_k, movie_filter? }
POST /search/image   multipart/form-data: file + top_k
POST /qa             { question, movie_filter? }
GET  /movies
GET  /health
GET  /video/{slug}
```

## Структура данных

```
data/
├── videos/
│   └── амели.mp4
├── frames/
│   └── амели/
│       ├── frame_00001.jpg ...
│       ├── metadata.json
│       ├── descriptions.json
│       └── subtitles.json
└── chroma_db/
    ├── clip_visual/       ← CLIP image embeddings (512d)
    ├── text_visual/       ← e5 описания + субтитры (768d)
    └── text_subtitles/    ← e5 реплики (768d)
```
