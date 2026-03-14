# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video RAG — semantic search system for finding video fragments by text queries. Built for the **MultiLingual Video Fragment Retrieval Challenge** (Kaggle, Master Hackathon ML 2026, Okko/Sber case).

**Metric:** Composite Recall Score = avg(SuccessRate@{1,3,5}, VideoRecall@{1,3,5}), IoU >= 0.5.

**Constraint:** Videos must never leave the server (legal requirement). External APIs allowed only for query processing and metadata.

## Architecture

Two system variants: **Kaggle** (max metric, H100) and **Business** (inference < 1 sec, demo).

### Kaggle Pipeline (`kaggle/pipeline/`)

7-step offline pipeline designed for H100 GPU (~21GB VRAM):

1. **step1_shots** — TransNetV2 shot detection + teammate JSON merge + micro-shot filtering (<2s)
2. **step2_extract** — Parallel: ffmpeg keyframes (CPU) + faster-whisper large-v3 ASR (GPU). Segments mapped to scenes by >50% overlap
3. **step3_vlm_caption** — Qwen3-VL-8B via vLLM (BF16). Multimodal: image + ASR text -> English caption. Batch size 64
4. **step4_scene_docs** — Merge shots + extractions + captions into scenes.jsonl
5. **step5_event_docs** — Sliding window (5 scenes, stride 2) for coarse-grained retrieval
6. **step6_index** — BGE-M3 encode (dense 1024d + sparse lexical) -> Elasticsearch bulk index
7. **search** — 4-channel hybrid (scenes BM25 + dense, events BM25 + dense) -> RRF -> bge-reranker-v2-m3 -> top-5

**Key data flow:** `shot_boundaries.json` -> `extractions.json` -> `captions.json` -> `scenes.jsonl` -> `events.jsonl` -> Elasticsearch indices

**Key separator:** Extraction/caption keys use double underscore: `{video_id}__{scene_idx}`

### Business Demo (`business/`)

FastAPI server with ChromaDB + CLIP + e5 embeddings. Lighter pipeline for real-time demo.

## Common Commands

```bash
# Kaggle pipeline (H100)
python -m kaggle.pipeline.run_pipeline                    # full pipeline
python -m kaggle.pipeline.run_pipeline --skip-shots       # skip step 1
python -m kaggle.pipeline.run_pipeline --search-only      # only generate submission.csv

# Business version
pip install -r business/requirements.txt
python business/indexer/run_pipeline.py --file "video.mp4" --title "Film"
cd business/api && uvicorn api.main:app --reload --port 8000
```

## Key Technical Details

- **BGE-M3 encode:** `BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)` returns `{'dense_vecs', 'lexical_weights'}`. Sparse weights are `{int_token_id: float_weight}`, converted to `{str: float}` for storage
- **FlagReranker:** `compute_score([['query', 'passage'], ...])` — batch format is list of pairs. Returns single float for one pair, list for multiple
- **vLLM multimodal:** `llm.generate({"prompt": ..., "multi_modal_data": {"image": pil_image}})`. Qwen3-VL uses `<|vision_start|><|image_pad|><|vision_end|>` tokens
- **faster-whisper:** `WhisperModel(model, device="cuda", compute_type="float16")`. `transcribe()` returns `(generator, info)` — segments are lazy, must iterate
- **FAISS-GPU:** `IndexFlatIP` (inner product = cosine for normalized BGE-M3 vectors). Build on GPU via `faiss.index_cpu_to_gpu()`, save as CPU index. Two indices: scenes + events
- **In-memory search:** BM25 via `rank_bm25.BM25Okapi` (pickle), sparse via dot-product on lexical_weights. No external services needed (no Elasticsearch)
- **Business e5 prefix:** indexing uses `"passage: "`, search uses `"query: "` — mismatch breaks retrieval
- **Business ChromaDB collections:** `clip_visual`, `text_visual`, `text_subtitles`
- **Query preprocessing** (`preproc/query_preprocessor.py`): rules -> SymSpell -> SAGE T5

## Environment Variables

- `GROQ_API_KEY` — Groq API (business: scene descriptions)
- `OPENROUTER_API_KEY` — OpenRouter API (business: QA endpoint)

## Data Layout

- `data/videos/`, `data/audio/` — media files (not committed)
- `data/frames/` / `KEYFRAMES_DIR` — extracted JPEG frames
- `data/chroma_db/` — ChromaDB (business version)
- Kaggle working dir outputs: `shot_boundaries.json`, `extractions.json`, `captions.json`, `scenes.jsonl`, `events.jsonl`, `faiss_*.index`, `*_meta.pkl`, `sparse_*.pkl`, `bm25_*.pkl`

## Language

Codebase comments and UI are in Russian. Queries support both Russian and English with typo tolerance.
