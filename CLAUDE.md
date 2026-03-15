# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video RAG — semantic search system for finding video fragments by text queries. Built for the **MultiLingual Video Fragment Retrieval Challenge** (Kaggle, Master Hackathon ML 2026, Okko/Sber case).

**Metric:** Composite Recall Score = avg(SuccessRate@{1,3,5}, VideoRecall@{1,3,5}), IoU >= 0.5.

**Constraint:** Videos must never leave the server (legal requirement). External APIs allowed only for query processing and metadata.

## Architecture

Two system variants: **Kaggle** (max metric, H100) and **Business** (inference < 1 sec, demo).

### Kaggle Pipeline (`kaggle/pipeline/`)

10-step offline pipeline designed for H100 GPU (~21GB VRAM):

1. **step1_shots** — TransNetV2 shot detection + teammate JSON merge + micro-shot filtering (<2s)
2. **step2_extract** — Parallel: ffmpeg keyframes (CPU, 16 threads) + faster-whisper large-v3-turbo ASR (GPU) or precomputed transcripts.pkl. Segments mapped to scenes by >50% overlap
3. **step3_vlm_caption** — Qwen3-VL-8B via vLLM (BF16, gpu_memory_utilization=0.85). Multimodal: image + ASR text -> English caption. Batch size 128
3b. **step3b_fix_captions** — LLM post-processing (Gemini via ProxyAPI): VLM caption + ASR -> improved bilingual EN+RU descriptions. No GPU needed
3c. **step3c_doc2query** — LLM synthetic query generation: 4 queries per scene (mixed RU+EN) for better recall. No GPU needed
4. **step4_scene_docs** — Merge shots + extractions + captions + synthetic queries + train augmentation into scenes.jsonl
5. **step5_event_docs** — Time-based sliding window (w=110s, step=10s) for coarse-grained retrieval
6. **step6_index** — BGE-M3 encode (dense 1024d + sparse lexical + ColBERT token vectors) -> FAISS CPU IndexFlatIP + train query index + colbert_scenes.pkl
6b. **step6b_finetune_reranker** — LoRA fine-tune bge-reranker-v2-m3 on train triplets (query, positive scene, hard negative)
7. **search** — Multi-query (corrected RU + translated EN) × 10 channels (dense + sparse + ColBERT × scenes + dense + sparse × events × 2 queries) -> RRF fusion (cross-channel score accumulation) -> dedup (IoU>0.3) -> top-50 reranker -> ±55s timecode expansion (cap 180s) -> video clustering -> top-5

**Key data flow:** `shot_boundaries.json` -> `extractions.json` -> `captions.json` -> `scenes.jsonl` -> `events.jsonl` -> FAISS indices

**Key separator:** Extraction/caption keys use double underscore: `{video_id}__{scene_idx}`

### Business Demo (`business/`)

FastAPI server with ChromaDB + CLIP + e5 embeddings. Lighter pipeline for real-time demo.

## Common Commands

```bash
# Kaggle pipeline (H100) — full
python -m kaggle.pipeline.run_pipeline

# LLM post-processing (no GPU, needs PROXY_API_KEY)
PROXY_API_KEY=... python -m kaggle.pipeline.step3b_fix_captions --workers 20
PROXY_API_KEY=... python -m kaggle.pipeline.step3c_doc2query --workers 10

# Rebuild docs (no GPU, seconds)
python -m kaggle.pipeline.step4_scene_docs
python -m kaggle.pipeline.step5_event_docs

# Index + search (GPU)
python -m kaggle.pipeline.step6_index
python -m kaggle.pipeline.step6b_finetune_reranker  # optional, ~10 min
python -m kaggle.pipeline.search
python -m kaggle.pipeline.search --no-cache  # force recompute retrieval

# Business version
pip install -r business/requirements.txt
python business/indexer/run_pipeline.py --file "video.mp4" --title "Film"
cd business/api && uvicorn api.main:app --reload --port 8000
```

## Key Technical Details

- **BGE-M3 encode:** `BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)` returns `{'dense_vecs', 'lexical_weights', 'colbert_vecs'}`. Sparse weights are `{int_token_id: float_weight}`, converted to `{str: float}` for storage. ColBERT vecs are per-token (n_tokens, 1024), stored as fp16
- **FlagReranker:** `compute_score([['query', 'passage'], ...])` — batch format is list of pairs. Returns single float for one pair, list for multiple
- **vLLM multimodal:** `llm.generate({"prompt": ..., "multi_modal_data": {"image": pil_image}})`. Qwen3-VL uses `<|vision_start|><|image_pad|><|vision_end|>` tokens
- **faster-whisper:** `WhisperModel(model, device="cuda", compute_type="float16")`. `transcribe()` returns `(generator, info)` — segments are lazy, must iterate
- **FAISS CPU:** `IndexFlatIP` (inner product = cosine for normalized BGE-M3 vectors). CPU-only — faiss-gpu not compiled for H100 sm_90. Three indices: scenes, events, train queries
- **Sparse search:** dot-product on lexical_weights dicts. No external services needed
- **ColBERT search:** MaxSim re-scoring of FAISS scene candidates — for each query token, max cosine sim to any doc token, then sum. Adds 2 RRF channels (scenes × 2 queries)
- **RRF fusion:** uid = `{video_id}_{start}_{end}` (no source in key) — same document from multiple channels gets accumulated score, not separate entries
- **Train→test matching:** FAISS cosine sim on train queries. score ≥ 0.92 → inject ground-truth answer with +2.0 reranker boost. score ≥ 0.80 → add to candidate pool
- **Timecode expansion:** Results expanded ±55s from scene center (110s window). Segments >180s trimmed to 180s centered. Train matches keep ground-truth timecodes
- **Language-aware reranking:** RU query matched to llm_caption_ru, EN to llm_caption_en. Dual reranking with max(score_ru, score_en)
- **Batch search pipeline:** Phase 1a: batch BGE-M3 encode all queries. Phase 1b: FAISS+sparse retrieval (cached to retrieval_cache.pkl). Phase 2: batch reranking (32 queries per GPU call). RERANKER_TOP_K=50, RERANKER_OUTPUT_K=10
- **Video clustering:** After reranking, prefer multiple fragments from top-2 scoring videos to fill top-5 slots
- **LLM proxy:** Gemini via `google-genai` SDK, base_url `https://api.proxyapi.ru/google`
- **Business e5 prefix:** indexing uses `"passage: "`, search uses `"query: "` — mismatch breaks retrieval
- **Business ChromaDB collections:** `clip_visual`, `text_visual`, `text_subtitles`

## Environment Variables

- `PROXY_API_KEY` — ProxyAPI key for Gemini (step3b, step3c)
- `GROQ_API_KEY` — Groq API (business: scene descriptions)
- `OPENROUTER_API_KEY` — OpenRouter API (business: QA endpoint)

## Data Layout

- `data/videos/`, `data/audio/` — media files (not committed)
- `data/frames/` / `KEYFRAMES_DIR` — extracted JPEG frames
- `data/pipeline/` — local working directory (auto-detected vs /kaggle/working)
- `data/chroma_db/` — ChromaDB (business version)
- Working dir outputs: `shot_boundaries.json`, `extractions.json`, `captions.json`, `scenes.jsonl`, `events.jsonl`, `faiss_*.index`, `*_meta.pkl`, `sparse_*.pkl`, `colbert_scenes.pkl`, `retrieval_cache.pkl`, `translated_data.csv`

## Language

Codebase comments and UI are in Russian. Queries support both Russian and English with typo tolerance.
