# 🎬 Video RAG — Semantic Search over Video Fragments

> **Master Hackathon ML 2026 · Okko/Sber case**  
> [MultiLingual Video Fragment Retrieval Challenge](https://www.kaggle.com/competitions/multi-lingual-video-fragment-retrieval-challenge) on Kaggle

---

## 🏆 Task

Given a text query in Russian or English — find the **top-5 most relevant video fragments** across a large movie corpus, with **second-level precision**.

**Metric:** `Composite Recall Score = avg(SuccessRate@{1,3,5}, VideoRecall@{1,3,5})`, IoU ≥ 0.5

---

## ⚡ How it works

The system has two stages: **offline preprocessing** (run once per video corpus) and **online inference** (run per query).

### 🔧 Stage 1 — Offline Preprocessing

```
Video corpus
  │
  ├─ [step 1]  TransNetV2          → shot boundaries (scene list per video)
  │
  ├─ [step 2]  ffmpeg (CPU)        → keyframe JPEG from scene midpoint
  │            faster-whisper      → ASR transcript (large-v3-turbo)
  │              └─ or transcripts.pkl (precomputed, skips Whisper)
  │
  ├─ [step 3]  Qwen3-VL-8B (vLLM) → multimodal caption: image + ASR → EN description
  │
  ├─ [step 4]  scenes.jsonl        ← caption + ASR + train query augmentation
  │
  ├─ [step 5]  events.jsonl        ← sliding window (w=110s, step=10s) for coarse retrieval
  │
  └─ [step 6]  BGE-M3              → dense (1024d) + sparse (lexical) + ColBERT vectors
               FAISS IndexFlatIP   → scene index + event index + train query index
               [step 6b] LoRA fine-tune bge-reranker-v2-m3 on train triplets (optional)
```

### 🔍 Stage 2 — Online Inference

```
Query: "герой плачет под дождём"
  │
  ├─ Preprocessing   normalize + SymSpell (EN) + transliteration
  ├─ Translation     corrected RU + translated EN (from translated_data.csv)
  ├─ Encoding        BGE-M3 batch encode all queries (dense + sparse)
  │
  ├─ Retrieval  ×10 channels (2 languages × 5 index types):
  │   ├─ FAISS dense  — scenes (RU + EN)
  │   ├─ FAISS dense  — events  (RU + EN)
  │   ├─ sparse dot   — scenes  (RU + EN)
  │   ├─ sparse dot   — events  (RU + EN)
  │   └─ ColBERT MaxSim — scenes (RU + EN)
  │
  ├─ Train→test match   if cosine ≥ 0.92 with train query → inject GT answer (boost +5.0)
  ├─ RRF fusion         Reciprocal Rank Fusion across all channels
  ├─ Dedup              IoU > 0.3 → remove overlapping duplicates → top-50
  │
  ├─ Reranking          bge-reranker-v2-m3, language-aware: max(score_ru, score_en)
  ├─ Timecode expand    ±55s from scene center (window 110s, cap 180s)
  └─ Video clustering   prefer fragments from top-2 scoring videos → top-5

  → submission.csv
```

---

## 🗂 Repository structure

```
video-rag/
├── kaggle/
│   ├── pipeline/
│   │   ├── config.py                    # paths & constants, auto-detect local/server
│   │   ├── run_pipeline.py              # orchestrator (entry point)
│   │   ├── step1_shots.py               # TransNetV2 shot detection
│   │   ├── step2_extract.py             # keyframes + ASR (parallel)
│   │   ├── step2_3_stream.py            # streaming pipeline (steps 2+3 merged)
│   │   ├── step3_vlm_caption.py         # Qwen3-VL-8B captioning
│   │   ├── step4_scene_docs.py          # build scenes.jsonl
│   │   ├── step5_event_docs.py          # build events.jsonl
│   │   ├── step6_index.py               # BGE-M3 → FAISS + train index
│   │   ├── step6b_finetune_reranker.py  # LoRA fine-tune reranker (optional)
│   │   ├── search.py                    # retrieval + reranking → submission.csv
│   │   ├── retry_failed.py              # re-caption failed scenes
│   │   └── import_transcripts.py        # merge teammate ASR
│   └── submit.ipynb                     # Kaggle notebook: load indexes → submit
├── preproc/
│   └── query_preprocessor.py            # normalization + SymSpell + SAGE + transliteration
├── setup_h100.sh                        # one-shot server setup script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick start

### 1. Server setup (run once)

```bash
bash setup_h100.sh
cd /kaggle/working/video-rag
```

### 2. Run offline preprocessing (H100, ~few hours)

```bash
# Full pipeline
python -m kaggle.pipeline.run_pipeline

# Streaming mode — steps 2+3 run in parallel (faster on H100)
python -m kaggle.pipeline.run_pipeline --stream

# With optional reranker fine-tuning (~10 min extra)
python -m kaggle.pipeline.run_pipeline --stream --finetune
```

**Skip flags** (useful for partial reruns):

```bash
python -m kaggle.pipeline.run_pipeline --skip-shots --skip-extract --skip-vlm
```

### 3. Rebuild index without re-running VLM/Whisper

```bash
python -m kaggle.pipeline.step4_scene_docs
python -m kaggle.pipeline.step5_event_docs
python -m kaggle.pipeline.step6_index
```

### 4. Generate submission

```bash
# With cached retrieval (fast)
python -m kaggle.pipeline.search

# Force recompute retrieval
python -m kaggle.pipeline.search --no-cache
```

### 5. Kaggle Notebook submission

```bash
# Pack indexes on server
tar czf indexes.tar.gz \
    /kaggle/working/*.index \
    /kaggle/working/*.pkl \
    /kaggle/working/scenes.jsonl \
    /kaggle/working/events.jsonl

# Upload as Kaggle Dataset → attach to notebook → run kaggle/submit.ipynb
```

---

## 🧱 Tech stack

| Component | Model / Tool |
|---|---|
| Shot detection | TransNetV2 |
| Keyframe extraction | ffmpeg (CPU, 16 threads) |
| Speech recognition | faster-whisper `large-v3-turbo` |
| Visual captioning | Qwen3-VL-8B via vLLM (BF16) |
| Embedding | BGE-M3 — dense 1024d + sparse + ColBERT |
| Vector index | FAISS `IndexFlatIP` (CPU) |
| Reranker | bge-reranker-v2-m3 (+ optional LoRA fine-tune) |
| Query preprocessing | SymSpell (EN) + SAGE T5 (RU, optional) |
| GPU | H100 (~21 GB VRAM) |

---

## ⚙️ Environment variables

| Variable | Used by | Required |
|---|---|---|
| `PROXY_API_KEY` | Gemini via ProxyAPI | not needed for base pipeline |

---

## 📋 Key implementation notes

<details>
<summary>BGE-M3 encoding</summary>

```python
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
output = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=True)
# output['dense_vecs']      — np.ndarray (N, 1024)
# output['lexical_weights'] — list of {int_token_id: float}
# output['colbert_vecs']    — list of np.ndarray (n_tokens, 1024), stored as fp16
```
</details>

<details>
<summary>Sparse search</summary>

No external services. Pure dot-product over `lexical_weights` dicts:
```python
score = sum(query_sparse.get(k, 0.0) * v for k, v in doc_sparse.items())
```
</details>

<details>
<summary>RRF fusion</summary>

UID = `{video_id}_{start}_{end}` — same fragment from multiple channels **accumulates** score:
```python
score[uid] += 1.0 / (RRF_K + rank)
```
</details>

<details>
<summary>Train→test matching</summary>

- cosine ≥ 0.92 → inject ground-truth answer with reranker score boost **+5.0**
- cosine ≥ 0.80 → add to candidate pool (competes normally)
</details>

<details>
<summary>Timecode expansion</summary>

Short scenes expanded to **±55s** from center (110s window). Hard cap: 180s.
Train matches keep original ground-truth timecodes unchanged.
</details>