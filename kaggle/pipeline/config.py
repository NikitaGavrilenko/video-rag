"""
config.py — paths, constants, and shared settings for the Kaggle pipeline.

Auto-detects environment: /kaggle/working (server) vs data/pipeline/ (local).
"""

import os
from pathlib import Path

# ── Auto-detect environment ────────────────────────────────────────────────
_KAGGLE_WORK = Path("/kaggle/working")
_LOCAL_ROOT = Path(__file__).parent.parent.parent  # video-rag/
_LOCAL_WORK = _LOCAL_ROOT / "data" / "pipeline"

if _KAGGLE_WORK.exists():
    WORK_DIR = _KAGGLE_WORK
    DATA_DIR = Path("/kaggle/input/competitions/multi-lingual-video-fragment-retrieval-challenge/video-rag")
else:
    WORK_DIR = _LOCAL_WORK
    DATA_DIR = _LOCAL_ROOT / "data"

VIDEO_DIR = DATA_DIR / "videos"
AUDIO_DIR = DATA_DIR / "audio"
TRAIN_CSV = DATA_DIR / "train" / "train_qa.csv"
TEST_CSV = DATA_DIR / "test" / "test.csv"
TRANSCRIPTS_PKL = DATA_DIR / "transcripts.pkl"

# ── Working directory (output) ──────────────────────────────────────────────
SHOTS_FILE = WORK_DIR / "shot_boundaries.json"
TEAMMATE_SHOTS_FILE = WORK_DIR / "teammate_shots.json"
TRANSCRIPTS_FILE = WORK_DIR / "transcripts_large.pkl"
KEYFRAMES_DIR = WORK_DIR / "keyframes"
SCENES_FILE = WORK_DIR / "scenes.jsonl"
EVENTS_FILE = WORK_DIR / "events.jsonl"

# ── Shot detection ──────────────────────────────────────────────────────────
MIN_SHOT_DURATION = 2.0  # секунды, фильтр микрошотов

# ── Whisper ASR ─────────────────────────────────────────────────────────────
WHISPER_MODEL = "large-v3-turbo"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"  # H100 supports bf16/fp16
WHISPER_BATCH_SIZE = 16
ASR_OVERLAP_SEC = 1.5  # расширение окна сцены для захвата ASR из соседних сцен

# ── VLM Captioning ─────────────────────────────────────────────────────────
VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
VLM_MAX_TOKENS = 256
VLM_BATCH_SIZE = 128
VLM_TENSOR_PARALLEL = 1  # single GPU sufficient for 8B

VLM_PROMPT_TEMPLATE = """Speech transcript for this scene: "{asr_text}"

Describe what is happening in this video scene. Use both the visual content and the speech transcript to create a comprehensive description in English. 2-4 sentences."""

# ── BGE-M3 ──────────────────────────────────────────────────────────────────
BGE_MODEL = "BAAI/bge-m3"
BGE_BATCH_SIZE = 512

# ── Reranker ────────────────────────────────────────────────────────────────
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_FINETUNED = WORK_DIR / "reranker_finetuned"
RERANKER_TOP_K = 50   # входной размер для reranker (после event→scene explosion)
RERANKER_OUTPUT_K = 10  # после reranker

# ── FAISS index ─────────────────────────────────────────────────────────────
FAISS_SCENES_INDEX = WORK_DIR / "faiss_scenes.index"
FAISS_EVENTS_INDEX = WORK_DIR / "faiss_events.index"
SCENES_META_FILE = WORK_DIR / "scenes_meta.pkl"
EVENTS_META_FILE = WORK_DIR / "events_meta.pkl"
SPARSE_SCENES_FILE = WORK_DIR / "sparse_scenes.pkl"
SPARSE_EVENTS_FILE = WORK_DIR / "sparse_events.pkl"
COLBERT_SCENES_FILE = WORK_DIR / "colbert_scenes.pkl"
BM25_SCENES_FILE = WORK_DIR / "bm25_scenes.pkl"
BM25_EVENTS_FILE = WORK_DIR / "bm25_events.pkl"

# ── Event documents ─────────────────────────────────────────────────────────
EVENT_WINDOW_SEC = 110  # секунд в окне (time-based)
EVENT_STRIDE_SEC = 10   # шаг скольжения в секундах

# ── Translated queries (pre-corrected + English translations) ──────────────
TRANSLATED_CSV = WORK_DIR / "translated_data.csv"

# ── LLM API (proxy for caption post-processing) ───────────────────────────
PROXY_API_BASE = "https://api.proxyapi.ru/google"
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "")
PROXY_API_MODEL = "gemini-2.0-flash-lite"

# ── Search ──────────────────────────────────────────────────────────────────
SEARCH_TOP_K_DENSE = 50
SEARCH_TOP_K_SPARSE = 50
RRF_K = 60
FINAL_TOP_N = 5
