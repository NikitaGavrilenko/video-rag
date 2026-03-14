"""
config.py — paths, constants, and shared settings for the Kaggle pipeline.
"""

from pathlib import Path

# ── Kaggle paths ────────────────────────────────────────────────────────────
DATA_DIR = Path("/kaggle/input/competitions/multi-lingual-video-fragment-retrieval-challenge/video-rag")
VIDEO_DIR = DATA_DIR / "videos"
AUDIO_DIR = DATA_DIR / "audio"
TRAIN_CSV = DATA_DIR / "train" / "train_qa.csv"
TEST_CSV = DATA_DIR / "test" / "test.csv"
TRANSCRIPTS_PKL = DATA_DIR / "transcripts.pkl"

# ── Working directory (output) ──────────────────────────────────────────────
WORK_DIR = Path("/kaggle/working")
SHOTS_FILE = WORK_DIR / "shot_boundaries.json"
TEAMMATE_SHOTS_FILE = WORK_DIR / "teammate_shots.json"  # приоритетная нарезка от тиммейта
TRANSCRIPTS_FILE = WORK_DIR / "transcripts_large.pkl"
KEYFRAMES_DIR = WORK_DIR / "keyframes"
SCENES_FILE = WORK_DIR / "scenes.jsonl"
EVENTS_FILE = WORK_DIR / "events.jsonl"

# ── Shot detection ──────────────────────────────────────────────────────────
MIN_SHOT_DURATION = 2.0  # секунды, фильтр микрошотов

# ── Whisper ASR ─────────────────────────────────────────────────────────────
WHISPER_MODEL = "large-v3"
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
RERANKER_TOP_K = 100  # входной размер для reranker
RERANKER_OUTPUT_K = 10  # после reranker

# ── FAISS index (GPU) ───────────────────────────────────────────────────────
FAISS_SCENES_INDEX = WORK_DIR / "faiss_scenes.index"
FAISS_EVENTS_INDEX = WORK_DIR / "faiss_events.index"
SCENES_META_FILE = WORK_DIR / "scenes_meta.pkl"     # metadata parallel to FAISS
EVENTS_META_FILE = WORK_DIR / "events_meta.pkl"
SPARSE_SCENES_FILE = WORK_DIR / "sparse_scenes.pkl"  # sparse vectors for scenes
SPARSE_EVENTS_FILE = WORK_DIR / "sparse_events.pkl"
BM25_SCENES_FILE = WORK_DIR / "bm25_scenes.pkl"      # BM25Okapi for scenes
BM25_EVENTS_FILE = WORK_DIR / "bm25_events.pkl"      # BM25Okapi for events

# ── Event documents ─────────────────────────────────────────────────────────
EVENT_WINDOW_SIZE = 5  # сцен в окне
EVENT_WINDOW_STRIDE = 2  # шаг скольжения

# ── Search ──────────────────────────────────────────────────────────────────
SEARCH_TOP_K_DENSE = 50
SEARCH_TOP_K_SPARSE = 50
RRF_K = 60
FINAL_TOP_N = 5
