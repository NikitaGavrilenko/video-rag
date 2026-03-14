"""
step2_3_stream.py — Streaming producer-consumer pipeline that merges keyframe
extraction (step 2) and VLM captioning (step 3).

Architecture:
    Producer 1 (CPU): ffmpeg keyframe extraction  ──┐
    Producer 2 (GPU): faster-whisper ASR per video  ──┤──→ Queue ──→ VLM Consumer (GPU)
                                                      │              batches of 64

As soon as BOTH keyframe AND asr_text are ready for a scene, it enters the VLM
queue.  The consumer drains the queue in batches and checkpoints after each one.

Output:
    extractions.json  — same format as step2
    captions.json     — same format as step3
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from PIL import Image
from vllm import LLM, SamplingParams

from .config import (
    AUDIO_DIR,
    KEYFRAMES_DIR,
    VLM_BATCH_SIZE,
    VLM_MAX_TOKENS,
    VLM_MODEL,
    VLM_PROMPT_TEMPLATE,
    VLM_TENSOR_PARALLEL,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
    WORK_DIR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

EXTRACTIONS_FILE = WORK_DIR / "extractions.json"
CAPTIONS_FILE = WORK_DIR / "captions.json"

SIMPLE_PROMPT = (
    "Describe what is happening in this video scene based on the visual content. "
    "2-4 sentences in English."
)

# ── Shared state ──────────────────────────────────────────────────────────────

# Per-scene tracking dict.  Each value is a dict with optional keys
# "keyframe_path" and "asr_text".  When both are present the scene is ready.
extraction_ready: dict[str, dict[str, str]] = {}
extraction_lock = threading.Lock()

# Scene keys that have both keyframe and asr_text ready (and not yet captioned).
vlm_queue: Queue[str | None] = Queue()

# Counters for progress reporting.
_keyframes_done = 0
_videos_transcribed = 0
_captions_done = 0
_counter_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _scene_key(video_id: str, scene_idx: int) -> str:
    return f"{video_id}__{scene_idx}"


def _mark_ready(key: str, field: str, value: str, already_captioned: set[str]) -> None:
    """Thread-safe update of *extraction_ready*.  Enqueues to VLM when complete."""
    with extraction_lock:
        entry = extraction_ready.setdefault(key, {})
        entry[field] = value
        if "keyframe_path" in entry and "asr_text" in entry:
            if key not in already_captioned:
                vlm_queue.put(key)


def _build_multimodal_input(image: Image.Image, text_prompt: str) -> dict[str, Any]:
    """Build a vLLM multimodal input dict for Qwen3-VL."""
    prompt = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{text_prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }


def _save_extractions() -> None:
    with extraction_lock:
        snapshot = {k: dict(v) for k, v in extraction_ready.items()}
    EXTRACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EXTRACTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def _save_captions(captions: dict[str, dict[str, str]]) -> None:
    CAPTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CAPTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=1)


# ── Producer 1 — Keyframe extraction (CPU) ───────────────────────────────────


def _extract_keyframe(
    video_path: Path,
    keyframe_time: float,
    out_path: Path,
    max_side: int = 768,
) -> Path:
    """Extract a single JPEG keyframe resized to *max_side* on the longest edge."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scale_filter = (
        f"scale='if(gte(iw,ih),{max_side},-2)':'if(gte(iw,ih),-2,{max_side})'"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hwaccel", "cuda",
        "-ss", str(keyframe_time),
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", scale_filter,
        "-q:v", "2",
        str(out_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path} @ {keyframe_time}s: "
            f"{result.stderr.decode(errors='replace')[:500]}"
        )
    return out_path


def keyframe_producer(
    shots: list[dict[str, Any]],
    already_captioned: set[str],
) -> None:
    """Extract keyframes for every scene.  Runs in its own thread."""
    global _keyframes_done

    total = sum(len(v["scenes"]) for v in shots)

    for video_entry in shots:
        video_id: str = video_entry["video_id"]
        audio_key: str = video_entry["audio_key"]
        video_path = AUDIO_DIR.parent / audio_key

        for scene in video_entry["scenes"]:
            scene_idx: int = scene["scene_idx"]
            keyframe_time: float = scene["keyframe_time"]
            out_path = KEYFRAMES_DIR / video_id / f"scene_{scene_idx:04d}.jpg"
            key = _scene_key(video_id, scene_idx)

            # Resume: if file already on disk, register and move on.
            if out_path.exists():
                _mark_ready(key, "keyframe_path", str(out_path), already_captioned)
                with _counter_lock:
                    _keyframes_done += 1
                continue

            try:
                _extract_keyframe(video_path, keyframe_time, out_path)
                _mark_ready(key, "keyframe_path", str(out_path), already_captioned)
            except Exception:
                logger.exception(
                    "Keyframe extraction failed: %s scene %d", video_id, scene_idx
                )
                _mark_ready(key, "keyframe_path", "", already_captioned)

            with _counter_lock:
                _keyframes_done += 1
                if _keyframes_done % 50 == 0 or _keyframes_done == total:
                    print(
                        f"[progress] Extracted {_keyframes_done} keyframes, "
                        f"Transcribed {_videos_transcribed} videos, "
                        f"Captioned {_captions_done} scenes"
                    )

    logger.info("Keyframe producer finished (%d frames).", _keyframes_done)


# ── Producer 2 — Whisper ASR (GPU) ───────────────────────────────────────────


def _audio_path_for_video(audio_key: str) -> Path | None:
    """Resolve an audio file from *audio_key* (e.g. 'videos/video_xxx.mp4')."""
    video_stem = Path(audio_key).stem
    video_hash = video_stem.replace("video_", "", 1)

    candidates = list(AUDIO_DIR.glob(f"audio_{video_hash}.*"))
    if candidates:
        return candidates[0]

    fallback = AUDIO_DIR.parent / audio_key
    if fallback.exists():
        return fallback

    return None


def _overlap_fraction(
    seg_start: float, seg_end: float, sc_start: float, sc_end: float
) -> float:
    seg_dur = seg_end - seg_start
    if seg_dur <= 0:
        return 0.0
    overlap = max(0.0, min(seg_end, sc_end) - max(seg_start, sc_start))
    return overlap / seg_dur


def whisper_producer(
    shots: list[dict[str, Any]],
    already_captioned: set[str],
) -> None:
    """Transcribe every unique audio file and map segments to scenes."""
    global _videos_transcribed

    from faster_whisper import WhisperModel

    logger.info("Loading faster-whisper model %s ...", WHISPER_MODEL)
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    logger.info("Whisper model loaded.")

    # Group videos by resolved audio path to avoid duplicate transcriptions.
    audio_to_videos: dict[str, list[dict[str, Any]]] = {}
    for video_entry in shots:
        audio_key = video_entry["audio_key"]
        audio_path = _audio_path_for_video(audio_key)
        if audio_path is None:
            logger.warning(
                "No audio found for %s (key=%s), skipping ASR",
                video_entry["video_id"],
                audio_key,
            )
            for scene in video_entry["scenes"]:
                key = _scene_key(video_entry["video_id"], scene["scene_idx"])
                _mark_ready(key, "asr_text", "", already_captioned)
            continue
        audio_to_videos.setdefault(str(audio_path), []).append(video_entry)

    seen_audio: dict[str, list[dict[str, Any]]] = {}
    total_audio = len(audio_to_videos)

    for idx, (ap_str, video_entries) in enumerate(audio_to_videos.items(), 1):
        print(f"[ASR] Transcribing {idx}/{total_audio}: {Path(ap_str).name}")

        if ap_str not in seen_audio:
            try:
                segments_iter, _info = model.transcribe(ap_str, beam_size=5, vad_filter=True)
                seg_list = [
                    {"start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in segments_iter
                ]
                seen_audio[ap_str] = seg_list
            except Exception:
                logger.exception("Whisper transcription failed for %s", ap_str)
                seen_audio[ap_str] = []

        segments = seen_audio[ap_str]

        for video_entry in video_entries:
            video_id = video_entry["video_id"]
            for scene in video_entry["scenes"]:
                scene_idx: int = scene["scene_idx"]
                sc_start: float = scene["start"]
                sc_end: float = scene["end"]

                matched_texts: list[str] = []
                for seg in segments:
                    if _overlap_fraction(seg["start"], seg["end"], sc_start, sc_end) > 0.5:
                        text = seg["text"].strip()
                        if text:
                            matched_texts.append(text)

                key = _scene_key(video_id, scene_idx)
                _mark_ready(key, "asr_text", " ".join(matched_texts), already_captioned)

        with _counter_lock:
            _videos_transcribed += 1
            print(
                f"[progress] Extracted {_keyframes_done} keyframes, "
                f"Transcribed {_videos_transcribed} videos, "
                f"Captioned {_captions_done} scenes"
            )

    logger.info("Whisper producer finished (%d audio files).", _videos_transcribed)


# ── Consumer — VLM captioning ────────────────────────────────────────────────


def vlm_consumer(
    num_producers: int,
    captions: dict[str, dict[str, str]],
) -> None:
    """Drain the VLM queue in batches, run inference, and checkpoint."""
    global _captions_done

    logger.info("Initialising vLLM with model %s ...", VLM_MODEL)
    llm = LLM(
        model=VLM_MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        tensor_parallel_size=VLM_TENSOR_PARALLEL,
        limit_mm_per_prompt={"image": 1},
    )
    sampling_params = SamplingParams(max_tokens=VLM_MAX_TOKENS, temperature=0.3)
    logger.info("vLLM ready.")

    sentinels_received = 0
    batch_keys: list[str] = []

    def _flush_batch() -> None:
        """Run VLM inference on the accumulated batch and checkpoint."""
        global _captions_done
        if not batch_keys:
            return

        inputs: list[dict[str, Any]] = []
        valid_keys: list[str] = []

        for key in batch_keys:
            with extraction_lock:
                entry = extraction_ready.get(key, {})
                kf_path = entry.get("keyframe_path", "")
                asr_text = entry.get("asr_text", "")

            if not kf_path or not Path(kf_path).exists():
                captions[key] = {"vlm_caption": ""}
                continue

            image = Image.open(kf_path).convert("RGB")
            if asr_text.strip():
                text_prompt = VLM_PROMPT_TEMPLATE.format(asr_text=asr_text)
            else:
                text_prompt = SIMPLE_PROMPT

            inputs.append(_build_multimodal_input(image, text_prompt))
            valid_keys.append(key)

        if inputs:
            outputs = llm.generate(inputs, sampling_params=sampling_params)
            for key, output in zip(valid_keys, outputs):
                captions[key] = {"vlm_caption": output.outputs[0].text.strip()}

        with _counter_lock:
            _captions_done += len(batch_keys)
            print(
                f"[progress] Extracted {_keyframes_done} keyframes, "
                f"Transcribed {_videos_transcribed} videos, "
                f"Captioned {_captions_done} scenes"
            )

        _save_captions(captions)
        batch_keys.clear()

    while sentinels_received < num_producers:
        try:
            item = vlm_queue.get(timeout=2.0)
        except Empty:
            # Timeout — flush a partial batch if we have anything queued.
            if batch_keys:
                _flush_batch()
            continue

        if item is None:
            sentinels_received += 1
            continue

        batch_keys.append(item)
        if len(batch_keys) >= VLM_BATCH_SIZE:
            _flush_batch()

    # Flush remaining scenes after all producers have finished.
    _flush_batch()
    logger.info("VLM consumer finished (%d captions total).", len(captions))


# ── Orchestration ─────────────────────────────────────────────────────────────


def main(shots: list[dict[str, Any]] | None = None) -> None:
    """Run the streaming extraction → captioning pipeline.

    Parameters
    ----------
    shots : list[dict], optional
        Pre-loaded shots list.  When *None*, loads from
        ``WORK_DIR / "shot_boundaries.json"``.
    """
    global extraction_ready, _keyframes_done, _videos_transcribed, _captions_done

    t0 = time.time()

    # ── Load shots if not provided ────────────────────────────────────────
    if shots is None:
        from .config import SHOTS_FILE

        logger.info("Loading shots from %s", SHOTS_FILE)
        with open(SHOTS_FILE, "r", encoding="utf-8") as f:
            shots = json.load(f)

    total_scenes = sum(len(v["scenes"]) for v in shots)
    logger.info("Loaded %d videos, %d scenes total.", len(shots), total_scenes)

    # ── Resume: load existing artefacts ───────────────────────────────────
    if EXTRACTIONS_FILE.exists():
        with open(EXTRACTIONS_FILE, "r", encoding="utf-8") as f:
            extraction_ready = json.load(f)
        logger.info("Resumed %d entries from extractions.json.", len(extraction_ready))
    else:
        extraction_ready = {}

    captions: dict[str, dict[str, str]] = {}
    if CAPTIONS_FILE.exists():
        with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
            captions = json.load(f)
        logger.info("Resumed %d captions from captions.json.", len(captions))

    already_captioned: set[str] = set(captions.keys())

    # Reset progress counters.
    _keyframes_done = 0
    _videos_transcribed = 0
    _captions_done = len(already_captioned)

    # ── Launch producers ──────────────────────────────────────────────────
    num_producers = 2

    def _run_producer(fn: Any, *args: Any) -> None:
        try:
            fn(*args)
        except Exception:
            logger.exception("Producer %s crashed", fn.__name__)
        finally:
            vlm_queue.put(None)  # sentinel

    kf_thread = threading.Thread(
        target=_run_producer,
        args=(keyframe_producer, shots, already_captioned),
        name="keyframe-producer",
        daemon=True,
    )
    whisper_thread = threading.Thread(
        target=_run_producer,
        args=(whisper_producer, shots, already_captioned),
        name="whisper-producer",
        daemon=True,
    )

    kf_thread.start()
    whisper_thread.start()

    # ── Run VLM consumer on the main thread ───────────────────────────────
    vlm_consumer(num_producers, captions)

    # ── Wait for producer threads to fully exit ───────────────────────────
    kf_thread.join()
    whisper_thread.join()

    # ── Final save of extractions ─────────────────────────────────────────
    _save_extractions()

    elapsed = time.time() - t0
    logger.info(
        "Pipeline done: %d extractions, %d captions (%.1fs).",
        len(extraction_ready),
        len(captions),
        elapsed,
    )


if __name__ == "__main__":
    main()
