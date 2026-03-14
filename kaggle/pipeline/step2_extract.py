"""
step2_extract.py — Extract keyframes (CPU/FFmpeg) and ASR transcripts (GPU/faster-whisper)
in parallel for every scene produced by step 1.

Input : SHOTS_FILE  (shot_boundaries.json)
Output: WORK_DIR / "extractions.json"
        KEYFRAMES_DIR / {video_id} / scene_{scene_idx:04d}.jpg
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config import (
    ASR_OVERLAP_SEC,
    AUDIO_DIR,
    KEYFRAMES_DIR,
    SHOTS_FILE,
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

# ---------------------------------------------------------------------------
# Stream 1 — FFmpeg keyframe extraction (CPU)
# ---------------------------------------------------------------------------


def _extract_keyframe(
    video_path: Path,
    keyframe_time: float,
    out_path: Path,
    max_side: int = 768,
) -> Path:
    """Extract a single JPEG keyframe resized to *max_side* on the longest edge."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # vf scale filter: scale so the longest side = max_side, keep aspect ratio
    scale_filter = (
        f"scale='if(gte(iw,ih),{max_side},-2)':'if(gte(iw,ih),-2,{max_side})'"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-hwaccel", "cuda",        # GPU-accelerated decoding via NVDEC
        "-ss", str(keyframe_time),
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", scale_filter,
        "-q:v", "2",
        str(out_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path} @ {keyframe_time}s: "
            f"{result.stderr.decode(errors='replace')[:500]}"
        )
    return out_path


def extract_all_keyframes(
    shots: list[dict[str, Any]],
) -> dict[str, str]:
    """Extract keyframes for every scene. Returns {video_id__scene_idx: path}."""
    results: dict[str, str] = {}
    total = sum(len(v["scenes"]) for v in shots)
    done = 0

    for video_entry in shots:
        video_id: str = video_entry["video_id"]
        audio_key: str = video_entry["audio_key"]
        # Resolve the video file — audio_key is relative like "videos/video_xxx.mp4"
        video_path = AUDIO_DIR.parent / audio_key

        for scene in video_entry["scenes"]:
            scene_idx: int = scene["scene_idx"]
            keyframe_time: float = scene["keyframe_time"]
            out_path = KEYFRAMES_DIR / video_id / f"scene_{scene_idx:04d}.jpg"
            key = f"{video_id}__{scene_idx}"

            # Resume support — skip if already extracted
            if out_path.exists():
                results[key] = str(out_path)
                done += 1
                continue

            try:
                _extract_keyframe(video_path, keyframe_time, out_path)
                results[key] = str(out_path)
            except Exception:
                logger.exception(
                    "Keyframe extraction failed: %s scene %d", video_id, scene_idx
                )
                results[key] = ""

            done += 1
            if done % 50 == 0 or done == total:
                print(f"[keyframes] {done}/{total}")

    return results


# ---------------------------------------------------------------------------
# Stream 2 — faster-whisper ASR (GPU)
# ---------------------------------------------------------------------------


def _audio_path_for_video(audio_key: str) -> Path | None:
    """Resolve an audio file from audio_key like 'videos/video_xxx.mp4'.

    We look for a matching file in AUDIO_DIR named audio_{hash}.* where
    {hash} comes from the video filename video_{hash}.ext.
    """
    video_stem = Path(audio_key).stem  # e.g. "video_xxx"
    video_hash = video_stem.replace("video_", "", 1)

    # Search AUDIO_DIR for audio_{hash}.*
    candidates = list(AUDIO_DIR.glob(f"audio_{video_hash}.*"))
    if candidates:
        return candidates[0]

    # Fallback: try the video file itself (ffmpeg/whisper can often read mp4)
    fallback = AUDIO_DIR.parent / audio_key
    if fallback.exists():
        return fallback

    return None


def _overlap_fraction(seg_start: float, seg_end: float, sc_start: float, sc_end: float) -> float:
    """Return fraction of the segment's duration that overlaps the scene."""
    seg_dur = seg_end - seg_start
    if seg_dur <= 0:
        return 0.0
    overlap_start = max(seg_start, sc_start)
    overlap_end = min(seg_end, sc_end)
    overlap = max(0.0, overlap_end - overlap_start)
    return overlap / seg_dur


def transcribe_all(
    shots: list[dict[str, Any]],
) -> dict[str, str]:
    """Transcribe every unique audio file and map segments to scenes.

    Returns {video_id__scene_idx: asr_text}.
    """
    from faster_whisper import WhisperModel  # heavy import, keep lazy

    logger.info("Loading faster-whisper model %s ...", WHISPER_MODEL)
    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    logger.info("Whisper model loaded.")

    results: dict[str, str] = {}
    seen_audio: dict[str, list[dict[str, Any]]] = {}  # audio_path -> segments list

    # Group videos by resolved audio path to avoid duplicate transcriptions
    audio_to_videos: dict[str, list[dict[str, Any]]] = {}
    for video_entry in shots:
        audio_key = video_entry["audio_key"]
        audio_path = _audio_path_for_video(audio_key)
        if audio_path is None:
            logger.warning("No audio found for %s (key=%s), skipping ASR", video_entry["video_id"], audio_key)
            for scene in video_entry["scenes"]:
                results[f"{video_entry['video_id']}__{scene['scene_idx']}"] = ""
            continue

        ap_str = str(audio_path)
        audio_to_videos.setdefault(ap_str, []).append(video_entry)

    total_audio = len(audio_to_videos)
    for idx, (ap_str, video_entries) in enumerate(audio_to_videos.items(), 1):
        print(f"[ASR] Transcribing {idx}/{total_audio}: {Path(ap_str).name}")

        # Transcribe once per audio file
        if ap_str not in seen_audio:
            try:
                segments_iter, _info = model.transcribe(
                    ap_str,
                    beam_size=5,
                    vad_filter=True,
                )
                seg_list = [
                    {"start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in segments_iter
                ]
                seen_audio[ap_str] = seg_list
            except Exception:
                logger.exception("Whisper transcription failed for %s", ap_str)
                seen_audio[ap_str] = []

        segments = seen_audio[ap_str]

        # Map segments to scenes for each video entry sharing this audio
        for video_entry in video_entries:
            video_id = video_entry["video_id"]
            for scene in video_entry["scenes"]:
                scene_idx: int = scene["scene_idx"]
                sc_start: float = scene["start"]
                sc_end: float = scene["end"]
                # Expand scene window to capture ASR from neighboring scenes
                asr_start = max(0.0, sc_start - ASR_OVERLAP_SEC)
                asr_end = sc_end + ASR_OVERLAP_SEC

                matched_texts: list[str] = []
                for seg in segments:
                    if _overlap_fraction(seg["start"], seg["end"], asr_start, asr_end) > 0.5:
                        text = seg["text"].strip()
                        if text:
                            matched_texts.append(text)

                results[f"{video_id}__{scene_idx}"] = " ".join(matched_texts)

    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> dict[str, dict[str, str]]:
    """Run keyframe extraction and ASR in parallel. Save extractions.json."""
    t0 = time.time()

    # Load shot boundaries
    logger.info("Loading shots from %s", SHOTS_FILE)
    with open(SHOTS_FILE, "r", encoding="utf-8") as f:
        shots: list[dict[str, Any]] = json.load(f)

    total_scenes = sum(len(v["scenes"]) for v in shots)
    logger.info("Loaded %d videos, %d scenes total.", len(shots), total_scenes)

    # Run both streams in parallel via threads.
    # Stream 1 (keyframes) is CPU/IO-bound, stream 2 (whisper) is GPU-bound —
    # threading is fine here since they don't compete for resources.
    keyframe_results: dict[str, str] = {}
    asr_results: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_kf = pool.submit(extract_all_keyframes, shots)
        future_asr = pool.submit(transcribe_all, shots)

        for future in as_completed([future_kf, future_asr]):
            try:
                result = future.result()
                if future is future_kf:
                    keyframe_results = result
                    logger.info("Keyframe extraction finished.")
                else:
                    asr_results = result
                    logger.info("ASR transcription finished.")
            except Exception:
                logger.exception("A stream failed")

    # Merge into unified output: (video_id, scene_idx) -> {keyframe_path, asr_text}
    all_keys = set(keyframe_results.keys()) | set(asr_results.keys())
    extractions: dict[str, dict[str, str]] = {}
    for key in sorted(all_keys):
        extractions[key] = {
            "keyframe_path": keyframe_results.get(key, ""),
            "asr_text": asr_results.get(key, ""),
        }

    # Save
    out_path = WORK_DIR / "extractions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(extractions, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    logger.info(
        "Step 2 done: %d extractions saved to %s (%.1fs)",
        len(extractions),
        out_path,
        elapsed,
    )
    return extractions


if __name__ == "__main__":
    main()
