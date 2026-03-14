"""
retry_failed.py — Re-extract failed keyframes (CPU) and re-caption with VLM.

Usage:
    python -m kaggle.pipeline.retry_failed

Finds scenes with empty keyframe_path or vlm_caption, re-extracts keyframes
using CPU-only ffmpeg, and runs VLM on them.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

from .config import (
    KEYFRAMES_DIR,
    SHOTS_FILE,
    VLM_BATCH_SIZE,
    VLM_MAX_TOKENS,
    VLM_MODEL,
    VLM_PROMPT_TEMPLATE,
    VLM_TENSOR_PARALLEL,
    WORK_DIR,
)

EXTRACTIONS_FILE = WORK_DIR / "extractions.json"
CAPTIONS_FILE = WORK_DIR / "captions.json"

SIMPLE_PROMPT = (
    "Describe what is happening in this video scene based on the visual content. "
    "2-4 sentences in English."
)


def _extract_keyframe_cpu(video_path: Path, keyframe_time: float, out_path: Path, max_side: int = 768) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scale_filter = f"scale='if(gte(iw,ih),{max_side},-2)':'if(gte(iw,ih),-2,{max_side})'"
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(keyframe_time),
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", scale_filter,
        "-q:v", "2",
        str(out_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    return result.returncode == 0


def _build_multimodal_input(image: Image.Image, text_prompt: str) -> dict:
    prompt = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{text_prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {"prompt": prompt, "multi_modal_data": {"image": image}}


def main() -> None:
    with open(SHOTS_FILE, "r", encoding="utf-8") as f:
        shots: list[dict] = json.load(f)
    with open(EXTRACTIONS_FILE, "r", encoding="utf-8") as f:
        extractions: dict[str, dict] = json.load(f)
    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
        captions: dict[str, dict] = json.load(f)

    # Build video_id → video_path mapping
    video_paths: dict[str, Path] = {}
    scene_map: dict[str, dict] = {}
    for video_entry in shots:
        vid = video_entry["video_id"]
        audio_key = video_entry["audio_key"]
        video_paths[vid] = Path("/kaggle/input/competitions/multi-lingual-video-fragment-retrieval-challenge/video-rag") / audio_key
        for scene in video_entry["scenes"]:
            key = f"{vid}__{scene['scene_idx']}"
            scene_map[key] = scene

    # Find failed scenes
    failed_keys: list[str] = []
    for key, cap in captions.items():
        if not cap.get("vlm_caption", ""):
            ext = extractions.get(key, {})
            kf = ext.get("keyframe_path", "")
            if not kf or not Path(kf).exists():
                failed_keys.append(key)

    print(f"[retry] Found {len(failed_keys)} scenes with empty caption + missing keyframe")
    if not failed_keys:
        print("[retry] Nothing to retry.")
        return

    # Re-extract keyframes
    fixed = 0
    for key in failed_keys:
        vid, scene_idx_str = key.split("__")
        scene_idx = int(scene_idx_str)
        scene = scene_map.get(key)
        if not scene:
            continue

        video_path = video_paths.get(vid)
        if not video_path or not video_path.exists():
            continue

        out_path = KEYFRAMES_DIR / vid / f"scene_{scene_idx:04d}.jpg"
        if out_path.exists() or _extract_keyframe_cpu(video_path, scene["keyframe_time"], out_path):
            extractions[key]["keyframe_path"] = str(out_path)
            fixed += 1

    print(f"[retry] Re-extracted {fixed}/{len(failed_keys)} keyframes")

    # Save updated extractions
    with open(EXTRACTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(extractions, f, ensure_ascii=False, indent=2)

    # Collect scenes to re-caption
    pending: list[str] = []
    for key in failed_keys:
        kf = extractions.get(key, {}).get("keyframe_path", "")
        if kf and Path(kf).exists():
            pending.append(key)

    print(f"[retry] {len(pending)} scenes to re-caption with VLM")
    if not pending:
        return

    # Init VLM
    llm = LLM(
        model=VLM_MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        tensor_parallel_size=VLM_TENSOR_PARALLEL,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(max_tokens=VLM_MAX_TOKENS, temperature=0.3)

    # Batch process
    for i in range(0, len(pending), VLM_BATCH_SIZE):
        batch_keys = pending[i:i + VLM_BATCH_SIZE]
        inputs = []
        valid_keys = []

        for key in batch_keys:
            ext = extractions[key]
            kf = ext.get("keyframe_path", "")
            if not kf or not Path(kf).exists():
                continue
            image = Image.open(kf).convert("RGB")
            asr_text = ext.get("asr_text", "").strip()[:2000]
            if asr_text:
                text_prompt = VLM_PROMPT_TEMPLATE.format(asr_text=asr_text)
            else:
                text_prompt = SIMPLE_PROMPT
            inputs.append(_build_multimodal_input(image, text_prompt))
            valid_keys.append(key)

        if inputs:
            outputs = llm.generate(inputs, sampling_params=sampling_params)
            for key, output in zip(valid_keys, outputs):
                captions[key] = {"vlm_caption": output.outputs[0].text.strip()}

        print(f"[retry] Captioned {min(i + VLM_BATCH_SIZE, len(pending))}/{len(pending)}")

    with open(CAPTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=1)

    print(f"[retry] Done. Updated {len(pending)} captions.")


if __name__ == "__main__":
    main()
