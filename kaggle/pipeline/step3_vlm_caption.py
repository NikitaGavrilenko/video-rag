"""
step3_vlm_caption.py — Generate multimodal captions using Qwen3-VL-8B via vLLM.

Reads keyframes + ASR text from extractions.json, produces captions.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

from .config import (
    VLM_BATCH_SIZE,
    VLM_MAX_TOKENS,
    VLM_MODEL,
    VLM_PROMPT_TEMPLATE,
    VLM_TENSOR_PARALLEL,
    WORK_DIR,
)

CAPTIONS_FILE = WORK_DIR / "captions.json"
EXTRACTIONS_FILE = WORK_DIR / "extractions.json"

SIMPLE_PROMPT = (
    "Describe what is happening in this video scene based on the visual content. "
    "2-4 sentences in English."
)


def _build_multimodal_input(image: Image.Image, text_prompt: str) -> dict:
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


def main() -> None:
    # ── Load inputs ──────────────────────────────────────────────────────
    with open(EXTRACTIONS_FILE, "r", encoding="utf-8") as f:
        extractions: dict[str, dict] = json.load(f)

    # ── Resume support ───────────────────────────────────────────────────
    captions: dict[str, dict] = {}
    if CAPTIONS_FILE.exists():
        with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
            captions = json.load(f)
        print(f"[step3] Resumed {len(captions)} existing captions.")

    # ── Collect scenes to process ────────────────────────────────────────
    pending_keys: list[str] = []
    for key in extractions:
        if key in captions:
            continue
        ext = extractions[key]
        keyframe_path = ext.get("keyframe_path", "")
        if not keyframe_path or not Path(keyframe_path).exists():
            # No keyframe — skip, record empty caption
            captions[key] = {"vlm_caption": ""}
            continue
        pending_keys.append(key)

    print(f"[step3] {len(pending_keys)} scenes to caption ({len(captions)} already done).")
    if not pending_keys:
        _save(captions)
        return

    # ── Init vLLM ────────────────────────────────────────────────────────
    llm = LLM(
        model=VLM_MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        tensor_parallel_size=VLM_TENSOR_PARALLEL,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.85,
    )
    sampling_params = SamplingParams(max_tokens=VLM_MAX_TOKENS, temperature=0.3)

    # ── Batch processing ─────────────────────────────────────────────────
    for batch_start in tqdm(
        range(0, len(pending_keys), VLM_BATCH_SIZE),
        desc="VLM batches",
    ):
        batch_keys = pending_keys[batch_start : batch_start + VLM_BATCH_SIZE]
        inputs: list[dict] = []

        for key in batch_keys:
            ext = extractions[key]
            image = Image.open(ext["keyframe_path"]).convert("RGB")
            asr_text = ext.get("asr_text", "").strip()

            if asr_text:
                text_prompt = VLM_PROMPT_TEMPLATE.format(asr_text=asr_text)
            else:
                text_prompt = SIMPLE_PROMPT

            inputs.append(_build_multimodal_input(image, text_prompt))

        outputs = llm.generate(inputs, sampling_params=sampling_params)

        for key, output in zip(batch_keys, outputs):
            caption_text = output.outputs[0].text.strip()
            captions[key] = {"vlm_caption": caption_text}

        # Checkpoint after each batch
        _save(captions)

    print(f"[step3] Done. {len(captions)} total captions saved to {CAPTIONS_FILE}")


def _save(captions: dict[str, dict]) -> None:
    with open(CAPTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
