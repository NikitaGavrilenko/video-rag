"""
step3b_fix_captions.py — LLM post-processing of VLM captions.

Reads captions.json + extractions.json, calls Gemini (via ProxyAPI)
to produce improved bilingual (EN+RU) descriptions that are:
  - grounded in VLM visual description + ASR transcript (no hallucinations)
  - focused on concepts/topics/techniques (query-friendly, not just visual)
  - concise (1-3 sentences per language)

Output: updates captions.json in-place, adding `llm_caption_en` and `llm_caption_ru` fields.

Usage:
  PROXY_API_KEY=... python -m kaggle.pipeline.step3b_fix_captions
  PROXY_API_KEY=... python -m kaggle.pipeline.step3b_fix_captions --batch-size 5 --workers 4
"""

from __future__ import annotations

import json
import argparse
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config import (
    PROXY_API_BASE,
    PROXY_API_KEY,
    PROXY_API_MODEL,
    WORK_DIR,
)

CAPTIONS_FILE = WORK_DIR / "captions.json"
EXTRACTIONS_FILE = WORK_DIR / "extractions.json"

# ── LLM prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a video scene description improver for a search index.
Your goal: rewrite scene descriptions so they are easy to find via semantic search queries.

Rules:
1. ONLY use information from the provided visual description and speech transcript.
2. DO NOT hallucinate or invent details not present in the sources.
3. Focus on WHAT is being demonstrated, taught, discussed, or shown — not just visual appearance.
4. Include key concepts, techniques, topics, and domain terms.
5. Be concise: 1-3 sentences per language.
6. Output valid JSON only — no markdown, no extra text."""

BATCH_PROMPT_TEMPLATE = """\
Process these {n} video scene descriptions. For each, produce an improved EN and RU description.

{scenes_block}

Output ONLY a JSON array with exactly {n} objects:
[{{"en": "...", "ru": "..."}}, ...]"""

SCENE_TEMPLATE = """\
Scene {i}:
Visual: {vlm}
Transcript: {asr}"""


# ── Helpers ─────────────────────────────────────────────────────────────────

def _build_client():
    """Create Google GenAI client for the proxy API."""
    from google import genai
    if not PROXY_API_KEY:
        raise ValueError(
            "PROXY_API_KEY not set. Export it:\n"
            "  PowerShell: $env:PROXY_API_KEY='your_key'\n"
            "  Bash: export PROXY_API_KEY=your_key"
        )
    return genai.Client(
        api_key=PROXY_API_KEY,
        http_options={"base_url": PROXY_API_BASE},
    )


def _call_llm(
    client,
    scenes: list[dict[str, str]],
    model: str = PROXY_API_MODEL,
    max_retries: int = 3,
) -> list[dict[str, str]] | None:
    """Call LLM with a batch of scenes. Returns list of {en, ru} dicts."""
    scenes_block = "\n\n".join(
        SCENE_TEMPLATE.format(
            i=i + 1,
            vlm=s.get("vlm", "").strip()[:1500] or "(no visual description)",
            asr=s.get("asr", "").strip()[:1000] or "(no transcript)",
        )
        for i, s in enumerate(scenes)
    )

    user_msg = BATCH_PROMPT_TEMPLATE.format(n=len(scenes), scenes_block=scenes_block)

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[user_msg],
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.3,
                    "max_output_tokens": 2048,
                },
            )
            text = resp.text.strip()

            # Strip markdown fences if present
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) == len(scenes):
                return parsed

            print(f"  [warn] LLM returned {len(parsed)} items, expected {len(scenes)}. Retry {attempt+1}")
        except json.JSONDecodeError as e:
            print(f"  [warn] JSON parse error: {e}. Retry {attempt+1}")
        except Exception as e:
            print(f"  [warn] LLM call failed: {e}. Retry {attempt+1}")
            time.sleep(2 ** attempt)

    return None


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM post-processing of VLM captions")
    parser.add_argument("--batch-size", type=int, default=5, help="Scenes per LLM call")
    parser.add_argument("--workers", type=int, default=4, help="Parallel LLM call workers")
    parser.add_argument("--model", type=str, default=PROXY_API_MODEL, help="LLM model name")
    parser.add_argument("--limit", type=int, default=0, help="Max scenes to process (0=all)")
    args = parser.parse_args()

    # Load data
    print("[step3b] Loading captions and extractions...")
    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
        captions: dict[str, dict] = json.load(f)
    extractions: dict[str, dict] = {}
    if EXTRACTIONS_FILE.exists():
        with open(EXTRACTIONS_FILE, "r", encoding="utf-8") as f:
            extractions = json.load(f)

    # Determine which keys need processing
    keys_todo = []
    for key, cap in captions.items():
        if cap.get("llm_caption_en"):
            continue  # already processed
        vlm = cap.get("vlm_caption", "")
        asr = extractions.get(key, {}).get("asr_text", "") if isinstance(extractions.get(key), dict) else ""
        if not vlm and not asr:
            continue  # nothing to improve
        keys_todo.append(key)

    if args.limit > 0:
        keys_todo = keys_todo[: args.limit]

    total = len(keys_todo)
    print(f"[step3b] {total} scenes to process ({len(captions) - total} already done or empty)")

    if total == 0:
        print("[step3b] Nothing to do.")
        return

    client = _build_client()

    # Build batches
    batches: list[list[str]] = []
    for i in range(0, total, args.batch_size):
        batches.append(keys_todo[i : i + args.batch_size])

    processed = 0
    failed = 0
    save_every = 50  # save checkpoint every N batches
    t0 = time.time()

    def _process_batch(batch_keys: list[str]) -> list[tuple[str, dict[str, str] | None]]:
        scenes_input = []
        for key in batch_keys:
            cap = captions.get(key, {})
            ext = extractions.get(key, {})
            scenes_input.append({
                "vlm": cap.get("vlm_caption", ""),
                "asr": ext.get("asr_text", "") if isinstance(ext, dict) else "",
            })
        results = _call_llm(client, scenes_input, model=args.model)
        if results is None:
            return [(k, None) for k in batch_keys]
        return list(zip(batch_keys, results))

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_batch, batch): batch for batch in batches}

        for i, future in enumerate(as_completed(futures)):
            results = future.result()
            for key, result in results:
                if result is None:
                    failed += 1
                    continue
                captions[key]["llm_caption_en"] = result.get("en", "")
                captions[key]["llm_caption_ru"] = result.get("ru", "")
                processed += 1

            # Progress
            done_batches = i + 1
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed - failed) / rate if rate > 0 else 0
            print(
                f"  [{done_batches}/{len(batches)}] "
                f"{processed} done, {failed} failed, "
                f"{rate:.1f} scenes/s, ETA {eta:.0f}s"
            )

            # Periodic checkpoint
            if done_batches % save_every == 0:
                _save_captions(captions)

    _save_captions(captions)
    elapsed = time.time() - t0
    print(f"[step3b] Done: {processed} improved, {failed} failed in {elapsed:.1f}s")


def _save_captions(captions: dict) -> None:
    with open(CAPTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=1)
    print(f"  [checkpoint] Saved {CAPTIONS_FILE}")


if __name__ == "__main__":
    main()
