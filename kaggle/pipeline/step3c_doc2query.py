"""
step3c_doc2query.py — Generate synthetic search queries for each scene.

For each scene, generates 3-5 queries (RU + EN) that a user might ask
to find this scene. These queries are stored in captions.json and later
added to scene_summary for embedding, dramatically improving recall.

Output: updates captions.json in-place, adding `synthetic_queries` field
        (list of strings, mixed RU+EN).

Usage:
  PROXY_API_KEY=... python -m kaggle.pipeline.step3c_doc2query
  PROXY_API_KEY=... python -m kaggle.pipeline.step3c_doc2query --workers 20 --batch-size 10
"""

from __future__ import annotations

import json
import argparse
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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
You are a search query generator for a video search engine.
Given a video scene description and speech transcript, generate realistic search queries
that a user might type to find this specific scene.

Rules:
1. Generate exactly {n_queries} queries total: mix of Russian and English.
2. Queries should be DIVERSE: different angles, concepts, keywords.
3. Include both specific (factual) and abstract (conceptual) queries.
4. Queries should sound natural — like real user searches, not descriptions.
5. Use information ONLY from the provided description and transcript.
6. Output valid JSON only — no markdown, no extra text."""

BATCH_PROMPT_TEMPLATE = """\
Generate search queries for these {n} video scenes.
For each scene, produce {n_queries} diverse search queries (mix of Russian and English).

{scenes_block}

Output ONLY a JSON array with exactly {n} arrays of {n_queries} strings each:
[["query1", "query2", ...], ["query1", "query2", ...], ...]"""

SCENE_TEMPLATE = """\
Scene {i}:
Description: {desc}
Transcript: {asr}"""


# ── Helpers ─────────────────────────────────────────────────────────────────

def _build_client():
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
    n_queries: int,
    model: str,
    max_retries: int = 3,
) -> list[list[str]] | None:
    """Call LLM with a batch of scenes. Returns list of query lists."""
    scenes_block = "\n\n".join(
        SCENE_TEMPLATE.format(
            i=i + 1,
            desc=s.get("desc", "").strip()[:1500] or "(no description)",
            asr=s.get("asr", "").strip()[:1000] or "(no transcript)",
        )
        for i, s in enumerate(scenes)
    )

    system = SYSTEM_PROMPT.format(n_queries=n_queries)
    user_msg = BATCH_PROMPT_TEMPLATE.format(
        n=len(scenes), n_queries=n_queries, scenes_block=scenes_block,
    )

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[user_msg],
                config={
                    "system_instruction": system,
                    "temperature": 0.7,
                    "max_output_tokens": 4096,
                },
            )
            text = resp.text.strip()

            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) == len(scenes):
                # Validate structure: each element should be a list of strings
                result = []
                for item in parsed:
                    if isinstance(item, list):
                        result.append([str(q) for q in item])
                    else:
                        result.append([])
                return result

            print(f"  [warn] LLM returned {len(parsed)} items, expected {len(scenes)}. Retry {attempt+1}")
        except json.JSONDecodeError as e:
            print(f"  [warn] JSON parse error: {e}. Retry {attempt+1}")
        except Exception as e:
            print(f"  [warn] LLM call failed: {e}. Retry {attempt+1}")
            time.sleep(2 ** attempt)

    return None


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic queries per scene")
    parser.add_argument("--batch-size", type=int, default=10, help="Scenes per LLM call")
    parser.add_argument("--workers", type=int, default=10, help="Parallel LLM call workers")
    parser.add_argument("--model", type=str, default=PROXY_API_MODEL, help="LLM model name")
    parser.add_argument("--n-queries", type=int, default=4, help="Queries per scene")
    parser.add_argument("--limit", type=int, default=0, help="Max scenes to process (0=all)")
    args = parser.parse_args()

    print("[step3c] Loading captions and extractions...")
    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
        captions: dict[str, dict] = json.load(f)
    extractions: dict[str, dict] = {}
    if EXTRACTIONS_FILE.exists():
        with open(EXTRACTIONS_FILE, "r", encoding="utf-8") as f:
            extractions = json.load(f)

    # Determine which keys need processing
    keys_todo = []
    for key, cap in captions.items():
        if cap.get("synthetic_queries"):
            continue
        # Use LLM caption if available, else VLM caption
        desc = cap.get("llm_caption_en", "") or cap.get("vlm_caption", "")
        asr = extractions.get(key, {}).get("asr_text", "") if isinstance(extractions.get(key), dict) else ""
        if not desc and not asr:
            continue
        keys_todo.append(key)

    if args.limit > 0:
        keys_todo = keys_todo[: args.limit]

    total = len(keys_todo)
    print(f"[step3c] {total} scenes to process ({len(captions) - total} already done or empty)")

    if total == 0:
        print("[step3c] Nothing to do.")
        return

    client = _build_client()

    batches: list[list[str]] = []
    for i in range(0, total, args.batch_size):
        batches.append(keys_todo[i : i + args.batch_size])

    processed = 0
    failed = 0
    save_every = 50
    t0 = time.time()

    def _process_batch(batch_keys: list[str]) -> list[tuple[str, list[str] | None]]:
        scenes_input = []
        for key in batch_keys:
            cap = captions.get(key, {})
            ext = extractions.get(key, {})
            desc = cap.get("llm_caption_en", "") or cap.get("vlm_caption", "")
            asr = ext.get("asr_text", "") if isinstance(ext, dict) else ""
            scenes_input.append({"desc": desc, "asr": asr})
        results = _call_llm(client, scenes_input, args.n_queries, model=args.model)
        if results is None:
            return [(k, None) for k in batch_keys]
        return list(zip(batch_keys, results))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_batch, batch): batch for batch in batches}

        for i, future in enumerate(as_completed(futures)):
            results = future.result()
            for key, result in results:
                if result is None:
                    failed += 1
                    continue
                captions[key]["synthetic_queries"] = result
                processed += 1

            done_batches = i + 1
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed - failed) / rate if rate > 0 else 0
            print(
                f"  [{done_batches}/{len(batches)}] "
                f"{processed} done, {failed} failed, "
                f"{rate:.1f} scenes/s, ETA {eta:.0f}s"
            )

            if done_batches % save_every == 0:
                _save(captions)

    _save(captions)
    elapsed = time.time() - t0
    print(f"[step3c] Done: {processed} scenes, {failed} failed in {elapsed:.1f}s")


def _save(captions: dict) -> None:
    with open(CAPTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=1)
    print(f"  [checkpoint] Saved {CAPTIONS_FILE}")


if __name__ == "__main__":
    main()
