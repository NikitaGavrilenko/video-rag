"""
run_pipeline.py — Sequential orchestrator for the Video RAG pipeline.

Steps:
  1. Shot detection          (step1_shots)
  2. Parallel extraction     (step2_extract: keyframes + ASR)
  3. VLM captioning          (step3_vlm_caption)
  4. Scene documents         (step4_scene_docs)
  5. Event documents         (step5_event_docs)
  6. Indexing                (step6_index: BGE-M3 + FAISS + sparse)
  6b. Reranker fine-tune     (step6b_finetune_reranker, optional)
  7. Search / submission     (search)

Usage:
  python -m kaggle.pipeline.run_pipeline
  python -m kaggle.pipeline.run_pipeline --skip-shots --skip-extract
  python -m kaggle.pipeline.run_pipeline --search-only
  python -m kaggle.pipeline.run_pipeline --stream   # streaming producer-consumer (steps 2+3)
"""

from __future__ import annotations

import argparse
import time
from typing import Callable


def _run_step(name: str, func: Callable[[], object]) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    func()
    print(f"[timing] {name} completed in {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Video RAG pipeline orchestrator")
    parser.add_argument("--skip-shots",   action="store_true", help="Skip step 1 (shot detection)")
    parser.add_argument("--skip-extract", action="store_true", help="Skip step 2 (keyframes + ASR)")
    parser.add_argument("--skip-vlm",     action="store_true", help="Skip step 3 (VLM captioning)")
    parser.add_argument("--skip-index",   action="store_true", help="Skip step 6 (indexing)")
    parser.add_argument("--finetune",     action="store_true", help="Run step 6b: LoRA fine-tune reranker")
    parser.add_argument("--search-only",  action="store_true", help="Only run search (skip all offline steps)")
    parser.add_argument("--stream",       action="store_true", help="Streaming pipeline: steps 2+3 merged")
    args = parser.parse_args()

    pipeline_t0 = time.time()

    if not args.search_only:
        if not args.skip_shots:
            from .step1_shots import main as step1_main
            _run_step("Step 1: Shot detection", step1_main)
        else:
            print("\n[skip] Step 1: Shot detection")

        if args.stream:
            if not args.skip_extract:
                from .step2_3_stream import main as stream_main
                _run_step("Steps 2+3: Streaming extraction + VLM captioning", stream_main)
            else:
                print("\n[skip] Steps 2+3: Streaming extraction + VLM captioning")
        else:
            if not args.skip_extract:
                from .step2_extract import main as step2_main
                _run_step("Step 2: Parallel extraction (keyframes + ASR)", step2_main)
            else:
                print("\n[skip] Step 2: Parallel extraction")

            if not args.skip_vlm:
                from .step3_vlm_caption import main as step3_main
                _run_step("Step 3: VLM captioning", step3_main)
            else:
                print("\n[skip] Step 3: VLM captioning")

        from .step4_scene_docs import main as step4_main
        _run_step("Step 4: Scene documents", step4_main)

        from .step5_event_docs import main as step5_main
        _run_step("Step 5: Event documents", step5_main)

        if not args.skip_index:
            from .step6_index import main as step6_main
            _run_step("Step 6: FAISS indexing", step6_main)
        else:
            print("\n[skip] Step 6: FAISS indexing")

        if args.finetune:
            from .step6b_finetune_reranker import main as step6b_main
            _run_step("Step 6b: LoRA fine-tune reranker", step6b_main)

    from .search import main as search_main
    _run_step("Step 7: Search & submission generation", search_main)

    print(f"\n{'='*60}")
    print(f"  Pipeline finished in {time.time() - pipeline_t0:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()