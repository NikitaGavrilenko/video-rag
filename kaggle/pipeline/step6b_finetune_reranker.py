"""
step6b_finetune_reranker.py — LoRA fine-tune BGE reranker on train data.

Builds (query, positive_passage, hard_negative) triplets from train_qa.csv,
then fine-tunes bge-reranker-v2-m3 with LoRA for fast domain adaptation.

Requires: scenes.jsonl, train_qa.csv, FAISS indices (from step6).
Output: fine-tuned reranker in WORK_DIR/reranker_finetuned/

Usage:
  python -m kaggle.pipeline.step6b_finetune_reranker
  python -m kaggle.pipeline.step6b_finetune_reranker --epochs 3 --lr 2e-5
"""

from __future__ import annotations

import json
import pickle
import random
import argparse
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

from .config import (
    BGE_MODEL,
    BGE_BATCH_SIZE,
    RERANKER_MODEL,
    SCENES_FILE,
    TRAIN_CSV,
    FAISS_SCENES_INDEX,
    SCENES_META_FILE,
    WORK_DIR,
)

FINETUNED_DIR = WORK_DIR / "reranker_finetuned"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _load_scenes() -> dict[str, dict]:
    """Load scenes.jsonl into {video_id__scene_idx: doc}."""
    lookup: dict[str, dict] = {}
    with open(SCENES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            key = f"{doc['video_id']}__{doc['scene_idx']}"
            lookup[key] = doc
    return lookup


def _iou(s1: float, e1: float, s2: float, e2: float) -> float:
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def _build_triplets(
    n_hard_negatives: int = 3,
) -> list[dict[str, str]]:
    """Build training triplets from train_qa.csv + scenes.jsonl.

    For each train query:
      - positive: scene_summary of overlapping scene (IoU >= 0.5)
      - hard negatives: top FAISS results that are NOT positives
    """
    df = pd.read_csv(TRAIN_CSV)
    scenes = _load_scenes()

    # Group scenes by video
    scenes_by_video: dict[str, list[dict]] = {}
    for key, doc in scenes.items():
        vid = doc["video_id"]
        if vid not in scenes_by_video:
            scenes_by_video[vid] = []
        scenes_by_video[vid].append(doc)

    # Load FAISS for hard negative mining
    faiss_index = faiss.read_index(str(FAISS_SCENES_INDEX))
    with open(SCENES_META_FILE, "rb") as f:
        scenes_meta: list[dict] = pickle.load(f)

    # Encode queries for hard negative mining
    from FlagEmbedding import BGEM3FlagModel
    print("[finetune] Loading BGE-M3 for hard negative mining...")
    bge = BGEM3FlagModel(BGE_MODEL, use_fp16=True)

    triplets: list[dict[str, str]] = []
    skipped = 0

    # Batch encode all queries
    queries = []
    query_data = []
    for _, row in df.iterrows():
        q = str(row["question"])
        vid = str(row["video_file"])
        start = float(row["start"])
        end = float(row["end"])

        # Also use English translation if available
        q_en = str(row.get("question_en", "")) if pd.notna(row.get("question_en")) else ""

        queries.append(q)
        query_data.append({
            "question": q,
            "question_en": q_en,
            "video_id": vid,
            "start": start,
            "end": end,
        })

    print(f"[finetune] Encoding {len(queries)} train queries...")
    output = bge.encode(queries, batch_size=BGE_BATCH_SIZE, return_dense=True, return_sparse=False)
    query_vecs = output["dense_vecs"]

    # Search FAISS for each query
    print("[finetune] Mining hard negatives...")
    all_scores, all_indices = faiss_index.search(
        query_vecs.astype(np.float32), n_hard_negatives + 20
    )

    for i, qd in enumerate(query_data):
        vid = qd["video_id"]
        q_start = qd["start"]
        q_end = qd["end"]

        # Find positive scenes (IoU >= 0.5)
        positive_texts = []
        positive_keys = set()
        for scene in scenes_by_video.get(vid, []):
            if _iou(q_start, q_end, scene["start"], scene["end"]) >= 0.5:
                text = scene.get("scene_summary", "")
                if text:
                    positive_texts.append(text)
                    positive_keys.add(f"{vid}__{scene['scene_idx']}")

        if not positive_texts:
            skipped += 1
            continue

        pos_text = positive_texts[0]

        # Hard negatives: top FAISS results that are NOT positives
        hard_negs = []
        for score, idx in zip(all_scores[i], all_indices[i]):
            if idx == -1:
                continue
            meta = scenes_meta[idx]
            key = f"{meta['video_id']}__{meta['scene_idx']}"
            if key in positive_keys:
                continue
            neg_text = meta.get("summary", "")
            if neg_text:
                hard_negs.append(neg_text)
            if len(hard_negs) >= n_hard_negatives:
                break

        # Create triplets
        for neg_text in hard_negs:
            # Use original language query
            triplets.append({
                "query": qd["question"],
                "positive": pos_text,
                "negative": neg_text,
            })
            # Also use EN translation if different
            if qd["question_en"] and qd["question_en"].lower() != qd["question"].lower():
                triplets.append({
                    "query": qd["question_en"],
                    "positive": pos_text,
                    "negative": neg_text,
                })

    print(f"[finetune] Built {len(triplets)} triplets ({skipped} queries had no matching scene)")
    del bge  # free GPU memory
    torch.cuda.empty_cache()
    return triplets


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RerankerDataset(Dataset):
    """Pairwise dataset: query+positive (label=1), query+negative (label=0)."""

    def __init__(self, triplets: list[dict[str, str]], tokenizer, max_length: int = 512):
        self.pairs: list[tuple[str, str, float]] = []
        for t in triplets:
            self.pairs.append((t["query"], t["positive"], 1.0))
            self.pairs.append((t["query"], t["negative"], 0.0))
        random.shuffle(self.pairs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        q, doc, label = self.pairs[idx]
        enc = self.tokenizer(
            q, doc,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(triplets: list[dict[str, str]], args: argparse.Namespace) -> None:
    print(f"[finetune] Loading reranker model: {RERANKER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL, num_labels=1, torch_dtype=torch.bfloat16,
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query", "key", "value", "dense"],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.cuda()

    dataset = RerankerDataset(triplets, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    # Linear warmup + cosine decay
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"[finetune] Training: {len(dataset)} pairs, {args.epochs} epochs, "
          f"batch_size={args.batch_size}, lr={args.lr}")

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1}/{args.epochs}, "
                      f"step {batch_idx+1}/{len(dataloader)}, "
                      f"loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1} done, avg_loss={avg_loss:.4f}")

    # Save merged model (LoRA merged back into base)
    print(f"[finetune] Merging LoRA and saving to {FINETUNED_DIR}...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(FINETUNED_DIR))
    tokenizer.save_pretrained(str(FINETUNED_DIR))
    print(f"[finetune] Done. Fine-tuned reranker saved to {FINETUNED_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune reranker on train data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--n-hard-negatives", type=int, default=3)
    args = parser.parse_args()

    triplets = _build_triplets(n_hard_negatives=args.n_hard_negatives)

    if not triplets:
        print("[finetune] No triplets built, aborting.")
        return

    train(triplets, args)


if __name__ == "__main__":
    main()
