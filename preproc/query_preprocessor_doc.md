# Query Preprocessor — Technical Documentation

## Overview

Three-stage query cleaning pipeline for multilingual video fragment retrieval. Corrects typos, normalizes encoding artifacts, and transliterates foreign scripts before queries enter the RAG pipeline.

**Input:** raw user query (Russian or English, may contain typos, mixed scripts, Arabic/Chinese characters)

**Output:** cleaned query optimized for embedding and retrieval

## Architecture

```
User query
    │
    ▼
┌──────────────────────────────┐
│  Stage 1: Rule-based         │  EN + RU   < 0.1ms   CPU
│  - Mid-word caps fix         │
│  - Latin↔Cyrillic homoglyphs │
│  - Merged word splitting     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stage 2: SymSpell           │  EN only   < 0.2ms   CPU
│  - Edit-distance correction  │
│  - Domain whitelist protect  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stage 3: SAGE T5 (optional) │  RU only   ~18ms     GPU
│  - Neural GEC correction     │
│  - Post-clean: «» restore,  │
│    zero-width space removal  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Transliteration             │  All       < 0.01ms  CPU
│  - Arabic → phonetic EN/RU   │
│  - Chinese → pinyin           │
│  - Appends to original query │
└──────────────┬───────────────┘
               │
               ▼
         Clean query
```

## Performance

| Configuration | Queries fixed | Latency | GPU | Use case |
|---|---|---|---|---|
| Rules + SymSpell | 293 / 812 (36%) | 0.23 ms | 0 | Production default |
| Rules + SymSpell + SAGE | 491 / 812 (60%) | 18 ms | T4 | Maximum quality |

**Throughput at 300 RPS:**

- Rules + SymSpell: 0.07 CPU cores (single thread handles it)
- With SAGE (50% RU traffic): +5.4 T4 GPU instances (separate microservice)

## Stage details

### Stage 1: Rule-based normalization

Deterministic regex transformations. Zero dependencies, zero false positives.

**Mid-word uppercase fix:**
Detects random capitalization injected as noise (e.g., competition data augmentation). Preserves acronyms (ASL, USA, FDM) by checking if adjacent characters are also uppercase.

| Input | Output |
|---|---|
| пользОваться | пользоваться |
| climactIc | climactic |
| beatBox → | beatbox (but ASL → ASL) |

**Latin↔Cyrillic homoglyph fix:**
Some characters look identical across scripts (Latin `o` = Cyrillic `о`). If a word is mostly Cyrillic but contains a Latin character (or vice versa), the minority characters are converted.

| Input | Output |
|---|---|
| транспортомo (Latin `o`) | транспортомо (Cyrillic `о`) |

**Merged word splitting:**
Inserts a space before an uppercase letter that follows a lowercase letter mid-word.

| Input | Output |
|---|---|
| чередованияЭнергии | чередования Энергии |

### Stage 2: SymSpell (English only)

Uses the SymSpell algorithm (pre-computed delete edits, O(1) lookup) with a built-in 82K-word English frequency dictionary.

**Why English only:** Russian morphology (6 cases, 3 genders, verb conjugations) makes edit-distance correction unreliable without a massive dictionary. Words like `транспортом` get "corrected" to `транспорт`, `Турции` to `Турнир`. SymSpell is only applied to tokens matching `^[a-zA-Z]{3,}$`.

**Domain whitelist:** Built automatically from `train_qa.csv` at initialization (~9K terms from question/answer text). Extended with a manual list of ~80 domain-specific terms (beatbox, Emirati, tanween, Ender, etc.). Any whitelisted word is skipped entirely.

| Without whitelist | With whitelist |
|---|---|
| beatbox → gearbox | beatbox ✓ |
| Emirati → emirate | Emirati ✓ |
| tanween → canteen | tanween ✓ |
| apps → apes | apps ✓ |
| ABCs → abs | ABCs ✓ |

### Stage 3: SAGE T5 (Russian, optional)

Model: `ai-forever/sage-fredt5-distilled-95m` (95M params, T5-based, trained on Russian grammatical error correction).

Handles typos that edit-distance methods cannot resolve for Russian:

| Input | Output |
|---|---|
| конструпции | конструкции |
| псолдовательности | последовательности |
| доогие | долгие |
| чередованияэнергии | чередования энергии |
| появлящюееся | появляющееся |

**Post-SAGE cleanup (`_post_sage_clean`):**

SAGE sometimes replaces `«»` with `""` and introduces zero-width Unicode spaces. The cleanup function:
1. Restores `«»` guillemets if the original query used them
2. Strips zero-width spaces (U+200B–U+200F, U+FEFF, U+00A0, U+2060)
3. Normalizes whitespace

**Deployment:** Runs as a separate GPU microservice to avoid competing with the RAG pipeline (embedding, reranker) for VRAM. Can be made async — serve the SymSpell-corrected result immediately, refine with SAGE if it changes the query.

### Foreign script transliteration

Arabic and Chinese characters in queries are transliterated to their phonetic English (or Russian) names and **appended** to the original query. Both forms are present so the embedding model can match either.

| Input | Output |
|---|---|
| My ق weakens like ك | My ق weakens like ك  My qaf weakens like kaf |
| звуки «ع» или «غ» | звуки «ع» или «غ»  звуки «айн» или «гайн» |
| используя 幹嘛 | используя 幹嘛  используя gan ma |

Phrase-level matches are applied first (longest match), then character-level for remaining individual letters. Arabic diacritics (harakat) are stripped before matching.

## Usage

```python
from query_preprocessor import QueryPreprocessor

# Production (Rules + SymSpell, no GPU)
qp = QueryPreprocessor('train_qa.csv')

# With SAGE (Rules + SymSpell + SAGE, needs GPU)
qp = QueryPreprocessor('train_qa.csv', use_sage=True, sage_device='cuda')

# Single query
clean = qp("climactIc narratvie seqence in ASL")

# Batch (SAGE correction is batched for efficiency)
results = qp.batch(list_of_queries)
```

## Dependencies

| Package | Required for | Install |
|---|---|---|
| pandas | Whitelist generation from CSV | `pip install pandas` |
| symspellpy | Stage 2 (EN correction) | `pip install symspellpy` |
| transformers | Stage 3 (SAGE) | `pip install transformers` |
| torch | Stage 3 (SAGE) | `pip install torch` |
| sentencepiece | Stage 3 (SAGE tokenizer) | `pip install sentencepiece` |

Stages degrade gracefully: if `symspellpy` is not installed, only rules + transliteration run. If `transformers` is not installed, SAGE is skipped.
