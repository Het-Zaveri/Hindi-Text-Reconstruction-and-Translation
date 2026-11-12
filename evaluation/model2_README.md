# Model 2: Hindi Sentence Ordering (Experimental)

This directory describes the experimental zero-shot sentence ordering pipeline built on top of Model 1.

---

## Overview

| Item | Detail |
|---|---|
| **Model** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Task** | Reorder shuffled Hindi sentences into their most coherent reading sequence |
| **Method** | Zero-shot cosine-similarity scoring over multilingual sentence embeddings |
| **Quality** | EXPERIMENTAL — no gold ordering references; no accuracy percentage claimed |
| **Device** | CPU by default; CUDA supported if available |

> [!IMPORTANT]
> This is a **zero-shot experimental method**. The ordering is determined by the internal cosine similarity score between adjacent sentence embeddings, **not** by any trained ordering classifier or human-verified ground truth. The similarity score reflects embedding-space coherence only. No accuracy against a gold ordering is reported.

---

## Pipeline Architecture

```
Raw Hindi .txt  →  [Model 1: Punctuation Restoration]  →  Punctuated text
                →  [Model 2: Sentence Segmentation]    →  Sentence list
                →  [MiniLM Encoder]                    →  Embeddings
                →  [Exact/Beam Search Ordering]        →  Ordered sequence
                →  [Integrity Verification]            →  Final output
```

### Search Strategy

| Input Size | Method | Description |
|---|---|---|
| ≤ 8 sentences | **Exact** | Exhaustive permutation search, guaranteed optimal |
| > 8 sentences | **Beam search** | Width-64 greedy beam, fast and near-optimal |

The transition threshold and beam width are configurable constants in `src/model2_ordering.py`.

---

## Directory Structure

```text
outputs/
├── model2_ordering/
│   ├── sample-N_ordered.txt           # Ordered Hindi text (original strings, unmodified)
│   ├── sample-N_sentence_mapping.jsonl # Per-sentence mapping: original → ordered position
│   └── model2_run_report.jsonl        # Per-file run metadata
└── pipeline_final/
    ├── sample-N_pipeline_output.txt   # Final combined pipeline output (Model 1 + 2)
    └── pipeline_report.jsonl          # Combined pipeline metadata for all 7 files
```

---

## Running Model 2 Standalone

```bash
# Order all 7 files using existing Model 1 outputs
python src/model2_ordering.py

# Order a single file
python src/model2_ordering.py --input outputs/sample-1_punctuated.txt --source data/sample-1.txt
```

## Running the Full Pipeline (Model 1 → Model 2)

```bash
# Use existing Model 1 punctuated outputs (recommended — avoids re-downloading Model 1)
python src/pipeline.py --skip-punctuation

# Full fresh run (downloads and runs both models)
python src/pipeline.py

# Single file
python src/pipeline.py --file data/sample-1.txt --skip-punctuation
```

---

## Sentence Integrity Guarantee

After ordering, every output is verified:
- **No sentences missing** — all original sentences appear in the output.
- **No sentences duplicated** — each sentence appears exactly once.
- **No sentences modified** — original strings are preserved character-for-character.

The integrity check is enforced programmatically and reported per file. Any failure raises a warning and is logged in the run report.

---

## Output Record Format

### `sample-N_sentence_mapping.jsonl`
```json
{
  "sentence_id": "sample-1_S001",
  "original_position": 3,
  "original_sentence": "भगवान राम के पूर्वज महाराज शिवि ने...",
  "ordered_sentence": "भगवान राम के पूर्वज महाराज शिवि ने..."
}
```

### `model2_run_report.jsonl`
```json
{
  "source_file": "sample-1.txt",
  "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "sentence_count": 47,
  "ordering_score": 32.4871,
  "search_method": "beam",
  "runtime_seconds": 4.21,
  "integrity": {"passed": true, "original_count": 47, "reordered_count": 47, ...},
  "quality_note": "EXPERIMENTAL zero-shot ordering. No gold reference ordering exists..."
}
```

---

## Running Unit Tests

```bash
# Model 2 tests only (30 tests)
python -m unittest evaluation/test_model2.py -v

# All tests (Model 1 + Model 2)
python -m unittest discover -s evaluation -p "test_*.py" -v
```

---

## Limitations

1. **No ground truth ordering** — the original files are continuous prose, not explicitly shuffled. The model orders sentences as if they were shuffled, but there is no reference to validate against.
2. **Embedding space ≠ discourse coherence** — MiniLM similarity captures semantic relatedness, not narrative progression. Ordering by adjacency coherence is a heuristic, not a linguistic guarantee.
3. **Large documents** — files with many sentences (> 8) use beam search, which may not find the global optimum. Increasing `BEAM_WIDTH` in `model2_ordering.py` improves quality at the cost of runtime.
4. **Domain sensitivity** — the model is multilingual and not fine-tuned for Hindi. Performance may differ across academic prose, conversational dialogue, and agricultural/scientific text.
