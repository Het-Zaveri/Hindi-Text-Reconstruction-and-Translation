# Model 1 Evaluation: Hindi Punctuation Restoration

This directory contains the complete, reproducible evaluation pipeline for **Model 1** (`Horizon-Labs/punctuation-restoration-small`) in the Hindi Text Reconstruction and Translation project.

---

## Directory Structure

```text
evaluation/
├── dataset/
│   ├── reference_dataset.jsonl    # Primary ground truth dataset (55 curated segments)
│   └── reference_dataset.csv      # CSV representation of the ground truth dataset
├── predictions/
│   └── model1_predictions.jsonl   # Exact model output for all reference samples
├── reports/
│   ├── metrics.json               # Structured evaluation metrics and metadata
│   ├── classification_report.txt  # Human-readable precision, recall, F1-score table
│   ├── confusion_matrix.csv       # Confusion matrix over punctuation marks and words
│   └── evaluation_summary.md      # Detailed qualitative and quantitative evaluation summary
├── evaluate.py                    # Standalone evaluation and inference execution script
├── test_pipeline.py               # Unit and regression test suite for evaluation components
└── README.md                      # This documentation file
```

---

## 1. Model Overview

- **Model ID**: `Horizon-Labs/punctuation-restoration-small`
- **Hugging Face Repository**: [https://huggingface.co/Horizon-Labs/punctuation-restoration-small](https://huggingface.co/Horizon-Labs/punctuation-restoration-small)
- **Architecture**: `AutoModelForTokenClassification` (~140 Million parameters)
- **Primary Mechanism**: Predicts punctuation tokens at subword token boundaries.
- **Supported Punctuation Labels**:
  - `0`: No punctuation (`_NONE_`)
  - `.`: Standard sentence-terminal mark (mapped to Hindi Danda `।` for Devanagari script `[ऀ-৿]`)
  - `,`: Comma (Alpa Viram `,`)
  - `?`: Question Mark (Prashna Chinha `?`)
  - `-`: Hyphen (`-`)
  - `:``: Colon (`:`)
- **Long Text Processing**: Overlapping chunk sliding window with `chunk_tokens=400` and `overlap_tokens=50`.

---

## 2. Ground Truth Reference Dataset

### Creation Methodology
- **Source Material**: 55 complete, grammatical segments extracted verbatim across all seven Hindi raw files (`sample-1.txt` to `sample-7.txt`).
- **Diversity of Domains**:
  - `sample-1.txt` & `sample-2.txt`: Religious pilgrimage, cultural history, temple architecture, Ganga Mela.
  - `sample-3.txt`: Jain Tirthankara history, Hastinapur geography, transit information.
  - `sample-4.txt` & `sample-5.txt`: Agricultural biology, Bt cotton seeds, sericulture, biochemistry (fibroin, sericin).
  - `sample-6.txt` & `sample-7.txt`: Conversational dialogues, vocative addressing, tag questions, humorous discourse markers.
- **Strict Word Integrity**: All original words from the source files are preserved exactly. Missing-data placeholders (`NULL`) are not treated as words.
- **Independent Annotation**: Reference annotations were created using Devanagari grammatical rules and discourse semantics prior to comparing against model outputs.

### Record Schema
Each JSONL record has the following structure:
```json
{
  "sample_id": "S001",
  "source_file": "sample-1.txt",
  "input_text": "भगवान राम के पूर्वज महाराज शिवि ने अपना वानप्रस्थ गढ़ मुक्तेश्वर में ही व्यतीत किया था",
  "reference_text": "भगवान राम के पूर्वज महाराज शिवि ने अपना वानप्रस्थ गढ़ मुक्तेश्वर में ही व्यतीत किया था।",
  "confidence": "high",
  "annotation_notes": "Declarative historical statement. Terminal purna viram (।)."
}
```

### Confidence Annotation
- **High Confidence (48 samples / 87.3%)**: Clear, unambiguous syntactic structures (e.g. standard declarative statements ending with `।`, clear direct questions with `?`, and imperative commands).
- **Medium Confidence (7 samples / 12.7%)**: Segments with stylistic or discourse punctuation ambiguity (e.g., vocatives with optional commas, parenthetical relative clauses, numerical ranges, or conversational dialogue where intonation defines query vs. statement).

> **Annotation Notice**: This initial reference dataset was created with AI assistance and linguistic verification. Before establishing this as an authoritative, gold-standard benchmark, formal validation by native Hindi speakers is required.

---

## 3. Evaluation Methodology

Rather than relying on naive character-level or string edit distance, punctuation is evaluated at **word boundaries**:

1. **Word-Punctuation Extraction**: Each text is parsed into `(word, punctuation)` pairs. Standalone punctuation marks attach to their preceding word. Punctuation attached to words is recognized and isolated.
2. **Word Sequence Alignment**: Uses `difflib.SequenceMatcher` on base words to align reference and prediction tokens. This guarantees that words are aligned correctly even in the presence of token variations.
3. **Punctuation Classification**:
   - Each aligned word boundary is assigned a true label ($y$) and predicted label ($\hat{y}$) from $\{\text{।}, \text{,}, \text{?}, \text{-}, \text{:}, \text{_NONE_}\}$.
   - **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$
   - **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
   - **F1-Score**: $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
   - **Micro Metrics**: Aggregated across all punctuation classes (excluding `_NONE_`).
   - **Macro Metrics**: Unweighted mean of F1 scores across all punctuation classes with support $> 0$.
   - **Exact Sequence Accuracy**: Fraction of sentences where the entire sequence of punctuation marks matches the reference exactly.
   - **Word Preservation Accuracy**: Fraction of base words preserved identically.

---

## 4. How to Run the Pipeline

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Run Full Evaluation (Inference + Metrics + Reports)
```bash
python evaluation/evaluate.py --force-inference
```

### Run Evaluation from Existing Predictions
```bash
python evaluation/evaluate.py --skip-inference
```

### Run Pipeline Tests
```bash
python -m unittest discover -s evaluation -p "test_*.py" -v
```

---

## 5. Summary of Results

| Metric | Overall Dataset ($N=55$) | High-Confidence Subset ($N=48$) |
| :--- | :--- | :--- |
| **Word Preservation Accuracy** | **100.00%** | **100.00%** |
| **Micro Precision** | **78.26%** | **78.95%** |
| **Micro Recall** | **73.97%** | **77.59%** |
| **Micro F1-Score** | **76.06%** | **78.26%** |
| **Macro F1-Score** | **70.22%** | **61.72%** |
| **Exact Sequence Accuracy** | **70.91%** | **75.00%** |

### Per-Class Performance
- **Purna Viram (`।`)**: F1-Score: **0.9278** (Precision: 0.9000, Recall: 0.9574, Support: 47)
- **Prashna Chinha (`?`)**: F1-Score: **0.5333** (Precision: 0.5714, Recall: 0.5000, Support: 8)
- **Alpa Viram (`,`)**: F1-Score: **0.3478** (Precision: 0.6667, Recall: 0.2353, Support: 17)
- **Hyphen (`-`)**: F1-Score: **1.0000** (Support: 1)
- **Colon (`:`)**: F1-Score: **0.0000** (Support: 0; model occasionally predicted `:` for vocatives)
- **Word without punctuation (`_NONE_`)**: F1-Score: **0.9848** (Support: 591)
