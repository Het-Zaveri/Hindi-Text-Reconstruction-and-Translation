# Hindi Text Reconstruction and Translation

A comprehensive NLP pipeline designed to process unpunctuated, unordered Hindi text, reconstruct coherent sentences, and translate them into English.

## Project Architecture & Roadmap

The project is structured into three consecutive NLP stages:

1. **Model 1: Hindi Punctuation Restoration** *(Active / Evaluated)*
   - Restores grammatical sentence-terminal punctuation (Purna Viram `।`, Prashna Chinha `?`) and internal punctuation (Alpa Viram `,`, Hyphen `-`, Colon `:`) into raw unpunctuated Hindi text.
2. **Model 2: Sentence Ordering and Reconstruction** *(Next Phase)*
   - Segments punctuated text into individual sentences and orders shuffled sentence segments into coherent paragraphs.
3. **Model 3: Hindi-to-English Translation** *(Future Phase)*
   - Translates coherent groups of five ordered Hindi sentences into natural English.

---

## Model 1: Hindi Punctuation Restoration and Evaluation

### 1. Pretrained Model Information
- **Model**: `Horizon-Labs/punctuation-restoration-small`
- **Hugging Face Hub**: [https://huggingface.co/Horizon-Labs/punctuation-restoration-small](https://huggingface.co/Horizon-Labs/punctuation-restoration-small)
- **Model Family**: Multilingual token classification model fine-tuned for punctuation restoration across multiple languages and scripts.
- **Parameters**: Approximately 140,643,846 parameters (~140M).
- **Supported Labels**:
  - `0`: No punctuation (`_NONE_`)
  - `.`: Standard sentence-terminal punctuation, dynamically mapped to Devanagari Danda (`।`) when preceded by Devanagari script codepoints (`[ऀ-৿]`).
  - `,`: Comma (Alpa Viram `,`)
  - `?`: Question Mark (Prashna Chinha `?`)
  - `-`: Hyphen (`-`)
  - `:`: Colon (`:`)
- **Long Text Processing**: Input texts are processed in sliding windows using `chunk_tokens=400` with an overlap of `overlap_tokens=50` to maintain sentence context across chunk boundaries.

---

### 2. Dataset Creation Methodology
The evaluation uses a separate reference set of 55 segments:
- **Corpus Extraction**: Extracted 55 representative Hindi sentence and utterance segments across all seven source files in the project (`sample-1.txt` to `sample-7.txt`).
- **Domain Coverage**:
  - Religious and cultural narratives (`sample-1.txt`, `sample-2.txt`)
  - Historical pilgrimage and transit data (`sample-3.txt`)
  - Scientific and agricultural expositions on Bt cotton and sericulture biology (`sample-4.txt`, `sample-5.txt`)
  - Conversational and colloquial dialogue with vocatives and tag questions (`sample-6.txt`, `sample-7.txt`)
- **Word Integrity**: Every reference sample was extracted verbatim from the raw texts. Word order and spelling were preserved 100%. `NULL` placeholders representing missing corpus content were stripped of word status and excluded from boundary evaluation.
- **Independent Annotation**: All punctuation annotations were established based on standard Hindi grammar rules (Devanagari orthographic conventions) independently before running model predictions.
- **Dataset Formats**: Available in `evaluation/dataset/reference_dataset.jsonl` (primary) and `evaluation/dataset/reference_dataset.csv`.

---

### 3. Annotation Limitations
- **AI-Assisted Initial Benchmark**: While created with rigorous Hindi grammatical criteria, the initial ground truth reference dataset was compiled with AI assistance. It requires independent verification by native Hindi speakers before being claimed as an authoritative gold standard.
- **Confidence Stratification**:
  - **High Confidence (48 samples, 87.3%)**: Clear, unambiguous syntactic boundaries (declarative assertions, direct questions, imperative commands).
  - **Medium Confidence (7 samples, 12.7%)**: Conversational lines with stylistic or intonational ambiguity (e.g. vocative commas, parenthetical relative clauses, and colloquial dialogue where a sentence could be either an emphatic statement or a question).

---

### 4. Evaluation Methodology
Punctuation restoration cannot be reliably measured by raw string edit distance or character overlap. This pipeline evaluates punctuation at **word boundaries**:
1. **Extraction**: Raw and reference strings are tokenized into base words and attached punctuation marks.
2. **Alignment**: Sequences of base words are aligned using `difflib.SequenceMatcher`.
3. **Boundary Classification**: Each word boundary is evaluated as a multi-class prediction against the ground truth.
4. **Metrics Evaluated**:
   - Word Preservation Accuracy ($\frac{\text{matching words}}{\text{total reference words}}$)
   - Exact Sequence Accuracy (fraction of sentences where all punctuation marks match exactly)
   - Class-wise Precision, Recall, and F1-score
   - Micro Precision, Micro Recall, and Micro F1 (aggregated over all non-null punctuation marks)
   - Macro F1 across punctuation classes with non-zero support

---

### 5. Quantitative Evaluation Results

The evaluation pipeline was executed on the 55 reference samples using actual model predictions generated by `Horizon-Labs/punctuation-restoration-small`:

| Metric | Overall Dataset ($N=55$) | High-Confidence Subset ($N=48$) |
| :--- | :--- | :--- |
| **Word Preservation Accuracy** | **100.00%** | **100.00%** |
| **Micro Precision** | **78.26%** | **78.95%** |
| **Micro Recall** | **73.97%** | **77.59%** |
| **Micro F1-Score** | **76.06%** | **78.26%** |
| **Macro F1-Score** | **70.22%** | **61.72%** |
| **Exact Sequence Accuracy** | **70.91%** | **75.00%** |

#### Class-Wise Performance Breakdown
| Punctuation Mark | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Purna Viram (`।`)** | **0.9000** | **0.9574** | **0.9278** | 47 |
| **Prashna Chinha (`?`)** | **0.5714** | **0.5000** | **0.5333** | 8 |
| **Alpa Viram (`,`)** | **0.6667** | **0.2353** | **0.3478** | 17 |
| **Hyphen (`-`)** | **1.0000** | **1.0000** | **1.0000** | 1 |
| **Colon (`:`)** | **0.0000** | **0.0000** | **0.0000** | 0 |
| **No Punctuation (`_NONE_`)** | **0.9815** | **0.9882** | **0.9848** | 591 |

---

### 6. Error Analysis & Qualitative Findings

#### High-Performing Patterns
- **Terminal Purna Viram (`।`)**: The model excels at detecting the end of declarative Hindi statements (F1: 0.9278, Recall: 95.74%). Verbs like `है`, `था`, `गए`, and `होता है` at sentence boundaries are consistently followed by `।`.
- **Lexical Preservation**: Word preservation accuracy is 100%. The model does not corrupt Devanagari matras, conjuncts, or loanwords.

#### Common Failure Modes
- **Under-Prediction of Internal Commas (Alpa Viram `,`)**: The recall for commas is low (23.53%). The model frequently misses commas before coordinating conjunctions (`किंतु`, `परंतु`, `लेकिन`) and relative clauses (`जिसमें`, `जिसके`). Out of 17 true commas, 9 were omitted as `_NONE_`.
- **Dialogue Header Bias (Colon `:` vs Comma `,`)**: In conversational texts (`sample-6.txt`, `sample-7.txt`), when an utterance starts with a character name (e.g., `अंजलि तुम्हारा फोन है`), the model frequently predicts a colon (`अंजलि: तुम्हारा फोन है?`) instead of a vocative comma (`अंजलि, तुम्हारा फोन है।`).
- **Conversational Questions vs. Declaratives**: Short questions without question words (`क्या`, `क्यों`) or with trailing tags (`ना`) are sometimes misclassified as statements with `।` or over-punctuated (`क्या कर रही है तू? यहां?`).

---

### 7. Hindi-Specific Linguistic Challenges
1. **Danda vs. Period**: Standard English tokenizers map terminal stops to `.`. The inference script must maintain Devanagari normalization logic to map `.` to `।`.
2. **Ambiguity in Spoken Syntax**: In colloquial Hindi cinema dialogues, sentences often lack overt interrogative markers (e.g. `तुम खुश हो बेटी`), relying on vocal pitch rather than grammatical markers.
3. **Compound Conjunctions**: Hindi allows flexible word order in subordinate clauses (`अतः हम कह सकते हैं कि...`), which challenges token classifiers trained predominantly on formal written English/multilingual text.

---

### 8. GPU Memory Compliance Status
- **Constraint**: The project specifies a maximum limit of **1.5 GB GPU RAM**.
- **Execution Device**: All benchmarks and inference tests were conducted on CPU (`device="cpu"`).
- **Physical Verification**: In this execution environment, dedicated CUDA GPU hardware is unavailable (`torch.cuda.is_available() == False`).
- **Compliance Status**: **NOT VERIFIED**. Actual peak GPU memory was not measured on physical hardware. No unsupported theoretical claims are made regarding GPU memory consumption.

---

### 9. Concrete Recommendations for Task 2 and Task 3

#### For Task 2 (Sentence Ordering & Reconstruction):
1. **Rely Primarily on Purna Viram (`।`) for Segmentation**: Because Model 1 achieves 95.7% recall and 90.0% precision on `।`, splitting text on `।` will yield high-precision sentence boundaries for the ordering model.
2. **Post-Process Vocative Colons**: Convert initial dialogue colons (`Name:`) into vocatives (`Name,`) before feeding sentences into the reconstruction model.
3. **Rule-Based Comma Insertion**: Supplement the model with lightweight rule-based comma insertions before adversative markers (`परंतु`, `किंतु`, `लेकिन`, `मगर`) to improve syntactic clarity before sentence ordering.

#### For Task 3 (Hindi-to-English Translation):
1. **Preserve Chunk Boundaries**: Maintain 5-sentence grouped blocks based on clean `।` delimiters.
2. **Clean Trailing Question Marks in Statements**: In rhetorical or indignant lines misclassified with `?`, verify sentence mood to prevent the English translation model from generating awkward questions instead of assertions.

---

## Quick Start & Reproduction

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Run Evaluation Pipeline
```bash
# Run fresh inference and generate all reports
python evaluation/evaluate.py --force-inference

# Re-run metrics from verified existing predictions cache
python evaluation/evaluate.py --skip-inference
```

### Run Unit Tests
```bash
python -m unittest discover -s evaluation -p "test_*.py" -v
```
