# Hindi Text Reconstruction and Translation

This repository studies a three-stage workflow for raw, shuffled Hindi text: restore punctuation, order the resulting sentences, and evaluate Hindi-to-English translation. Models 1 and 2 form the implemented Hindi reconstruction pipeline. Model 3 is a separate translation experiment; the tested IndicTrans2 candidate produced malformed text and is **not acceptable as final translation output**.

The project combines pretrained models with custom preprocessing, search, orchestration, and integrity checks. It is a research prototype, not a validated production translator.

## Key features

- Hindi punctuation restoration with word-preservation evaluation.
- Sentence segmentation at Devanagari danda, question mark, exclamation mark, and line boundaries.
- Multilingual sentence embeddings and pairwise cosine-similarity scoring.
- Heuristic sentence-order reconstruction with exact search for short inputs and beam search for longer inputs.
- Sentence IDs, source-to-output mappings, and exact sentence-preservation checks.
- An experimental Hindi-to-English translation evaluation using five-sentence batches.
- CPU-based development and regression testing; no NVIDIA GPU was available for verification.

## Architecture

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

The main pipeline in `src/pipeline.py` runs Models 1 and 2 only. Its `--method minilm` mode selects MiniLM; the default `compare` mode additionally runs a directional mBERT next-sentence-prediction experiment. Translation is invoked separately after reconstruction. For the MiniLM score, cosine similarity is symmetric and indicates semantic affinity, not transition direction.

## Models and status

| Stage | Model | Architecture and verified specification | Language/task | Status |
|---|---|---|---|---|
| 1 | [Horizon-Labs/punctuation-restoration-small](https://huggingface.co/Horizon-Labs/punctuation-restoration-small) | mmBERT-small token-classification model; about 140M parameters per model card; Apache-2.0 | Multilingual punctuation restoration; Hindi tested here | Implemented and evaluated; 55-record reference set is provisional |
| 2 | [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | 12-layer BERT-family encoder; hidden size 384; mean pooling to 384-dimensional sentence embeddings; Apache-2.0 | Sentence embeddings across 50 languages, including Hindi | Selected practical ordering heuristic; integrity checked, ordering accuracy unknown |
| 2 comparison | [bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) | BERT encoder with NSP head loaded and verified by project code | Directional next-sentence scoring | Experimental comparison method, not the selected primary method |
| 3 experiment | [IndicTrans2 Indic-En Distilled 200M](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) | Encoder-decoder translation model; model card calls it distilled 200M; loaded checkpoint has 211,780,608 parameters; MIT | Hindi `hin_Deva` to English `eng_Latn` | Rejected for final-quality use: 37/344 outputs contain malformed `u093C` fragments; other semantic errors; all outputs UNVALIDATED |
| 3 rejected | [Helsinki-NLP/opus-mt-hi-en](https://huggingface.co/Helsinki-NLP/opus-mt-hi-en) | OPUS-MT transformer-align/Marian model; parameter count not stated on the card; Apache-2.0 | Hindi to English | Pilot rejected for malformed, incomplete translation |
| 3 rejected | [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M) | Multilingual encoder-decoder; 418M variant; MIT | Many-to-many translation across 100 languages, including Hindi and English | Pilot rejected for repetition loop and severe mistranslations |

Model specifications above come from the linked model cards and, for IndicTrans2 runtime parameter count and output measurements, the local benchmark. Check [Models](docs/MODELS.md) for caveats and sources. No model is claimed to meet the 1.5 GB GPU-memory target: **GPU-memory compliance: NOT VERIFIED**.

## Method overview

### 1. Punctuation restoration

`src/run_on_dataset.py` loads the model publisher's `punctuate.py` helper, cleans whitespace and `NULL` placeholders, and processes each `data/*.txt` file. The model predicts punctuation labels after tokens; its helper reconstructs text and maps sentence-ending periods to Devanagari danda where appropriate. The evaluation compares word-boundary punctuation against a provisional 55-record annotation set. Detailed results and annotation limitations are in [Evaluation](docs/EVALUATION.md).

### 2. Sentence ordering

`src/model2_ordering.py` splits punctuated text on line boundaries or whitespace after `।`, `?`, or `!`, assigns stable sentence IDs, and obtains normalized 384-dimensional MiniLM embeddings. It scores candidate adjacent sentences by cosine similarity and searches for an ordering: exhaustive permutations for at most eight sentences, beam search (default width 64) above that. Each output is then checked as a multiset against its source sentences. This verifies preservation, not that the proposed order is historically or narratively correct. See [Methodology](docs/METHODOLOGY.md).

### 3. Translation experiment

`scripts/test_model3_indictrans2.py` uses the official IndicTransToolkit `IndicProcessor`, source code `hin_Deva`, target code `eng_Latn`, and CPU encoder-decoder inference. Sentences are passed in groups of five and postprocessed in order. The full run generated seven provisional files and a manifest for 344 sentences in 71 groups. Quality is **UNVALIDATED** and unsuitable for final use: 37 translations contain malformed `u093C` fragments, and semantic errors are present. No automatic repair was applied. See the [Model 3 audit](outputs/translation/indictrans2/model3_indictrans2_report.md).

## Data and outputs

The seven professor-provided raw Hindi inputs are UTF-8 plain-text files, `data/sample-1.txt` through `data/sample-7.txt`. Model 1 reads all `data/*.txt` files and writes punctuated files under `outputs/`; Model 2 writes ordered text, mappings, and run reports under `outputs/model2_ordering/`; the orchestration pipeline writes reconstructed Hindi under `outputs/pipeline_final/`. Model 3 outputs are under `outputs/translation/indictrans2/`. Sentence mappings carry sample-local stable IDs and original positions. See [Architecture](docs/ARCHITECTURE.md) for the actual paths and data flow.

## Setup and usage

Use PowerShell from the repository root. The examples below match the project entry points; options for scripts with a CLI were checked against their `--help` output. A suitable existing environment is required; see [Installation](docs/INSTALLATION.md) for the separate Model 3 environment and dependency files.

```powershell
# Model 1: punctuate every data/*.txt file (downloads its model/helper as needed).
.\venv\Scripts\python.exe src\run_on_dataset.py

# Model 2: run the selected MiniLM method using existing punctuated files.
.\venv\Scripts\python.exe src\model2_ordering.py --method minilm

# Full Hindi reconstruction: Model 1 then MiniLM ordering on all seven inputs.
.\venv\Scripts\python.exe src\pipeline.py --method minilm

# Optional Model 1 metrics using the saved predictions cache.
.\venv\Scripts\python.exe evaluation\evaluate.py --skip-inference

# Model 3: five-sentence experimental pilot from an ordered file.
.\.venv_model3_indictrans2\Scripts\python.exe scripts\test_model3_indictrans2.py `
  --input outputs\pipeline_final\sample-1_final_output.txt --count 5 `
  --group-size 5 --device cpu `
  --output-json outputs\translation\indictrans2\sample-1_rerun.json

# Existing regression tests (the project's venv in this checkout).
.\venv\Scripts\python.exe -m unittest discover -s evaluation -p "test_*.py" -v
```

Commands that rerun inference may overwrite corresponding output files. Model files are fetched from Hugging Face on demand; gated model access, including IndicTrans2 access acceptance, must be completed by the user. No token is stored in the repository.

## Evaluation snapshot

| Component | Result | Important limitation |
|---|---|---|
| Model 1 punctuation | Micro F1 76.06%; macro F1 70.22%; exact punctuation sequence accuracy 70.91%; word preservation 100% on 55 records | Reference annotations were AI-assisted and require validation by fluent Hindi speakers |
| Model 2 MiniLM | Seven files; sentence counts 49, 49, 47, 50, 52, 45, 52; preservation checks passed | No authoritative ordering references, so no ordering accuracy is reported |
| Model 3 IndicTrans2 | 344 sentences, 71 groups; 37 malformed `u093C` outputs | Other semantic errors; all English outputs UNVALIDATED; candidate rejected for final-quality use |
| Existing regression suite | 65/65 passed in 23.074 seconds, rerun during this documentation task in the existing project `venv` | Software regression result, not proof of linguistic accuracy |

## Hardware and limitations

Development was reported on Windows 11 with an Intel Core i5-8365U, integrated graphics, and no NVIDIA GPU. The project target is below 1.5 GB of GPU memory per model. **GPU-memory compliance: NOT VERIFIED.** IndicTrans2's CPU benchmark reports a 4.071-second five-sentence pilot and 371.475 seconds total inference for 344 sentences; CPU RAM and downloaded file size do not imply GPU VRAM use.

Other limitations include the provisional punctuation references, lack of correct sentence-order references, symmetric cosine affinity, and unvalidated/defective machine translation. The system requires fluent Hindi-English review before any translation is treated as acceptable. See [Evaluation](docs/EVALUATION.md) and [future work](docs/METHODOLOGY.md#limitations-and-next-steps).

## Repository structure

```text
.
├── README.md
├── docs/                         # Architecture, models, methods, evaluation, setup
├── benchmark_mbert.py            # Existing ordering benchmark utility
├── benchmark_mbert_results.json  # Existing benchmark results
├── benchmark_mbert.log           # Existing benchmark log (untracked local artifact)
├── cleanup_audit.md              # Existing audit notes
├── data/                         # sample-1.txt … sample-7.txt
├── src/
│   ├── punctuation_test.py       # Small Model 1 example
│   ├── run_on_dataset.py         # Model 1 batch runner
│   ├── model2_ordering.py        # MiniLM and mBERT ordering methods
│   └── pipeline.py               # Model 1 → Model 2 orchestration
├── scripts/
│   ├── test_model3_candidate.py
│   ├── test_model3_m2m100.py
│   └── test_model3_indictrans2.py
├── evaluation/
│   ├── dataset/                   # CSV/JSONL punctuation references
│   ├── predictions/               # Model 1 cached predictions
│   ├── reports/                   # Metrics and classification reports
│   ├── README.md
│   ├── evaluate.py
│   ├── test_model2.py
│   └── test_pipeline.py
├── outputs/
│   ├── sample-*_punctuated.txt
│   ├── model2_ordering/           # Ordered text, mappings, NSP comparisons
│   ├── pipeline_final/            # Reconstructed Hindi and pipeline reports
│   └── translation/
│       ├── indictrans2/           # Pilot, report, manifest, UNVALIDATED outputs
│       └── ...                    # Preserved Helsinki and M2M100 experiment files
├── notebooks/                     # IndicTrans2 pilot notebook
├── requirements.txt
├── requirements-model3.txt
├── requirements-model3-indictrans2.txt
└── .gitignore
```

The current workspace also contains the untracked local archive `git hub pusshing.zip`; it is not needed for the application or recommended for the documentation commit. Repeated sample outputs are grouped with `sample-*` patterns. Environments, caches, and model weights are omitted.

## Documentation

- [Architecture and data flow](docs/ARCHITECTURE.md)
- [Model specifications and status](docs/MODELS.md)
- [Methodology](docs/METHODOLOGY.md)
- [Evaluation and limitations](docs/EVALUATION.md)
- [Windows installation and commands](docs/INSTALLATION.md)

## Academic integrity and reproducibility

Models 1–3 are pretrained checkpoints. Sentence segmentation, MiniLM cosine scoring/search, ID mapping, evaluation alignment, and orchestration are project code; MiniLM sentence ordering is a heuristic, not a trained directional transition model. Reported metrics and integrity checks are tied to the local artifacts cited above. Punctuation labels are provisional, sentence ordering lacks gold references, translation has no reference-based score, and GPU compliance is unverified. The 65/65 regression result is a software test result, not proof of linguistic accuracy. Do not infer success from a model having executed.
