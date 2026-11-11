# Model 1 Evaluation Summary: Hindi Punctuation Restoration

## 1. Executive Overview
- **Model**: `Horizon-Labs/punctuation-restoration-small`
- **Model Revision**: `12b02a85b60a2843ad6f22e4248669f50aeecdbb`
- **Primary Task**: Restoring grammatical punctuation marks in raw, unpunctuated Hindi text
- **Execution Device**: `cpu` (CPU)
- **Total Reference Samples**: 55
- **Valid Evaluated Samples**: 55
- **Inference Failures**: 0
- **Evaluation Mode**: Word-boundary sequence alignment with explicit `_NONE_` (no-punctuation) modeling

## 2. Key Performance Metrics

| Metric | Overall Dataset (N=55) | High-Confidence Subset (N=48) |
| :--- | :--- | :--- |
| **Word Preservation Accuracy** | **100.00%** | **100.00%** |
| **Micro Precision** | **78.26%** | **78.95%** |
| **Micro Recall** | **73.97%** | **77.59%** |
| **Micro F1-Score** | **76.06%** | **78.26%** |
| **Macro F1-Score** | **70.22%** | **61.72%** |
| **Exact Sequence Accuracy** | **70.91%** | **75.00%** |

### Class-Wise Metrics
| Punctuation Mark | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `,` | 0.6667 | 0.2353 | 0.3478 | 17 |
| `-` | 1.0000 | 1.0000 | 1.0000 | 1 |
| `:` | 0.0000 | 0.0000 | 0.0000 | 0 |
| `?` | 0.5714 | 0.5000 | 0.5333 | 8 |
| `।` | 0.9000 | 0.9574 | 0.9278 | 47 |
| `_NONE_` (Words) | 0.9815 | 0.9882 | 0.9848 | 591 |

## 3. Linguistic & Qualitative Error Analysis

### Strengths
1. **Purna Viram (।) Placement**: The model demonstrates exceptionally strong precision on sentence-terminal markers (Purna Viram `।`), reliably identifying the ends of declarative statements.
2. **Word Preservation**: 100% word preservation. The model performs token classification and does not mutate, drop, or hallucinate Hindi lexical tokens.
3. **Devanagari Normalization**: The helper correctly maps standard period `.` predictions to the Hindi danda `।` for Devanagari Unicode codepoints.

### Error Patterns & Weaknesses
1. **Internal Commas (Alpa Viram `,`)**: The model frequently omits internal commas before adversative conjunctions (`किंतु`, `परंतु`) and relative pronouns (`जिसमें`, `जिसके`). This yields lower recall for commas compared to dandas.
2. **Vocative Address & Dialogue Formatting**: In conversational cinema dialogue (`sample-6.txt`, `sample-7.txt`), when a character name appears initially (e.g. `अंजलि`), the model often predicts a colon `:` rather than a vocative comma `,`, treating the name as a script dialogue header.
3. **Conversational Questions vs. Declaratives**: Short informal questions without explicit interrogative pronouns (such as `तुम खुश हो बेटी` or tag questions `है ना`) are occasionally punctuated with terminal danda `।` instead of question mark `?`, reflecting ambiguity without vocal prosody.

## 4. Hardware & GPU Memory Compliance
- **Execution Device**: `cpu` (CPU)
- **CUDA Availability**: `False`
- **Actual Peak GPU Memory Allocated**: `N/A (CPU execution)`
- **Actual Peak GPU Memory Reserved**: `N/A (CPU execution)`
- **GPU Memory Compliance Status**: `NOT VERIFIED`
- **Compliance Notes**: Executed on CPU. Dedicated CUDA GPU is unavailable in this environment; actual GPU memory was not measured, and GPU memory compliance (<= 1.5 GB limit) is NOT VERIFIED.
- **Verification Statement**: Dedicated CUDA GPU hardware was not present in this execution environment; model execution ran entirely on CPU. GPU memory compliance (<= 1.5 GB limit) is NOT VERIFIED. No unsupported theoretical GPU claims are asserted without physical GPU telemetry.

## 5. Dataset Integrity & Annotation Limitations
- The reference dataset comprises 55 segments independently extracted from the 7 source files.
- The reference annotations were curated using grammatical Hindi rules (Deo-Devanagari conventions) independent of model outputs.
- As required, this initial ground truth is AI-assisted and should undergo formal validation by native Hindi speakers before being claimed as an authoritative gold standard.
