# Evaluation results and limitations

This document reports saved project results. Metrics describe their particular data and evaluation procedures; none should be generalized beyond that evidence.

## Model 1 — punctuation restoration

The recorded evaluation uses 55 Hindi records and cached/generated predictions from `Horizon-Labs/punctuation-restoration-small`.

| Metric | Result |
|---|---:|
| Micro F1 | 76.06% |
| Macro F1 | 70.22% |
| Exact punctuation sequence accuracy | 70.91% |
| Word preservation | 100.00% |

The word-preservation figure reflects these 55 comparisons. The reference annotations were AI-assisted, with confidence categories described in the evaluation report. They require validation by fluent Hindi speakers before being treated as authoritative gold labels. See `evaluation/reports/evaluation_summary.md`, `metrics.json`, and `classification_report.txt`.

## Model 2 — sentence ordering

The selected MiniLM-based run processed seven files with sentence counts 49, 49, 47, 50, 52, 45, and 52 (344 total). The saved run report marks integrity as passed for each file: every segmented sentence appears once and text is unchanged. This is structural preservation only.

No authoritative correct sentence order is available. Similarity scores, beam-search scores, and agreement with the optional mBERT method are not ordering accuracy. No accuracy, precision, or recall for order reconstruction is claimed.

## Model 3 — translation experiments

### IndicTrans2

The CPU benchmark processed all seven ordered files: **344 sentences in 71 groups** of up to five. The manifest has one record per source sentence. Recorded inference errors: 0; empty outputs: 0; truncation flags: 0. These are technical completeness checks, not translation-quality metrics.

The independent quality audit found malformed literal `u093C` fragments in **37/344 translations (10.8%)**:

| Sample | Affected translations |
|---|---:|
| sample-1 | 2 |
| sample-2 | 1 |
| sample-4 | 7 |
| sample-5 | 14 |
| sample-7 | 13 |
| **Total** | **37** |

Other semantic translation errors are present. The defect was traced through preprocessed input, generated token decoding, processor postprocessing, and file writing. It is visible in decoded generation and survives postprocessing. No safe deterministic corpus-wide correction was established; original translation outputs remain unchanged. All translations are **UNVALIDATED**. Candidate status: **NOT ACCEPTABLE - TRANSLATION QUALITY POOR**. No BLEU, chrF, COMET, or accuracy score is available because the project has no reference English translations.

The benchmark reports a five-sentence pilot inference time of 4.071 seconds, peak process working set 1,182.45 MB, and 371.475 seconds of total inference for 344 sentences. The downloaded snapshot's summed file length is about 921.51 MB decimal. CPU working set and model disk size are not GPU-memory measurements.

### Rejected candidates

- Helsinki-NLP `opus-mt-hi-en`: pilot produced malformed English and omitted source content.
- M2M100 418M converted to CTranslate2 INT8: combined decoding looped on a repeated token, and sentence-level results included serious entity/meaning errors and an offensive mistranslation.

Original pilot and benchmark artifacts are retained under `outputs/translation/`; these candidates were not selected for final-quality translation.

## Regression tests

The regression suite was rerun during this documentation task in the existing project `venv` (Python 3.14.7): **65/65 passed in 23.074 seconds**. This is a software test result, not a linguistic-quality measure. To rerun:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s evaluation -p "test_*.py" -v
```

Passing software tests are not proof of linguistic correctness.

## Hardware compliance

The development machine is reported as an Intel Core i5-8365U system with integrated graphics and no NVIDIA GPU. The project target is less than 1.5 GB GPU memory per model. **GPU-memory compliance: NOT VERIFIED.** CPU RAM measurements, parameter counts, and checkpoint disk sizes do not substitute for GPU telemetry.
