# Model 3 IndicTrans2 evaluation report

**Recommendation: `CANDIDATE NOT ACCEPTABLE - TRANSLATION QUALITY POOR`**

The official IndicTrans2 model ran on CPU in the existing Windows environment. Audit found malformed `u093C` escape fragments in **37 of 344** translations (10.8%) across samples 1, 2, 4, 5, and 7. The defect is unresolved; original translations were not changed and no speculative corrections were made. All seven files remain **UNVALIDATED**. Given the text corruption and known semantic errors, this candidate is not acceptable as final translation output without repair and human review.

## A. Environment

- Python: 3.12.10
- Existing environment: `.venv_model3_indictrans2` (reused; not recreated)
- PyTorch: 2.6.0+cpu; CUDA unavailable
- Transformers: 4.51.3
- NumPy: 2.1.3
- SentencePiece: 0.2.0
- IndicTransToolkit: 1.1.1
- `pip check`: **PASS**, no broken requirements
- No dependencies were installed or changed during this resumed run.

## B. Toolkit compatibility

`from IndicTransToolkit import IndicProcessor` and `IndicProcessor(inference=True)` work on native Windows. The package documentation still says it is not built or tested for Windows ([PyPI project guidance](https://pypi.org/project/indictranstoolkit/)); this run confirms the installed toolkit worked in this environment. The official processor performed preprocessing and postprocessing; no replacement/no-op processor was used.

## C. Model

- Model ID: `ai4bharat/indictrans2-indic-en-dist-200M`
- License: MIT, as listed on the [Hugging Face model card](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M)
- Actual parameter count: 211,780,608
- Downloaded snapshot: 27 files, 921,510,625 bytes (921.51 MB decimal / 879.18 MiB by summed file lengths)
- Pilot tokenizer load: 1.771 seconds
- Pilot model load: 4.530 seconds
- Pilot CPU inference: 4.071 seconds for one batch of five
- Pilot peak working set: 1,182.45 MB
- Full-file run peak working set: 1,525–1,805 MB per process
- GPU-memory compliance: **NOT VERIFIED**

CPU working set is not a GPU-memory measurement. No NVIDIA GPU is present.

## D. Five-sentence pilot

The exact first five Hindi sentences from `outputs/pipeline_final/sample-1_final_output.txt` were cross-checked against the original Helsinki and M2M100 pilot inputs. They were submitted together as one ordered batch of five.

| # | Hindi source | Actual English translation |
|---:|---|---|
| 1 | उस समय गढ़ मुक्तेश्वर बल्लभ संप्रदाय का मुख्य केन्द्र था। | At that time Garh\ u093C was the main center of the Mukteshwar Ballabh sect. |
| 2 | गढ़ मुक्तेश्वर का नाम शिव बल्लभपुर पड़ा। | Garh\ u093C Mukteshwar got the name Shiv Ballabhpur. |
| 3 | ज्ञात रहे कि शिव बल्लभपुर को ही अब गढ़ मुक्तेश्वर के नाम से जाना जाता है। | It may be noted that Shiv Ballabhpur itself is now known as Garh Mukteshwar. |
| 4 | महाराज शिव ने भगवान परशुराम से यहाँ शिव मंदिर की स्थापना कराई थी। | Maharaj Shiva got Lord Parshuram to establish a Shiva temple here. |
| 5 | गंगा किनारे यह भगवान शंकर की प्रसिद्ध पुरी है। | It is the famous Puri of Lord Shankar on the banks of the Ganges. |

All five outputs are nonempty, in source order, and mapped one-to-one. The first two mangle Garhmukteshwar as `Garh\ u093C`. While this small pilot was more coherent than the rejected pilots, the broader audit found the same fragment class in 37/344 outputs; the pilot alone overstated how promising the full candidate is. This is not human validation.

### Malformed fragment trace and correction decision

The defect was replayed with affected sentences from samples 1, 4, 5, and 7. At the earliest observed stage, official `IndicProcessor.preprocess_batch` represents nukta-bearing input with a literal `\\u093C` sequence in its preprocessed text. Generated token IDs decode to English containing literal forms such as `\\ u093C` (for example, `Garh\\ u093C`, `silkworm\\ u093C`, and `Bad\\ u093C`). `IndicProcessor.postprocess_batch` returns those fragments unchanged, and UTF-8 JSONL/text writing preserves them. The defect is present in decoded model output; neither postprocessing nor file encoding introduces it.

The audit found 37 affected translations: sample 1: 2; sample 2: 1; sample 4: 7; sample 5: 14; sample 7: 13. Cases include escapes appended to translated words, fragments embedded inside words, and outputs with semantic errors. **No safe general correction was established.** Blindly deleting the marker could alter intended words or hide mistranslations, and no reference translations exist to recover missing text. No correction function or corrected files were created. Original translation files remain unchanged; the count is 37 before and 37 after.

## E. Candidate comparison

The Helsinki pilot merged content, mangled names, and omitted later source sentences. M2M100 combined decoding repeated “Buddha” until a loop; its sentence-level results included the offensive “bullshit community,” mangled place names, and substantial meaning loss. IndicTrans2's pilot was more coherent, but 37/344 full-set translations contain malformed fragments and semantic errors are present. The full candidate is not acceptable without repair and human review. No BLEU, chrF, COMET, or accuracy score is reported because no reference translations exist.

## F. Performance

| Run | Sentences | Groups | Inference time | Peak CPU working set |
|---|---:|---:|---:|---:|
| Pilot | 5 | 1 | 4.071 s | 1,182.45 MB |
| Seven files | 344 | 71 | 371.475 s total | 1,525–1,805 MB across runs |

Aggregate full-set throughput was about 0.93 sentences per inference second. The full-set runs loaded the tokenizer in 1.44–1.93 seconds and the model in 4.43–5.19 seconds per process. These CPU measurements do not establish GPU-memory use.

## G. Dataset processing

All seven ordered files were processed with groups of five and a partial final group where needed. There were 344 nonempty source lines, 344 unique manifest records, seven provisional text files, and 71 groups. Alignment validation confirmed that every manifest source matches its ordered source line and every manifest translation matches the corresponding text output. All checked files decode as strict UTF-8. There were zero inference errors, empty outputs, or truncation flags. Translation quality remains **UNVALIDATED** throughout.

The source Hindi outputs were not modified. A script-side metric bug was found and corrected during this run: IndicTrans2's tokenizer `encode` path treats input as language-tagged source text, so it cannot count English output tokens. The script now counts English tokens via the tokenizer's target SentencePiece path. The corrected pilot and all seven files completed with exit code 0.

## H. Tests and checks

- `.\.venv_model3_indictrans2\Scripts\python.exe -m pip check` — **PASS**.
- Official processor import/initialization — **PASS**.
- `.\.venv_model3_indictrans2\Scripts\python.exe -m py_compile scripts/test_model3_indictrans2.py` — **PASS**.
- Standalone script `--help` — **PASS**.
- Pilot inference — **PASS**, five sentences, one batch, zero technical failures.
- Dataset inference — **PASS**, seven files, 344 sentences, 71 groups, zero technical failures.
- `.\venv\Scripts\python.exe -m unittest discover -s evaluation -p "test_*.py" -v` — **PASS, 65/65 tests in 25.385 seconds**.

## I. GPU compliance

**GPU-memory compliance: NOT VERIFIED.** This computer has no NVIDIA GPU. CPU RAM figures above are reported separately and are not used as a GPU estimate.

## J. Recommendation

**`CANDIDATE NOT ACCEPTABLE - TRANSLATION QUALITY POOR`**

The five-sentence pilot was aligned and more coherent than the saved Helsinki and M2M100 pilot outputs. The broader audit is decisive: 37 of 344 translations contain malformed `u093C` fragments, and semantic errors are present. No safe automatic correction was found. Keep the originals unchanged and **UNVALIDATED**; do not use this candidate as final translation output until the defects are repaired and a human reviews it.

## K. Files changed

Created or updated for Model 3:

- `scripts/test_model3_indictrans2.py`
- `outputs/translation/indictrans2/sample-1_indictrans2_pilot.json`
- `outputs/translation/indictrans2/sample-1_indictrans2_pilot.md`
- `outputs/translation/indictrans2/sample-1_indictrans2_pilot.txt`
- `outputs/translation/indictrans2/model3_indictrans2_report.md`
- `outputs/translation/indictrans2/model3_indictrans2_benchmark.json`
- `outputs/translation/indictrans2/translation_manifest.jsonl`
- `outputs/translation/indictrans2/sample-1_indictrans2_unvalidated.txt` through `sample-7_indictrans2_unvalidated.txt`

The new environment remains `.venv_model3_indictrans2/` and is ignored by the narrowly scoped `.gitignore` entry added in the earlier task turn. Model 1 and Model 2 code and outputs, original Helsinki and M2M100 artifacts, root `requirements.txt`, `requirements-model3.txt`, and `.venv_model3` were preserved. No commit or push was made.
