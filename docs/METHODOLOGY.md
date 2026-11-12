# Methodology

## Stage 1 — Hindi punctuation restoration

### Input preparation

`src/run_on_dataset.py` reads UTF-8 text from `data/*.txt`. It removes the literal placeholder `NULL` by replacing it with whitespace, collapses repeated whitespace, then calls the publisher-provided `Punctuator` helper downloaded from the Hugging Face repository. `src/pipeline.py` contains the same preprocessing path for orchestration.

### Inference and reconstruction

The selected checkpoint is a multilingual token classifier. The publisher's helper maps token-level predictions back to word positions and renders punctuation. Labels include no punctuation, period, comma, question mark, hyphen, and colon. For Devanagari text, period predictions may be rendered as danda (`।`) by the helper. The task is punctuation restoration only; this project expects source words to be retained.

### Evaluation

`evaluation/evaluate.py` aligns base words and punctuation at word boundaries, computes exact punctuation-sequence accuracy and class/micro/macro metrics, and separately measures word preservation. Its 55-reference dataset was compiled with AI assistance and is **provisional**, not independently validated gold data. The evaluation report records 100% word preservation on these records; that is not a universal guarantee.

## Stage 2 — sentence ordering

### Segmentation and identity

`src/model2_ordering.py::segment_sentences` splits on newlines and on whitespace after `।`, `?`, or `!`, retaining terminal punctuation in each sentence. `assign_sentence_ids` assigns deterministic IDs based on input stem and original sentence position. These IDs make order mappings auditable.

### MiniLM embeddings and cosine score

The primary practical method calls `SentenceTransformer.encode` with normalized embeddings. The model card describes mean pooling into 384-dimensional vectors. For normalized vectors, the dot product equals cosine similarity:

\[
\operatorname{sim}(A,B)=\frac{A\cdot B}{\|A\|\|B\|}.
\]

The implementation sums adjacent similarities in each proposed permutation. It uses exact permutation search for at most eight sentences and beam search (default width 64) for larger sets. The seven files in the recorded run contain 45–52 sentences each, so beam search was used.

### Interpretation limitation

Cosine similarity is symmetric: `sim(A,B) = sim(B,A)`. A high score means two sentences are semantically close; it does not say which one comes first. The search is therefore a semantic-affinity heuristic. It may group related material yet reverse chronology, discourse relations, or narrative direction. It is not a trained directional next-sentence predictor and no ordering accuracy is claimed without reference orders.

### Optional directional comparison

The same source implements an mBERT next-sentence-prediction comparison. It computes a directional margin from the NSP logits and uses a directional transition matrix with sequence search. Code verifies that the checkpoint supplied the NSP head. This is an experimental comparator, not the selected practical MiniLM configuration, and comparison between two predictions is not ground-truth accuracy.

### Preservation validation

After ordering, the checker compares sentence counts and sentence multiplicities. It verifies a permutation of the segmented strings: no sentence is missing, added, duplicated, or edited. This check says nothing about linguistic correctness of the permutation.

## Stage 3 — translation experiment

The separate script `scripts/test_model3_indictrans2.py` reads nonempty lines as ordered sentences. It calls official `IndicProcessor(inference=True)`, preprocesses groups of five with source `hin_Deva` and target `eng_Latn`, tokenizes the tagged inputs, checks source length before generation, runs CPU `AutoModelForSeq2SeqLM.generate` under inference mode, batch-decodes, and applies official target-language postprocessing. Sentence index and group index are retained in JSON output and the full-run JSONL manifest.

The run produced one output record for each of 344 source sentences, grouped into 71 batches, with no technical errors, empty outputs, or truncation flags. This is separate from human translation quality. The independent audit found literal malformed `u093C` fragments in 37 outputs plus semantic errors. No automatic correction was applied because generic deletion/replacement could alter words and no gold English references are available. The outputs remain **UNVALIDATED**, and IndicTrans2 is rejected as an acceptable final candidate.

## Limitations and next steps

- Validate the punctuation references with fluent Hindi speakers and maintain explicit annotation guidance.
- Collect authoritative sentence orders before measuring ordering accuracy; consider a directional discourse model or learned transition scorer.
- Obtain bilingual human review and trusted references before selecting a Hindi-English translation model.
- Investigate and fix the IndicTrans2 output corruption upstream or evaluate another authorized candidate; do not repair translations by blind string replacement.
- Measure GPU peak memory on suitable NVIDIA hardware. CPU RAM and model-file size do not establish VRAM compliance.
