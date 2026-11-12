# Pretrained models

Model-card information was checked against the linked Hugging Face pages. Parameter counts are reported only where the card names a size or where the local loaded checkpoint was counted. A model page's `License` field is not a statement that this project obtained additional permission beyond the listed license and any separate gated-access terms.

## Models used in the workflow

| Role | Model and official card | Architecture / size | Language and task | License | Project status |
|---|---|---|---|---|---|
| Punctuation restoration | [Horizon-Labs/punctuation-restoration-small](https://huggingface.co/Horizon-Labs/punctuation-restoration-small) | mmBERT-small token classifier; model card describes the small version as 140M | Card describes punctuation after tokens for 88 languages; Hindi tested in this project | Apache-2.0 | Implemented and measured on a provisional 55-record Hindi reference set |
| Selected sentence encoder | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | BERT encoder with 12 layers, hidden size 384, 12 attention heads; mean pooling produces 384-d vectors; parameter count not stated on card | 50 languages; paraphrase/semantic sentence embeddings | Apache-2.0 | Used as practical semantic-affinity heuristic; order accuracy unavailable |
| Ordering comparison | [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) | Multilingual BERT encoder; model card lists about 0.2B parameters; project loads `BertForNextSentencePrediction` and verifies NSP-head weights | 104 languages; directional next-sentence score | Apache-2.0 | Optional experimental comparator; no ordering accuracy claim |

## Translation candidates evaluated

| Candidate | Card-stated architecture/size | Languages | License | Local result |
|---|---|---|---|---|
| [AI4Bharat IndicTrans2 Indic-En Distilled 200M](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) | Distilled encoder-decoder seq2seq variant named 200M; the locally loaded checkpoint counted 211,780,608 parameters | Card lists 26 languages; Hindi-to-English run used `hin_Deva` → `eng_Latn` | MIT; model access is separately gated by Hugging Face terms | Rejected for final-quality use: 37/344 translations contain malformed `u093C` fragments and semantic errors are present; outputs remain UNVALIDATED |
| [Helsinki-NLP/opus-mt-hi-en](https://huggingface.co/Helsinki-NLP/opus-mt-hi-en) | OPUS-MT `transformer-align` model; parameter count not stated on the card | Hindi (`hi`) → English (`en`) | Apache-2.0 | Rejected after a malformed pilot that omitted source content |
| [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M) | Multilingual encoder-decoder, 418M variant | 100 languages; model card includes Hindi and English | MIT | Rejected after repetition-loop and severe sentence-level errors; CTranslate2 INT8 pilot artifacts are preserved |

### Specification notes

- The punctuation card labels the model “140M” and says it is built on mmBERT-small; no exact runtime parameter count is asserted here.
- The MiniLM card states the 384-dimensional embedding size and shows a 12-layer `BertModel` plus mean pooling. Its current model card does not publish a parameter count.
- IndicTrans2's card names the distilled 200M model and labels its task as translation/text-to-text. The project's loaded model class and parameter count are recorded in `outputs/translation/indictrans2/model3_indictrans2_benchmark.json`.
- Helsinki's card lists `transformer-align` and the `marian` tag; no parameter count is asserted.
- M2M100's card describes multilingual encoder-decoder translation and uses a 418M model variant name. This project benchmarked a separately converted CTranslate2 INT8 package; its approximately 470.73 MiB disk size is not a GPU-memory measurement.
- The punctuation card headline and repository metadata differ in their language-count wording (88/89); this documentation avoids relying on an exact count for the project result.

## Official references

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Sentence Transformers documentation](https://sbert.net/docs/)
- [AI4Bharat IndicTrans2 repository](https://github.com/AI4Bharat/IndicTrans2)
- [IndicTransToolkit PyPI documentation](https://pypi.org/project/indictranstoolkit/) and [source repository](https://github.com/VarunGumma/IndicTransToolkit)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)

## Resource caveat

The project target is under 1.5 GB GPU memory for each pretrained model. No CUDA-capable NVIDIA GPU was available, so **GPU-memory compliance: NOT VERIFIED** for all candidates. Parameter count, CPU process RAM, and downloaded checkpoint size cannot substitute for a measured GPU peak.
