# Model 3: Hindi-to-English Translation Candidate Comparison

**Evaluation Date:** 2026-11-12 18:12:21
**Evaluation Scope:** 5-sentence ordered pilot from `outputs/pipeline_final/sample-1_final_output.txt`
**Execution Target:** Native Windows CPU (Intel Core i5-8365U, 4 threads)

---

## 1. Candidate Overview

| Property | Helsinki-NLP/opus-mt-hi-en | facebook/m2m100_418M (CTranslate2 INT8) |
|---|---|---|
| **Architecture** | MarianMT (6-layer enc-dec, ~75M params) | M2M100 (12-layer enc-dec, ~418M params) |
| **Inference Engine** | PyTorch native (`transformers`) | CTranslate2 (`ctranslate2.Translator`) |
| **Precision / Quantization** | FP32 | INT8 |
| **Model Disk Size** | ~300 MB | 470.73 MB |
| **Inference Speed (5 sentences)** | 3.316s | 3.809s (individual batch) / 5.485s (penalized passage) |
| **Source Tokens** | 83 | 94 |
| **GPU Memory Compliance** | NOT VERIFIED (CPU test) | **NOT VERIFIED** (CPU test) |
| **Quality Status** | **Unacceptable** | **Unacceptable** |

---

## 2. Side-by-Side Sentence Comparison

### Original Hindi Source Sentences:
1. `उस समय गढ़ मुक्तेश्वर बल्लभ संप्रदाय का मुख्य केन्द्र था।`
2. `गढ़ मुक्तेश्वर का नाम शिव बल्लभपुर पड़ा।`
3. `ज्ञात रहे कि शिव बल्लभपुर को ही अब गढ़ मुक्तेश्वर के नाम से जाना जाता है।`
4. `महाराज शिव ने भगवान परशुराम से यहाँ शिव मंदिर की स्थापना कराई थी।`
5. `गंगा किनारे यह भगवान शंकर की प्रसिद्ध पुरी है।`

### Helsinki-NLP Combined Translation:
> *"The fortress was then the center of a free God-star sect, and the name of the God of gods was found in the Shivpurpur. It was known that the Shivpur is now known as the Freema of Freema."*

### M2M100 (CTranslate2 INT8) Combined Passage Translation:
- **Standard Beam Search (beam=4):**
  > *"At that time, the main center of the Buddha was the Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha Buddha"*
  *(Catastrophic failure: model entered an infinite token repetition loop repeating "Buddha" until max decoding length was hit).*
- **With Repetition Penalty (penalty=1.2, no_repeat_ngram=3):**
  > *"At that time, the main center of the Buddha was the Buddhist Church. The Buddha's name was Shiva Bullbapur. It was known that Shiva Ballbapur is only now known by the name of Buddha. Maharashtra Shiva had established Shiva Temple here from God Parshuram."*
  *(Severe semantic hallucination: invented "the Buddha was the Buddhist Church", mistranslated "महाराज शिव" as "Maharashtra Shiva", and completely dropped sentence 5).*

### M2M100 Individual Sentence Translations:
1. **HI:** `उस समय गढ़ मुक्तेश्वर बल्लभ संप्रदाय का मुख्य केन्द्र था।`
   **EN:** `At that time, it was the main center of the bullshit community.`
   *(Severe error: 'बल्लभ संप्रदाय' is translated as vulgar phrase 'bullshit community').*
2. **HI:** `गढ़ मुक्तेश्वर का नाम शिव बल्लभपुर पड़ा।`
   **EN:** `The name of Shiva was Shiva.`
   *(Mangled tautology: loses 'Garhmukteshwar' and 'Vallabhpur').*
3. **HI:** `ज्ञात रहे कि शिव बल्लभपुर को ही अब गढ़ मुक्तेश्वर के नाम से जाना जाता है।`
   **EN:** `It is known that Shiva is now known by the name of the God.`
   *(Mangled clause: misses proper nouns).*
4. **HI:** `महाराज शिव ने भगवान परशुराम से यहाँ शिव मंदिर की स्थापना कराई थी।`
   **EN:** `Maharaj Shiva established the Shiva Temple here from God Perchuram.`
   *(Partially intelligible: identifies Maharaj Shiva, Shiva temple, and Parshuram as 'Perchuram').*
5. **HI:** `गंगा किनारे यह भगवान शंकर की प्रसिद्ध पुरी है।`
   **EN:** `This is the famous Puri of God Shankar.`
   *(Intelligible literal translation: 'famous Puri of God Shankar').*

---

## 3. Detailed Comparative Quality Assessment

### A. Completeness & Hallucinations
- **Helsinki-NLP:** Omitted sentences 4 and 5 entirely. Hallucinated "free God-star sect" and "Freema of Freema".
- **M2M100 Combined:** Under standard beam search, completely degenerated into repeating "Buddha". Under anti-repetition decoding, hallucinated Buddhist entities that do not exist anywhere in the Shaivite source text ("the Buddha was the Buddhist Church", "name of Buddha"), confused "महाराज शिव" with the Indian state of "Maharashtra", and dropped sentence 5.
- **M2M100 Individual:** Avoided passage degeneration, but produced an unacceptable profanity ("bullshit community") for the revered Vaishnava/Hindu Vallabh sect ("बल्लभ संप्रदाय") due to subword alignment failure.

### B. Suitability for Project
Neither candidate demonstrates adequate domain vocabulary or syntactic stability for Hindi devotional, historical, and geographical prose:
- Both fail to recognize important historical place names like **Garhmukteshwar (गढ़ मुक्तेश्वर)**.
- Both hallucinate or drop sentences on multi-sentence inputs.
- M2M100 introduces vulgar artifacts when translating isolated sentences.

---

## 4. Benchmark & Hardware Evidence

- **CPU Model:** Intel Core i5-8365U (4 physical cores / 8 threads, 1.60 GHz base)
- **Model Loading Latency:** 2.51 s
- **Batch Inference Latency (5 sentences):** 4.649 s (21.94 tokens/sec)
- **Passage Inference Latency:** 7.483 s
- **Peak Working Set RAM:** 1089.02 MB
- **Model Disk Footprint:** 470.73 MB
- **GPU VRAM Compliance:** **NOT VERIFIED** (No dedicated NVIDIA GPU hardware on test system).

---

## 5. Recommendation

**Status:** `CANDIDATE NOT ACCEPTABLE - TRANSLATION QUALITY POOR`

While the CTranslate2 INT8 technical pipeline functioned smoothly on native Windows (efficient CPU memory footprint of ~1088 MB peak RAM and ~470 MB disk footprint), the actual translation outputs from `facebook/m2m100_418M` are deeply degraded, including an infinite repetition loop on continuous text, severe hallucinations, and a vulgar mistranslation. It is not viable for this project.
