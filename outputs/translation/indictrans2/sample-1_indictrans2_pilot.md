# IndicTrans2 sample 1 pilot

**Status: completed on CPU. Pilot quality: UNVALIDATED. Overall candidate recommendation after full-corpus audit: NOT ACCEPTABLE — TRANSLATION QUALITY POOR.**

Model: `ai4bharat/indictrans2-indic-en-dist-200M`
Batching: five sentences in one ordered batch; beam size 5
Inference time: 4.071 seconds
Tokenizer load: 1.771 seconds; model load: 4.530 seconds
Peak process working set: 1,182.45 MB

| # | Hindi source sentence | Actual English translation |
|---:|---|---|
| 1 | उस समय गढ़ मुक्तेश्वर बल्लभ संप्रदाय का मुख्य केन्द्र था। | At that time Garh\ u093C was the main center of the Mukteshwar Ballabh sect. |
| 2 | गढ़ मुक्तेश्वर का नाम शिव बल्लभपुर पड़ा। | Garh\ u093C Mukteshwar got the name Shiv Ballabhpur. |
| 3 | ज्ञात रहे कि शिव बल्लभपुर को ही अब गढ़ मुक्तेश्वर के नाम से जाना जाता है। | It may be noted that Shiv Ballabhpur itself is now known as Garh Mukteshwar. |
| 4 | महाराज शिव ने भगवान परशुराम से यहाँ शिव मंदिर की स्थापना कराई थी। | Maharaj Shiva got Lord Parshuram to establish a Shiva temple here. |
| 5 | गंगा किनारे यह भगवान शंकर की प्रसिद्ध पुरी है। | It is the famous Puri of Lord Shankar on the banks of the Ganges. |

All five outputs are nonempty and aligned in order. No dropped or duplicated source sentence or repetition loop was found. The literal `\ u093C` in the first two translations mangles Garhmukteshwar and needs review. Other checked entities are rendered as Ballabh sect, Shiv Ballabhpur, Maharaj Shiva, Lord Parshuram, and Ganges/Lord Shankar. These are preliminary checks by an AI coding agent, not human validation.

The Helsinki pilot was severely malformed and omitted source content. M2M100 had a repetition loop in combined decoding and serious mistranslations in its per-sentence results, including an offensive rendering of “बल्लभ संप्रदाय.” IndicTrans2 is more coherent on this five-sentence pilot, but the full-corpus audit found malformed `u093C` fragments in 37 of 344 sentences across samples 1, 2, 4, 5, and 7. The defect remains unresolved; this pilot and all full-file translations remain **UNVALIDATED**.
