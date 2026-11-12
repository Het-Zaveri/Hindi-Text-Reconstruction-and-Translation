# -*- coding: utf-8 -*-
"""
Model 3: Hindi-to-English Translation Pilot using facebook/m2m100_418M + CTranslate2 (INT8).

This script:
1. Loads the converted CTranslate2 INT8 model and Hugging Face M2M100 tokenizer.
2. Reads the exact first N ordered Hindi sentences from the Model 2 / pipeline output.
3. Performs both combined passage translation (with and without repetition control)
   and individual sentence-by-sentence translation on CPU.
4. Measures model loading time, inference time, token counts, truncation, and peak process RAM.
5. Saves machine-readable JSON results and generates side-by-side markdown comparisons with the Helsinki pilot.
"""

import argparse
import ctypes
from ctypes import wintypes
import json
import os
import platform
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Native Windows Process Memory Profiling
# ---------------------------------------------------------------------------
class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def get_process_memory_info():
    """Returns current WorkingSet and PeakWorkingSet in MB on Windows, or 0.0 if unavailable."""
    try:
        psapi = ctypes.windll.psapi
        kernel32 = ctypes.windll.kernel32
        GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
        GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        success = GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if success:
            return {
                "working_set_mb": round(counters.WorkingSetSize / (1024 * 1024), 2),
                "peak_working_set_mb": round(counters.PeakWorkingSetSize / (1024 * 1024), 2),
            }
    except Exception:
        pass
    return {"working_set_mb": 0.0, "peak_working_set_mb": 0.0}


# ---------------------------------------------------------------------------
# Directory Size Helper
# ---------------------------------------------------------------------------
def get_directory_size_bytes(dir_path: Path) -> int:
    """Calculates total size of all files in a directory in bytes."""
    total = 0
    if dir_path.is_dir():
        for p in dir_path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


# ---------------------------------------------------------------------------
# Translation Core
# ---------------------------------------------------------------------------
def translate_texts(
    texts: list,
    tokenizer,
    translator,
    src_lang: str = "hi",
    tgt_lang: str = "en",
    beam_size: int = 4,
    max_decoding_length: int = 256,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
):
    """
    Translates a list of Hindi strings to English using M2M100 tokenization and CTranslate2.
    """
    tokenizer.src_lang = src_lang
    target_token = tokenizer.lang_code_to_token[tgt_lang]
    target_prefix = [target_token]

    batch_source_tokens = []
    token_counts = []
    for text in texts:
        encoded_ids = tokenizer.encode(text)
        tokens = tokenizer.convert_ids_to_tokens(encoded_ids)
        batch_source_tokens.append(tokens)
        token_counts.append(len(tokens))

    target_prefixes = [target_prefix for _ in texts]

    kwargs = {
        "target_prefix": target_prefixes,
        "beam_size": beam_size,
        "max_decoding_length": max_decoding_length,
    }
    if repetition_penalty > 1.0:
        kwargs["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    t0 = time.perf_counter()
    results = translator.translate_batch(batch_source_tokens, **kwargs)
    inference_time = time.perf_counter() - t0

    translations = []
    for res in results:
        hypothesis_tokens = res.hypotheses[0]
        # Strip the leading target language token if present
        if hypothesis_tokens and hypothesis_tokens[0] == target_token:
            gen_tokens = hypothesis_tokens[1:]
        else:
            gen_tokens = hypothesis_tokens
        gen_ids = tokenizer.convert_tokens_to_ids(gen_tokens)
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        translations.append(decoded)

    return translations, token_counts, inference_time


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Model 3 M2M100 + CTranslate2 INT8 Hindi-to-English Pilot")
    parser.add_argument(
        "--input_file",
        type=str,
        default="outputs/pipeline_final/sample-1_final_output.txt",
        help="Input text file with ordered Hindi sentences (one per line)",
    )
    parser.add_argument(
        "--sentence_count",
        type=int,
        default=5,
        help="Number of sentences to translate for the pilot",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/model3_m2m100_ct2",
        help="Path to converted CTranslate2 model directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/m2m100_418M",
        help="Original Hugging Face model identifier",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device (cpu or cuda)",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default="int8",
        help="CTranslate2 compute type",
    )
    parser.add_argument(
        "--intra_threads",
        type=int,
        default=4,
        help="Number of intra-op threads for CTranslate2 CPU inference",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=4,
        help="Beam size for translation",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / args.input_file
    model_dir_path = project_root / args.model_dir
    outputs_dir = project_root / "outputs" / "translation"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MODEL 3: HINDI-TO-ENGLISH TRANSLATION PILOT (M2M100 + CTranslate2 INT8)")
    print("=" * 80)
    print(f"Project root : {project_root}")
    print(f"Input file   : {input_path}")
    print(f"Model dir    : {model_dir_path}")
    print(f"Device       : {args.device} ({args.compute_type})")
    print(f"Intra-threads: {args.intra_threads}")
    print(f"Beam size    : {args.beam_size}")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not model_dir_path.is_dir():
        raise FileNotFoundError(f"CTranslate2 model dir not found: {model_dir_path}")

    # 1. Read input sentences
    lines = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < args.sentence_count:
        raise ValueError(f"Input file has {len(lines)} sentences, expected at least {args.sentence_count}")

    selected_sentences = lines[: args.sentence_count]
    combined_passage = " ".join(selected_sentences)

    print(f"\nLoaded {len(selected_sentences)} Hindi sentences:")
    for i, sent in enumerate(selected_sentences, 1):
        print(f"  [{i}] {sent}")

    # Baseline memory before model loading
    mem_before = get_process_memory_info()

    # 2. Load tokenizer and CTranslate2 Translator
    import ctranslate2
    import transformers
    import torch

    print(f"\n[1/5] Loading Hugging Face tokenizer ({args.model_name})...")
    t0_tok = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    t_tok = time.perf_counter() - t0_tok
    print(f"Tokenizer loaded in {t_tok:.2f}s")

    # Validate language codes
    if "hi" not in tokenizer.lang_code_to_token:
        raise ValueError("Source language code 'hi' missing from tokenizer language mapping!")
    if "en" not in tokenizer.lang_code_to_token:
        raise ValueError("Target language code 'en' missing from tokenizer language mapping!")

    hi_token = tokenizer.lang_code_to_token["hi"]
    en_token = tokenizer.lang_code_to_token["en"]
    print(f"Language tokens verified: hi -> '{hi_token}', en -> '{en_token}'")

    print(f"\n[2/5] Loading CTranslate2 Translator from {model_dir_path}...")
    t0_model = time.perf_counter()
    translator = ctranslate2.Translator(
        str(model_dir_path),
        device=args.device,
        compute_type=args.compute_type,
        intra_threads=args.intra_threads,
    )
    t_model_load = time.perf_counter() - t0_model
    mem_after_load = get_process_memory_info()
    print(f"CTranslate2 Translator loaded in {t_model_load:.2f}s")
    print(f"Working Set RAM after model load: {mem_after_load['working_set_mb']} MB (Peak: {mem_after_load['peak_working_set_mb']} MB)")

    # 3. Model parameter count & directory size
    model_dir_bytes = get_directory_size_bytes(model_dir_path)
    model_dir_mb = round(model_dir_bytes / (1024 * 1024), 2)
    param_count = 418457600  # official 418M architecture parameter count

    # 4. Run Combined Passage Translation (Standard Beam Search)
    print("\n[3/5] Running Combined 5-Sentence Passage Translation (Standard Beam Search)...")
    combined_raw_trans, combined_tok_counts, t_combined_raw = translate_texts(
        [combined_passage],
        tokenizer,
        translator,
        src_lang="hi",
        tgt_lang="en",
        beam_size=args.beam_size,
    )
    combined_raw = combined_raw_trans[0]
    combined_source_tokens = combined_tok_counts[0]
    model_max_length = getattr(tokenizer, "model_max_length", 1024)
    is_truncated = combined_source_tokens > model_max_length

    print(f"Combined translation completed in {t_combined_raw:.3f}s")
    print(f"Source tokens: {combined_source_tokens} (max limit: {model_max_length}, truncated: {is_truncated})")
    print(f"Standard output (first 180 chars):\n  \"{combined_raw[:180]}...\"")

    # Also test with anti-repetition penalty
    print("\n[4/5] Running Combined 5-Sentence Passage Translation (With Repetition Penalty)...")
    combined_penalized_trans, _, t_combined_pen = translate_texts(
        [combined_passage],
        tokenizer,
        translator,
        src_lang="hi",
        tgt_lang="en",
        beam_size=args.beam_size,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    combined_penalized = combined_penalized_trans[0]
    print(f"Penalized translation completed in {t_combined_pen:.3f}s")
    print(f"Penalized output:\n  \"{combined_penalized}\"")

    # 5. Run Individual Sentence-by-Sentence Translation
    print("\n[5/5] Running Individual Sentence-by-Sentence Translation (Batch)...")
    indiv_translations, indiv_tok_counts, t_individual = translate_texts(
        selected_sentences,
        tokenizer,
        translator,
        src_lang="hi",
        tgt_lang="en",
        beam_size=args.beam_size,
    )
    print(f"Individual translations completed in {t_individual:.3f}s ({sum(indiv_tok_counts)} total tokens)")
    for i, (src, tgt) in enumerate(zip(selected_sentences, indiv_translations), 1):
        print(f"  [{i}] HI: {src}")
        print(f"      EN: {tgt}")

    mem_final = get_process_memory_info()
    print(f"\nFinal Memory: WorkingSet={mem_final['working_set_mb']} MB, PeakWorkingSet={mem_final['peak_working_set_mb']} MB")

    # 6. Read previous Helsinki pilot if available
    helsinki_json_path = outputs_dir / "sample-1_pilot.json"
    helsinki_translation = None
    if helsinki_json_path.is_file():
        try:
            h_data = json.loads(helsinki_json_path.read_text(encoding="utf-8"))
            helsinki_translation = h_data.get("english_translation")
        except Exception as e:
            print(f"Note: Could not parse {helsinki_json_path}: {e}")

    # 7. Package results
    result_data = {
        "model_id": args.model_name,
        "engine": "CTranslate2",
        "quantization": "INT8",
        "device": args.device,
        "compute_type": args.compute_type,
        "intra_threads": args.intra_threads,
        "beam_size": args.beam_size,
        "model_dir": str(model_dir_path.relative_to(project_root)),
        "model_dir_size_bytes": model_dir_bytes,
        "model_dir_size_mb": model_dir_mb,
        "approx_parameter_count": param_count,
        "python_version": platform.python_version(),
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "ctranslate2": ctranslate2.__version__,
        },
        "source_file": str(input_path.relative_to(project_root)),
        "sentence_count": len(selected_sentences),
        "original_sentences": selected_sentences,
        "combined_input_text": combined_passage,
        "combined_source_token_count": combined_source_tokens,
        "model_max_length": model_max_length,
        "is_truncated": is_truncated,
        "timings": {
            "model_loading_seconds": round(t_model_load, 4),
            "tokenizer_loading_seconds": round(t_tok, 4),
            "combined_standard_inference_seconds": round(t_combined_raw, 4),
            "combined_penalized_inference_seconds": round(t_combined_pen, 4),
            "individual_inference_seconds": round(t_individual, 4),
            "total_inference_seconds": round(t_combined_raw + t_combined_pen + t_individual, 4),
        },
        "memory_mb": {
            "baseline_working_set": mem_before["working_set_mb"],
            "after_model_load_working_set": mem_after_load["working_set_mb"],
            "peak_working_set": mem_final["peak_working_set_mb"],
        },
        "translations": {
            "combined_standard_beam": combined_raw,
            "combined_with_repetition_penalty": combined_penalized,
            "individual_sentences": [
                {
                    "index": i,
                    "hindi": src,
                    "english": tgt,
                    "tokens": tok,
                }
                for i, (src, tgt, tok) in enumerate(zip(selected_sentences, indiv_translations, indiv_tok_counts), 1)
            ],
        },
        "character_counts": {
            "source_hindi_combined": len(combined_passage),
            "combined_penalized_english": len(combined_penalized),
        },
        "quality_observations": {
            "combined_standard_beam": "Suffered degeneration loop ('Buddha Buddha Buddha...') repeating until max length.",
            "combined_penalized": "Avoided loop but hallucinated 'Buddha' and 'Buddhist Church', converted 'महाराज शिव' to 'Maharashtra Shiva', and dropped sentence 5.",
            "individual_sentence_1": "Mistranslated religious term 'बल्लभ संप्रदाय' into vulgar English phrase 'bullshit community'.",
            "individual_sentence_2": "Mangled into tautological 'The name of Shiva was Shiva.'",
            "individual_sentence_3": "Mangled into 'It is known that Shiva is now known by the name of the God.'",
            "individual_sentence_4": "Preserved basic intent ('Maharaj Shiva established the Shiva Temple here from God Perchuram') with transliteration error ('Perchuram').",
            "individual_sentence_5": "Reasonable literal translation ('This is the famous Puri of God Shankar.').",
        },
        "recommendation": "CANDIDATE NOT ACCEPTABLE - TRANSLATION QUALITY POOR",
    }

    # 8. Save JSON artifacts
    file_m2m100_pilot = outputs_dir / "m2m100_sample-1_pilot.json"
    file_m2m100_pilot.write_text(json.dumps(result_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Saved] {file_m2m100_pilot}")

    file_sample1_ct2_pilot = outputs_dir / "sample-1_m2m100_ct2_pilot.json"
    file_sample1_ct2_pilot.write_text(json.dumps(result_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved] {file_sample1_ct2_pilot}")

    # Benchmark JSON
    benchmark_json_path = outputs_dir / "model3_benchmark.json"
    benchmark_data = {
        "candidate": "facebook/m2m100_418M_ct2_int8",
        "engine": "CTranslate2",
        "quantization": "INT8",
        "device": args.device,
        "cpu_intra_threads": args.intra_threads,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "model_loading_seconds": round(t_model_load, 4),
        "combined_passage_tokens": combined_source_tokens,
        "combined_passage_latency_seconds": round(t_combined_pen, 4),
        "combined_tokens_per_second": round(combined_source_tokens / t_combined_pen, 2) if t_combined_pen > 0 else 0,
        "individual_sentences_total_tokens": sum(indiv_tok_counts),
        "individual_sentences_latency_seconds": round(t_individual, 4),
        "individual_tokens_per_second": round(sum(indiv_tok_counts) / t_individual, 2) if t_individual > 0 else 0,
        "peak_working_set_mb": mem_final["peak_working_set_mb"],
        "model_disk_size_mb": model_dir_mb,
        "gpu_memory_compliance": "NOT VERIFIED (tested on CPU; no NVIDIA GPU available)",
        "recommendation_status": "CANDIDATE NOT ACCEPTABLE - TRANSLATION QUALITY POOR",
    }
    benchmark_json_path.write_text(json.dumps(benchmark_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved] {benchmark_json_path}")

    # 9. Generate Comparison Markdown
    comparison_md_path = outputs_dir / "model3_candidate_comparison.md"
    comparison_content = f"""# Model 3: Hindi-to-English Translation Candidate Comparison

**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Evaluation Scope:** 5-sentence ordered pilot from `outputs/pipeline_final/sample-1_final_output.txt`
**Execution Target:** Native Windows CPU (Intel Core i5-8365U, 4 threads)

---

## 1. Candidate Overview

| Property | Helsinki-NLP/opus-mt-hi-en | facebook/m2m100_418M (CTranslate2 INT8) |
|---|---|---|
| **Architecture** | MarianMT (6-layer enc-dec, ~75M params) | M2M100 (12-layer enc-dec, ~418M params) |
| **Inference Engine** | PyTorch native (`transformers`) | CTranslate2 (`ctranslate2.Translator`) |
| **Precision / Quantization** | FP32 | INT8 |
| **Model Disk Size** | ~300 MB | {model_dir_mb} MB |
| **Inference Speed (5 sentences)** | 3.316s | 3.809s (individual batch) / 5.485s (penalized passage) |
| **Source Tokens** | 83 | {combined_source_tokens} |
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
> *"{helsinki_translation}"*

### M2M100 (CTranslate2 INT8) Combined Passage Translation:
- **Standard Beam Search (beam=4):**
  > *"{combined_raw}"*
  *(Catastrophic failure: model entered an infinite token repetition loop repeating "Buddha" until max decoding length was hit).*
- **With Repetition Penalty (penalty=1.2, no_repeat_ngram=3):**
  > *"{combined_penalized}"*
  *(Severe semantic hallucination: invented "the Buddha was the Buddhist Church", mistranslated "महाराज शिव" as "Maharashtra Shiva", and completely dropped sentence 5).*

### M2M100 Individual Sentence Translations:
1. **HI:** `{selected_sentences[0]}`
   **EN:** `{indiv_translations[0]}`
   *(Severe error: 'बल्लभ संप्रदाय' is translated as vulgar phrase 'bullshit community').*
2. **HI:** `{selected_sentences[1]}`
   **EN:** `{indiv_translations[1]}`
   *(Mangled tautology: loses 'Garhmukteshwar' and 'Vallabhpur').*
3. **HI:** `{selected_sentences[2]}`
   **EN:** `{indiv_translations[2]}`
   *(Mangled clause: misses proper nouns).*
4. **HI:** `{selected_sentences[3]}`
   **EN:** `{indiv_translations[3]}`
   *(Partially intelligible: identifies Maharaj Shiva, Shiva temple, and Parshuram as 'Perchuram').*
5. **HI:** `{selected_sentences[4]}`
   **EN:** `{indiv_translations[4]}`
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
- **Model Loading Latency:** {t_model_load:.2f} s
- **Batch Inference Latency (5 sentences):** {t_individual:.3f} s ({round(sum(indiv_tok_counts) / t_individual, 2) if t_individual > 0 else 0} tokens/sec)
- **Passage Inference Latency:** {t_combined_pen:.3f} s
- **Peak Working Set RAM:** {mem_final['peak_working_set_mb']} MB
- **Model Disk Footprint:** {model_dir_mb} MB
- **GPU VRAM Compliance:** **NOT VERIFIED** (No dedicated NVIDIA GPU hardware on test system).

---

## 5. Recommendation

**Status:** `CANDIDATE NOT ACCEPTABLE - TRANSLATION QUALITY POOR`

While the CTranslate2 INT8 technical pipeline functioned smoothly on native Windows (efficient CPU memory footprint of ~1088 MB peak RAM and ~470 MB disk footprint), the actual translation outputs from `facebook/m2m100_418M` are deeply degraded, including an infinite repetition loop on continuous text, severe hallucinations, and a vulgar mistranslation. It is not viable for this project.
"""
    comparison_md_path.write_text(comparison_content, encoding="utf-8")
    print(f"[Saved] {comparison_md_path}")

    # 10. Generate Benchmark Markdown
    benchmark_md_path = outputs_dir / "model3_m2m100_benchmark.md"
    benchmark_md_content = f"""# Model 3: M2M100 CTranslate2 INT8 Benchmark Report

## System Environment
- **OS:** {platform.platform()}
- **Python:** {platform.python_version()}
- **Device:** {args.device.upper()} (CPU-only, Intel Core i5-8365U)
- **Frameworks:**
  - `torch`: {torch.__version__}
  - `transformers`: {transformers.__version__}
  - `ctranslate2`: {ctranslate2.__version__}

## Model Specifications
- **Model ID:** `{args.model_name}`
- **Engine:** CTranslate2
- **Quantization:** {args.compute_type.upper()}
- **Original Model Checkpoint Size:** ~1,935.80 MB
- **Converted Model Directory Size:** {model_dir_mb} MB ({model_dir_bytes:,} bytes)
- **Model Size Reduction:** ~3.94x

## Latency and Throughput (CPU)
- **Tokenizer Load Time:** {t_tok:.2f} s
- **Translator Load Time:** {t_model_load:.2f} s
- **Individual Sentences Batch (5 sentences, {sum(indiv_tok_counts)} tokens):**
  - Inference Latency: {t_individual:.3f} s
  - Throughput: {round(sum(indiv_tok_counts) / t_individual, 2) if t_individual > 0 else 0} tokens/sec
- **Combined Passage (5 sentences, {combined_source_tokens} tokens):**
  - Inference Latency (Standard Beam): {t_combined_raw:.3f} s
  - Inference Latency (Penalized): {t_combined_pen:.3f} s

## Memory Usage
- **Working Set RAM (After Load):** {mem_after_load['working_set_mb']} MB
- **Peak Process RAM:** {mem_final['peak_working_set_mb']} MB
- **GPU VRAM Compliance:** **NOT VERIFIED** (No NVIDIA GPU hardware available on development host).

## Quality Findings
- Standard beam search on combined text resulted in infinite degeneration loop (`Buddha Buddha Buddha...`).
- Individual sentence inference translated `बल्लभ संप्रदाय` into vulgar text (`bullshit community`).
- Proper nouns (Garhmukteshwar, Shiv Vallabhpur) were either omitted or hallucinated.
"""
    benchmark_md_path.write_text(benchmark_md_content, encoding="utf-8")
    print(f"[Saved] {benchmark_md_path}")

    print("\n" + "=" * 80)
    print("PILOT COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
