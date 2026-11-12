# -*- coding: utf-8 -*-
"""
Model 3: Translation Candidate Smoke Test (Helsinki-NLP/opus-mt-hi-en).

Translates the first five ordered sentences from sample-1_final_output.txt on CPU.
Checks input truncation against model maximum supported length.
Outputs pilot translation JSON to outputs/translation/sample-1_pilot.json.
"""

import json
import os
import sys
import time
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "outputs" / "pipeline_final" / "sample-1_final_output.txt"
OUTPUT_DIR = BASE_DIR / "outputs" / "translation"
OUTPUT_FILE = OUTPUT_DIR / "sample-1_pilot.json"

MODEL_ID = "Helsinki-NLP/opus-mt-hi-en"
DEVICE = "cpu"
NUM_SENTENCES = 5


def main():
    print("=" * 70)
    print("Model 3: Translation Candidate Smoke Test")
    print(f"  Model ID: {MODEL_ID}")
    print(f"  Device  : {DEVICE}")
    print("=" * 70)

    if not INPUT_FILE.is_file():
        print(f"Error: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    # 1. Read input file
    lines = [line.strip() for line in INPUT_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < NUM_SENTENCES:
        print(f"Error: Expected at least {NUM_SENTENCES} sentences, found {len(lines)}")
        sys.exit(1)

    selected_sentences = lines[:NUM_SENTENCES]
    # Combine preserving original order (separated by space)
    combined_hindi_text = " ".join(selected_sentences)

    print(f"\nOriginal Hindi Sentences ({NUM_SENTENCES}):")
    for idx, sent in enumerate(selected_sentences, 1):
        print(f"  [{idx}] {sent}")

    print(f"\nCombined Input Text:\n  {combined_hindi_text}")

    # 2. Load model and tokenizer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    print(f"\nLoading model and tokenizer: {MODEL_ID} (device={DEVICE}) ...")
    t_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()
    t_load = time.time() - t_load_start
    print(f"Model loaded in {t_load:.2f}s")

    # 3. Tokenize without forced truncation to inspect exact source token count
    tokens_untruncated = tokenizer(combined_hindi_text, return_tensors="pt", truncation=False)
    source_token_count = tokens_untruncated["input_ids"].shape[1]
    model_max_length = getattr(tokenizer, "model_max_length", 512)
    # Ensure standard reasonable ceiling if tokenizer reports very large integer
    if model_max_length > 10000:
        model_max_length = getattr(model.config, "max_position_embeddings", 512)

    is_truncated = source_token_count > model_max_length
    print(f"\nToken Length Analysis:")
    print(f"  Source token count   : {source_token_count}")
    print(f"  Supported input limit: {model_max_length}")
    print(f"  Input truncated      : {is_truncated}")

    if is_truncated:
        print(f"  WARNING: Input length ({source_token_count}) exceeds supported limit ({model_max_length})!")
    else:
        print(f"  Input safely within limit ({source_token_count} <= {model_max_length}).")

    # 4. Prepare model inputs (with truncation=True for standard safety)
    inputs = tokenizer(combined_hindi_text, return_tensors="pt", truncation=True, max_length=model_max_length)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 5. Translate
    print("\nRunning translation inference...")
    t_infer_start = time.time()
    with torch.no_grad():
        translated_tokens = model.generate(**inputs, max_length=512)
    t_inference = time.time() - t_infer_start

    english_translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True).strip()

    # 6. Print summary
    print("\n" + "=" * 70)
    print("Translation Result:")
    print(f"  Model ID          : {MODEL_ID}")
    print(f"  Device            : {DEVICE}")
    print(f"  Source token count: {source_token_count}")
    print(f"  Inference time    : {t_inference:.3f}s")
    print(f"  Truncation status : {'EXCEEDS LIMIT' if is_truncated else 'NOT TRUNCATED'}")
    print(f"\nEnglish Translation:\n  {english_translation}")
    print("=" * 70)

    # 7. Save result
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_data = {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "source_file": str(INPUT_FILE.relative_to(BASE_DIR)),
        "sentence_count": NUM_SENTENCES,
        "original_sentences": selected_sentences,
        "combined_input_text": combined_hindi_text,
        "source_token_count": source_token_count,
        "model_max_length": model_max_length,
        "is_truncated": is_truncated,
        "inference_time_seconds": round(t_inference, 4),
        "english_translation": english_translation,
    }

    OUTPUT_FILE.write_text(json.dumps(result_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPilot translation output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
