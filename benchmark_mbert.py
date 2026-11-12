# -*- coding: utf-8 -*-
"""
mBERT NSP CPU Benchmark — 12-sentence subset of sample-1.

Measures actual inference cost with dynamic padding and max_length=64.
Runs batch sizes 8, 16, 32 on the SAME 12 sentences (132 directional pairs).
Reports: sentence count, pair count, elapsed seconds, pairs/sec, matrix shape,
         truncation rate, batch size, device, installed package versions,
         NSP head weight norm, and asymmetry check.

If inference fails for any reason, the benchmark is marked BLOCKED and the
exact error is printed — no results are fabricated.
"""

import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Project root is the directory containing this script
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_PATH = PROJECT_ROOT / "benchmark_mbert_results.json"
LOG_PATH = PROJECT_ROOT / "benchmark_mbert.log"

PUNCTUATED_FILE = PROJECT_ROOT / "outputs" / "sample-1_punctuated.txt"
SUBSET_SIZE = 12          # number of sentences in the benchmark subset
BATCH_SIZES = [8, 16, 32]
MAX_LENGTH = 64
DEVICE = "cpu"


def get_package_version(pkg_name: str) -> str:
    try:
        return importlib.import_module(pkg_name).__version__
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return "unknown"


def count_truncated_pairs(sentences, tokenizer, max_length):
    """Count how many sentence pairs are truncated at max_length tokens."""
    truncated = 0
    total = len(sentences) * (len(sentences) - 1)
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            enc = tokenizer(
                sentences[i], sentences[j],
                add_special_tokens=True,
                truncation=False,
            )
            if len(enc["input_ids"]) > max_length:
                truncated += 1
    return truncated, total


def run_benchmark():
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("mBERT NSP CPU Benchmark")
    log(f"  Subset size : {SUBSET_SIZE} sentences ({SUBSET_SIZE*(SUBSET_SIZE-1)} directional pairs)")
    log(f"  max_length  : {MAX_LENGTH}")
    log(f"  batch_sizes : {BATCH_SIZES}")
    log(f"  Device      : {DEVICE}")
    log("=" * 70)

    # --- Package versions ---
    versions = {
        "transformers": get_package_version("transformers"),
        "tokenizers":   get_package_version("tokenizers"),
        "huggingface_hub": get_package_version("huggingface_hub"),
        "torch":        get_package_version("torch"),
        "sentence_transformers": get_package_version("sentence_transformers"),
        "numpy":        get_package_version("numpy"),
    }
    log("\nPackage versions:")
    for k, v in versions.items():
        log(f"  {k}: {v}")

    # --- Check input file ---
    if not PUNCTUATED_FILE.is_file():
        log(f"\nBLOCKED: Punctuated file not found: {PUNCTUATED_FILE}")
        result = {"status": "BLOCKED", "error": f"File not found: {PUNCTUATED_FILE}",
                  "versions": versions}
        RESULTS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        LOG_PATH.write_text("\n".join(log_lines))
        return

    # --- Segmentation ---
    try:
        from src.model2_ordering import segment_sentences, verify_and_load_mbert_nsp, compute_pairwise_nsp_matrix
    except Exception as e:
        log(f"\nBLOCKED: Import error: {e}\n{traceback.format_exc()}")
        result = {"status": "BLOCKED", "error": str(e), "traceback": traceback.format_exc(),
                  "versions": versions}
        RESULTS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        LOG_PATH.write_text("\n".join(log_lines))
        return

    punc_text = PUNCTUATED_FILE.read_text(encoding="utf-8").strip()
    all_sentences = segment_sentences(punc_text)
    total_sentences = len(all_sentences)
    log(f"\nFull sample-1 segmentation: {total_sentences} sentences")
    log(f"  Expected per spec: 49")
    if total_sentences != 49:
        log(f"  WARNING: got {total_sentences}, spec says 49 — continuing with actual count")

    # Take first SUBSET_SIZE sentences
    sentences = all_sentences[:SUBSET_SIZE]
    n = len(sentences)
    expected_pairs = n * (n - 1)
    log(f"\nSubset: {n} sentences → {expected_pairs} directional pairs")
    content_hash = hashlib.sha256("\x00".join(sentences).encode("utf-8")).hexdigest()[:16]
    log(f"  Content hash (first 16 chars): {content_hash}")
    log(f"\nSentences in subset:")
    for idx, s in enumerate(sentences, 1):
        log(f"  [{idx:02d}] {s[:80]}{'...' if len(s)>80 else ''}")

    # --- Load model ---
    log("\nLoading mBERT NSP model...")
    try:
        model, tokenizer, vinfo = verify_and_load_mbert_nsp(device=DEVICE)
    except Exception as e:
        log(f"\nBLOCKED: Model loading failed: {e}\n{traceback.format_exc()}")
        result = {"status": "BLOCKED", "error": str(e), "traceback": traceback.format_exc(),
                  "versions": versions}
        RESULTS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        LOG_PATH.write_text("\n".join(log_lines))
        return

    log(f"  Model ID   : {vinfo['model_id']}")
    log(f"  Head status: {vinfo['status']}")
    log(f"  Weight norm: {vinfo['weight_norm']}  Bias norm: {vinfo['bias_norm']}")

    # --- Truncation measurement ---
    log(f"\nMeasuring truncation at max_length={MAX_LENGTH}...")
    try:
        trunc_count, trunc_total = count_truncated_pairs(sentences, tokenizer, MAX_LENGTH)
        trunc_rate = trunc_count / trunc_total if trunc_total > 0 else 0.0
        log(f"  Pairs exceeding {MAX_LENGTH} tokens: {trunc_count}/{trunc_total} ({trunc_rate:.1%})")
    except Exception as e:
        log(f"  WARNING: Truncation measurement failed: {e}")
        trunc_count, trunc_total, trunc_rate = -1, expected_pairs, -1.0

    # --- Benchmarks ---
    bench_results = []
    for batch_size in BATCH_SIZES:
        log(f"\n--- Batch size {batch_size} ---")
        try:
            t0 = time.perf_counter()
            matrix = compute_pairwise_nsp_matrix(
                sentences=sentences,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                device=DEVICE,
                cache_path=None,        # force real inference, no cache
                max_length=MAX_LENGTH,
                model_id=vinfo["model_id"],
            )
            elapsed = time.perf_counter() - t0

            actual_pairs = n * (n - 1)
            rate = actual_pairs / elapsed if elapsed > 0 else 0.0

            # Asymmetry check: for every off-diagonal pair verify M[i,j] != M[j,i]
            asym_violations = 0
            for i in range(n):
                for j in range(n):
                    if i != j and matrix[i, j] == matrix[j, i]:
                        asym_violations += 1
            asym_ok = asym_violations == 0

            # Sample scores
            sample_scores = {}
            if n >= 2:
                sample_scores["M[0,1]"] = round(float(matrix[0, 1]), 4)
                sample_scores["M[1,0]"] = round(float(matrix[1, 0]), 4)
                sample_scores["asymmetric_0_1"] = bool(matrix[0, 1] != matrix[1, 0])

            entry = {
                "batch_size": batch_size,
                "sentence_count": n,
                "total_pairs": actual_pairs,
                "matrix_shape": list(matrix.shape),
                "elapsed_seconds": round(elapsed, 4),
                "pairs_per_sec": round(rate, 2),
                "asymmetry_ok": asym_ok,
                "asymmetry_violations": asym_violations,
                "max_length": MAX_LENGTH,
                "device": DEVICE,
                "truncated_pairs": trunc_count,
                "truncation_rate": round(trunc_rate, 4),
                "sample_scores": sample_scores,
            }
            bench_results.append(entry)
            log(f"  Elapsed    : {elapsed:.4f}s")
            log(f"  Pairs/sec  : {rate:.1f}")
            log(f"  Matrix     : {matrix.shape}")
            log(f"  Asymmetry  : {'OK' if asym_ok else f'FAIL ({asym_violations} violations)'}")
            log(f"  M[0,1]={sample_scores.get('M[0,1]','?')}, M[1,0]={sample_scores.get('M[1,0]','?')}")

        except Exception as e:
            log(f"  BLOCKED: Inference failed with batch_size={batch_size}: {e}")
            log(traceback.format_exc())
            bench_results.append({
                "batch_size": batch_size,
                "status": "BLOCKED",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    # --- Runtime extrapolation for full dataset (not running it yet) ---
    log("\n--- Runtime estimate for all 7 files (extrapolation only) ---")
    file_pair_counts = {
        "sample-1": 49 * 48,
        "sample-2 to sample-7": "varies (avg ~49 sentences each)",
    }
    successful = [r for r in bench_results if "elapsed_seconds" in r]
    if successful:
        best = min(successful, key=lambda r: r["elapsed_seconds"])
        best_rate = best["pairs_per_sec"]
        # estimate based on 49 sentences (actual full sample-1)
        full_pairs = 49 * 48   # 2352
        est_per_file = full_pairs / best_rate if best_rate > 0 else None
        est_7_files = est_per_file * 7 if est_per_file else None
        log(f"  Best rate (batch={best['batch_size']}): {best_rate:.1f} pairs/sec")
        log(f"  Full sample-1 pairs (49 sents): {full_pairs}")
        log(f"  Estimated time per file: {est_per_file:.1f}s" if est_per_file else "  N/A")
        log(f"  Estimated time for 7 files: {est_7_files:.1f}s" if est_7_files else "  N/A")
        log("  NOTE: This is an extrapolation. Run only after user approval.")
        extrapolation = {
            "best_batch_size": best["batch_size"],
            "best_rate_pairs_per_sec": best_rate,
            "full_sample1_pairs": full_pairs,
            "estimated_seconds_per_file": round(est_per_file, 2) if est_per_file else None,
            "estimated_seconds_7_files": round(est_7_files, 2) if est_7_files else None,
            "note": "Extrapolation only. 7-file run not executed.",
        }
    else:
        extrapolation = {"status": "BLOCKED — no successful benchmark run"}
        log("  BLOCKED — no successful benchmark to extrapolate from.")

    # --- Compile full result ---
    result = {
        "status": "COMPLETED" if any("elapsed_seconds" in r for r in bench_results) else "BLOCKED",
        "benchmark_subset_size": n,
        "full_sample1_sentence_count": total_sentences,
        "expected_sentence_count": 49,
        "content_hash": content_hash,
        "nsp_head_verification": vinfo,
        "truncation": {
            "max_length": MAX_LENGTH,
            "pairs_exceeding_max_length": trunc_count,
            "total_pairs": trunc_total,
            "truncation_rate": round(trunc_rate, 4) if trunc_rate >= 0 else "measurement_failed",
        },
        "benchmarks": bench_results,
        "extrapolation": extrapolation,
        "package_versions": versions,
    }

    RESULTS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    LOG_PATH.write_text("\n".join(log_lines))
    log(f"\nResults saved to {RESULTS_PATH}")
    log(f"Log saved to {LOG_PATH}")
    log("=" * 70)


if __name__ == "__main__":
    run_benchmark()
