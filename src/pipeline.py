# -*- coding: utf-8 -*-
"""
Main Pipeline Entry Point: Model 1 (Punctuation) → Model 2 (Sentence Ordering).

Orchestrates the complete Hindi Text Reconstruction workflow:
  1. Accept raw unpunctuated, shuffled Hindi text file.
  2. Model 1: Restores punctuation (Purna Viram ।, Prashna Chinha ?, Alpa Viram ,).
  3. Sentence Segmentation: Splits text into grammatically complete sentences.
  4. Sentence ID Assignment: Deterministic unique IDs (e.g. sample-1_S001).
  5. Model 2: Orders sentences using either or both:
     - Method A: MiniLM Baseline (Cosine Similarity)
     - Method B: Multilingual BERT Next Sentence Prediction (Directional Margin)
  6. Reconstructs final ordered Hindi text using verbatim original sentence strings.

Usage:
  # Run full pipeline with comparison on all 7 files (using cached Model 1 punctuation)
  python src/pipeline.py --skip-punctuation

  # Run full pipeline from raw text (Model 1 inference + Model 2 ordering)
  python src/pipeline.py

  # Run on a single file
  python src/pipeline.py --file data/sample-1.txt --skip-punctuation

  # Choose specific Model 2 method (compare, minilm, mbert_nsp)
  python src/pipeline.py --method mbert_nsp --skip-punctuation
"""

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODEL2_OUTPUTS_DIR = OUTPUTS_DIR / "model2_ordering"
PIPELINE_OUTPUTS_DIR = OUTPUTS_DIR / "pipeline_final"

for d in (MODEL2_OUTPUTS_DIR, PIPELINE_OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL1_ID = "Horizon-Labs/punctuation-restoration-small"
MODEL2_MINILM_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL2_MBERT_ID = "google-bert/bert-base-multilingual-cased"


# ---------------------------------------------------------------------------
# Stage 1: Model 1 Punctuation Restoration
# ---------------------------------------------------------------------------

def load_model1(device: Optional[str] = None):
    """Download the official helper and instantiate the Punctuator."""
    from huggingface_hub import hf_hub_download
    print(f"Loading Model 1: {MODEL1_ID} ...")
    helper_path = hf_hub_download(repo_id=MODEL1_ID, filename="punctuate.py")
    spec = importlib.util.spec_from_file_location("punctuate", helper_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["punctuate"] = module
    spec.loader.exec_module(module)
    punctuator = module.Punctuator(MODEL1_ID, device=device)
    print("Model 1 loaded successfully.")
    return punctuator


def run_punctuation(
    source_path: Path,
    model1,
    output_dir: Path = OUTPUTS_DIR,
) -> Path:
    """Run Model 1 punctuation restoration on a raw Hindi source file."""
    raw_text = source_path.read_text(encoding="utf-8").strip()
    raw_text = re.sub(r"\bNULL\b", " ", raw_text)
    raw_text = re.sub(r"\s+", " ", raw_text).strip()

    t0 = time.time()
    punctuated = str(model1(raw_text))
    elapsed = time.time() - t0

    out_path = output_dir / f"{source_path.stem}_punctuated.txt"
    out_path.write_text(punctuated, encoding="utf-8")
    print(f"  Model 1: restored punctuation in {elapsed:.2f}s -> {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Stage 2: Main Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    source_files: List[Path],
    method: str = "compare",
    device: str = "cpu",
    skip_punctuation: bool = False,
    beam_width: int = 64,
    batch_size: int = 32,
    output_dir: Path = PIPELINE_OUTPUTS_DIR,
) -> List[Dict[str, Any]]:
    """Run the complete end-to-end pipeline on source files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Punctuation Stage
    model1 = None
    punctuated_paths = []

    for src in source_files:
        punc_path = OUTPUTS_DIR / f"{src.stem}_punctuated.txt"
        if skip_punctuation and punc_path.is_file():
            print(f"  [Model 1] Reusing cached punctuated output: {punc_path.name}")
            punctuated_paths.append(punc_path)
        else:
            if model1 is None:
                model1 = load_model1(device=device)
            punc_path = run_punctuation(src, model1, output_dir=OUTPUTS_DIR)
            punctuated_paths.append(punc_path)

    # 2. Ordering Stage (import model2 functions)
    model2_mod_path = Path(__file__).parent / "model2_ordering.py"
    spec = importlib.util.spec_from_file_location("model2_ordering", model2_mod_path)
    model2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model2)

    ordering_results = model2.run_model2(
        source_files=source_files,
        punctuated_files=punctuated_paths,
        method=method,
        device=device,
        beam_width=beam_width,
        batch_size=batch_size,
        output_dir=MODEL2_OUTPUTS_DIR,
    )

    # 3. Final Reconstructed Outputs & Integrity Verification
    pipeline_records = []
    for src, punc_path, ord_res in zip(source_files, punctuated_paths, ordering_results):
        stem = src.stem
        rec: Dict[str, Any] = {
            "source_file": src.name,
            "punctuated_file": punc_path.name,
            "sentence_count": ord_res.get("sentence_count", 0),
            "device": device,
            "gpu_memory_compliance": "NOT VERIFIED" if device == "cpu" else "CHECK_CUDA",
        }

        # Handle MiniLM output — only copy if method included MiniLM
        minilm_ordered_file = MODEL2_OUTPUTS_DIR / f"{stem}_ordered_minilm.txt"
        ran_minilm = method in ("minilm", "compare", "both") and minilm_ordered_file.is_file()
        if ran_minilm:
            rec["minilm"] = ord_res.get("minilm", {})
            final_minilm = output_dir / f"{stem}_final_output_minilm.txt"
            final_minilm.write_text(minilm_ordered_file.read_text(encoding="utf-8"), encoding="utf-8")
            rec["final_output_minilm"] = final_minilm.name

        # Handle mBERT NSP output — only copy if method included mBERT
        mbert_ordered_file = MODEL2_OUTPUTS_DIR / f"{stem}_ordered_mbert_nsp.txt"
        ran_mbert = method in ("mbert_nsp", "compare", "both") and mbert_ordered_file.is_file()
        if ran_mbert:
            rec["mbert_nsp"] = ord_res.get("mbert_nsp", {})
            final_mbert = output_dir / f"{stem}_final_output_mbert.txt"
            final_mbert.write_text(mbert_ordered_file.read_text(encoding="utf-8"), encoding="utf-8")
            rec["final_output_mbert"] = final_mbert.name

        # Primary final output is determined by the method ACTUALLY run in this invocation.
        # MiniLM-only runs must NOT pick up a stale mBERT file from a previous run.
        # compare/both: mBERT is preferred as primary.
        final_file = output_dir / f"{stem}_final_output.txt"
        if method == "minilm" and ran_minilm:
            primary_source = minilm_ordered_file
            primary_model = MODEL2_MINILM_ID
        elif method == "mbert_nsp" and ran_mbert:
            primary_source = mbert_ordered_file
            primary_model = MODEL2_MBERT_ID
        elif method in ("compare", "both") and ran_mbert:
            primary_source = mbert_ordered_file
            primary_model = MODEL2_MBERT_ID
        elif method in ("compare", "both") and ran_minilm:
            primary_source = minilm_ordered_file
            primary_model = MODEL2_MINILM_ID
        else:
            primary_source = None
            primary_model = None

        if primary_source is not None and primary_source.is_file():
            final_file.write_text(primary_source.read_text(encoding="utf-8"), encoding="utf-8")
            rec["final_output_file"] = final_file.name
            rec["primary_model"] = primary_model

        if "agreement" in ord_res:
            rec["agreement"] = ord_res["agreement"]

        pipeline_records.append(rec)
        print(f"  Pipeline [{stem}]: final output saved -> {final_file.name}")

    # Write pipeline report
    report_file = output_dir / "pipeline_report.jsonl"
    with report_file.open("w", encoding="utf-8") as f:
        for r in pipeline_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nPipeline report saved to {report_file}")

    return pipeline_records


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full Hindi NLP Pipeline: Model 1 (Punctuation) → Model 2 (Ordering)."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to single source .txt file (default: all 7 professor-provided files).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="compare",
        choices=["compare", "minilm", "mbert_nsp", "both"],
        help="Ordering method to execute (default: 'compare').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for execution (default: 'cpu').",
    )
    parser.add_argument(
        "--skip-punctuation",
        action="store_true",
        help="Reuse existing Model 1 punctuated outputs from outputs/.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=64,
        help="Beam width for beam search (default: 64).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for mBERT NSP pairwise inference (default: 32).",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Hindi Text Reconstruction Pipeline: Model 1 → Model 2")
    print(f"  Model 1: {MODEL1_ID}")
    print(f"  Model 2: MiniLM Baseline & mBERT Directional NSP")
    print(f"  Method : {args.method} | Device: {args.device} | Skip Punctuation: {args.skip_punctuation}")
    print("=" * 72)

    if args.file:
        source_files = [Path(args.file)]
    else:
        source_files = sorted(DATA_DIR.glob("sample-*.txt"))

    if not source_files:
        print("ERROR: No source files found in data/")
        sys.exit(1)

    t0 = time.time()
    results = run_pipeline(
        source_files=source_files,
        method=args.method,
        device=args.device,
        skip_punctuation=args.skip_punctuation,
        beam_width=args.beam_width,
        batch_size=args.batch_size,
    )
    total_elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print(f"Pipeline Completed in {total_elapsed:.1f}s")
    print(f"All outputs written to {PIPELINE_OUTPUTS_DIR} and {MODEL2_OUTPUTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
