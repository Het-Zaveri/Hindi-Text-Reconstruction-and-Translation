#!/usr/bin/env python
"""Translate ordered Hindi sentence lines with the official IndicTrans2 toolkit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

MODEL_ID = "ai4bharat/indictrans2-indic-en-dist-200M"
SRC_LANG = "hin_Deva"
TGT_LANG = "eng_Latn"


def process_peak_rss_mb() -> float | None:
    """Return peak Windows process working set (or None if unavailable)."""
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb):
            return round(c.PeakWorkingSetSize / (1024 * 1024), 2)
    except Exception:
        return None
    return None


def read_lines(path: Path, count: int | None) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as f:
        rows = [line.strip() for line in f if line.strip()]
    return rows if count is None else rows[:count]


def translate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit import IndicProcessor

    device = torch.device(args.device)
    sentences = read_lines(args.input, args.count)
    if not sentences:
        raise ValueError(f"No non-empty sentence lines in {args.input}")
    ip = IndicProcessor(inference=True)

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer_load_s = time.perf_counter() - started
    started = time.perf_counter()
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    model_load_s = time.perf_counter() - started
    params = sum(p.numel() for p in model.parameters())
    max_source = min(int(getattr(model.config, "max_position_embeddings", 512)), 512)
    results: list[dict[str, Any]] = []
    batch_times: list[float] = []

    for start in range(0, len(sentences), args.group_size):
        group = sentences[start:start + args.group_size]
        group_idx = start // args.group_size + 1
        entries: list[dict[str, Any]] = []
        try:
            preprocessed = ip.preprocess_batch(group, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
            # Check untruncated lengths first. Never silently discard source tokens.
            encoded = tokenizer(preprocessed, padding=False, truncation=False, add_special_tokens=True)
            counts = [len(ids) for ids in encoded["input_ids"]]
            too_long = [i for i, n in enumerate(counts) if n > max_source]
            if too_long:
                raise ValueError(f"source exceeds safe model input limit {max_source}: local indexes {too_long}")
            batch = tokenizer(preprocessed, padding="longest", truncation=False,
                              return_tensors="pt", return_attention_mask=True).to(device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**batch, use_cache=True, min_length=0, max_length=256,
                                           num_beams=args.num_beams, num_return_sequences=1)
            elapsed = time.perf_counter() - started
            batch_times.append(elapsed)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
            translations = ip.postprocess_batch(decoded, lang=TGT_LANG)
            if len(translations) != len(group):
                raise RuntimeError(f"batch alignment failure: expected {len(group)}, got {len(translations)}")
            # IndicTrans2's tokenizer parses source-language tags in its normal
            # encode path. English outputs must use its target SentencePiece path.
            out_counts = [len(tokenizer._tgt_tokenize(text)) for text in translations]
            for local, (src, tgt, n_in, n_out) in enumerate(zip(group, translations, counts, out_counts)):
                entries.append({"sentence_index": start + local + 1, "group_index": group_idx,
                                "hindi": src, "english": tgt, "source_token_count": n_in,
                                "output_token_count": n_out, "generation_seconds": elapsed / len(group),
                                "batch_generation_seconds": elapsed,
                                "empty_output": not bool(tgt.strip()), "truncated": False, "error": None})
        except Exception as exc:
            entries = [{"sentence_index": start + local + 1, "group_index": group_idx,
                        "hindi": src, "english": None, "source_token_count": None,
                        "output_token_count": None, "generation_seconds": None,
                        "batch_generation_seconds": None,
                        "empty_output": None, "truncated": None, "error": f"{type(exc).__name__}: {exc}"}
                       for local, src in enumerate(group)]
        results.extend(entries)

    return {"model_id": args.model_id, "device": str(device), "source_language": SRC_LANG,
            "target_language": TGT_LANG, "input_file": str(args.input), "sentence_count": len(sentences),
            "group_size": args.group_size, "group_count": (len(sentences) + args.group_size - 1) // args.group_size,
            "num_beams": args.num_beams, "parameter_count": params,
            "tokenizer_load_seconds": tokenizer_load_s, "model_load_seconds": model_load_s,
            "inference_seconds": sum(batch_times), "batch_inference_seconds": batch_times,
            "sentences": results, "peak_working_set_mb": process_peak_rss_mb()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 file with one Hindi sentence per line")
    parser.add_argument("--output-json", type=Path, help="JSON output path (default: <input>.indictrans2.json)")
    parser.add_argument("--output-text", type=Path, help="Optional UTF-8 English output, one line per source")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--count", type=int, help="Translate first N non-empty lines; default all")
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="This project runs inference on CPU")
    parser.add_argument("--num-beams", type=int, default=5)
    args = parser.parse_args()
    if args.group_size < 1 or (args.count is not None and args.count < 1):
        parser.error("--group-size and --count must be positive")
    if args.output_json is None:
        args.output_json = args.input.with_suffix(".indictrans2.json")
    report = translate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_text:
        args.output_text.parent.mkdir(parents=True, exist_ok=True)
        with args.output_text.open("w", encoding="utf-8", newline="\n") as out:
            for item in report["sentences"]:
                out.write((item["english"] if item["english"] is not None else "") + "\n")
    print(f"Wrote {args.output_json}; {report['sentence_count']} sentences, "
          f"{report['group_count']} groups, {report['inference_seconds']:.3f}s inference")
    return 0 if all(s["error"] is None and not s["empty_output"] for s in report["sentences"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
