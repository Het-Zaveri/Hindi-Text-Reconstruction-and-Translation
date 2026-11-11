# -*- coding: utf-8 -*-
"""
Comprehensive unit and regression tests for the Model 1 evaluation pipeline.
Covers:
  1. Exact sequence accuracy comparing both words AND punctuation in order.
  2. Cache validation rejecting duplicate IDs, missing IDs, mismatched input text,
     wrong model IDs, and stale revisions.
  3. Inference failure isolation (storing failures separately, never scoring error strings,
     reporting failed counts, and calculating metrics only on valid predictions).
  4. Hardware and CPU reporting (reporting NOT VERIFIED on CPU, actual GPU telemetry null).
  5. Text normalization, tokenization, number preservation, and Hindi Unicode integrity.
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Fix stdout encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.evaluate import (
    clean_input_text,
    extract_word_punc_pairs,
    align_word_punc_pairs,
    compute_evaluation_metrics,
    load_reference_dataset,
    verify_cached_predictions,
    get_device_info,
    measure_gpu_memory,
    MODEL_ID,
    MODEL_REVISION,
    PUNCT_MARKS,
    NONE_CLASS,
)


class TestPipelineEdgeCases(unittest.TestCase):
    """Test suite covering tokenization, normalization, alignment, and evaluation logic."""

    # -----------------------------------------------------------------------
    # 1. Text cleaning tests
    # -----------------------------------------------------------------------
    def test_clean_input_text_empty_string(self):
        self.assertEqual(clean_input_text(""), "")
        self.assertEqual(clean_input_text("   "), "")

    def test_clean_input_text_repeated_spaces(self):
        raw = "भारत   एक    विशाल   देश   है"
        expected = "भारत एक विशाल देश है"
        self.assertEqual(clean_input_text(raw), expected)

    def test_clean_input_text_null_placeholders(self):
        raw = "भटनी से NULL तथा दूसरी छपरा NULL से"
        expected = "भटनी से तथा दूसरी छपरा से"
        self.assertEqual(clean_input_text(raw), expected)

    # -----------------------------------------------------------------------
    # 2. Tokenization & punctuation extraction tests
    # -----------------------------------------------------------------------
    def test_extract_word_punc_pairs_attached_punctuation(self):
        text = "भारत, एक विशाल देश है।"
        pairs = extract_word_punc_pairs(text)
        self.assertEqual(pairs, [
            ("भारत", ","),
            ("एक", NONE_CLASS),
            ("विशाल", NONE_CLASS),
            ("देश", NONE_CLASS),
            ("है", "।"),
        ])

    def test_extract_word_punc_pairs_standalone_punctuation(self):
        text = "7 000 - 9 000"
        pairs = extract_word_punc_pairs(text)
        self.assertEqual(pairs, [
            ("7", NONE_CLASS),
            ("000", "-"),
            ("9", NONE_CLASS),
            ("000", NONE_CLASS),
        ])

    def test_extract_word_punc_pairs_period_normalization(self):
        text = "भारत एक देश है."
        pairs = extract_word_punc_pairs(text)
        self.assertEqual(pairs[-1], ("है", "।"))

    def test_extract_word_punc_pairs_preserve_decimals(self):
        text = "मूल्य 3.14 और 1,000 रुपये है।"
        pairs = extract_word_punc_pairs(text)
        self.assertEqual(pairs, [
            ("मूल्य", NONE_CLASS),
            ("3.14", NONE_CLASS),
            ("और", NONE_CLASS),
            ("1,000", NONE_CLASS),
            ("रुपये", NONE_CLASS),
            ("है", "।"),
        ])

    def test_extract_word_punc_pairs_hindi_unicode(self):
        text = "ऋषि मुनियों की संस्कृति एवं क्षत्रिय धर्म।"
        pairs = extract_word_punc_pairs(text)
        words = [w for w, _ in pairs]
        self.assertIn("ऋषि", words)
        self.assertIn("संस्कृति", words)
        self.assertIn("क्षत्रिय", words)
        self.assertEqual(pairs[-1], ("धर्म", "।"))

    # -----------------------------------------------------------------------
    # 3. Alignment tests
    # -----------------------------------------------------------------------
    def test_align_word_punc_pairs_exact(self):
        ref_pairs = [("भारत", NONE_CLASS), ("है", "।")]
        pred_pairs = [("भारत", NONE_CLASS), ("है", "।")]
        aligned = align_word_punc_pairs(ref_pairs, pred_pairs)
        self.assertEqual(aligned, [
            (NONE_CLASS, NONE_CLASS, True),
            ("।", "।", True),
        ])

    def test_align_word_punc_pairs_mismatched_punc(self):
        ref_pairs = [("भारत", ","), ("है", "।")]
        pred_pairs = [("भारत", NONE_CLASS), ("है", "?")]
        aligned = align_word_punc_pairs(ref_pairs, pred_pairs)
        self.assertEqual(aligned, [
            (",", NONE_CLASS, True),
            ("।", "?", True),
        ])

    def test_align_word_punc_pairs_missing_word(self):
        ref_pairs = [("भारत", NONE_CLASS), ("एक", NONE_CLASS), ("है", "।")]
        pred_pairs = [("भारत", NONE_CLASS), ("है", "।")]
        aligned = align_word_punc_pairs(ref_pairs, pred_pairs)
        matched = sum(1 for _, _, m in aligned if m)
        self.assertEqual(matched, 2)

    def test_align_word_punc_pairs_extra_word(self):
        ref_pairs = [("भारत", NONE_CLASS), ("है", "।")]
        pred_pairs = [("भारत", NONE_CLASS), ("महान", NONE_CLASS), ("है", "।")]
        aligned = align_word_punc_pairs(ref_pairs, pred_pairs)
        matched = sum(1 for _, _, m in aligned if m)
        self.assertEqual(matched, 2)
        unmatched = [a for a in aligned if not a[2]]
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(unmatched[0], (NONE_CLASS, NONE_CLASS, False))

    def test_align_word_punc_pairs_replaced_word(self):
        ref_pairs = [("भारत", ","), ("देश", "।")]
        pred_pairs = [("इंडिया", ","), ("देश", "।")]
        aligned = align_word_punc_pairs(ref_pairs, pred_pairs)
        self.assertFalse(aligned[0][2])
        self.assertTrue(aligned[1][2])
        self.assertEqual(aligned[1], ("।", "।", True))

    def test_align_word_punc_pairs_string_inputs(self):
        ref_str = "भारत, एक देश है।"
        pred_str = "भारत एक देश है।"
        aligned = align_word_punc_pairs(ref_str, pred_str)
        self.assertEqual(len(aligned), 4)
        self.assertEqual(aligned[0], (",", NONE_CLASS, True))
        self.assertEqual(aligned[-1], ("।", "।", True))

    # -----------------------------------------------------------------------
    # 4. Exact Sequence Accuracy (Words AND Punctuation)
    # -----------------------------------------------------------------------
    def test_exact_sequence_accuracy_matching_words_and_punct(self):
        ref = [{"sample_id": "T01", "reference_text": "भारत एक विशाल देश है।", "input_text": "भारत एक विशाल देश है"}]
        pred = [{"sample_id": "T01", "prediction_text": "भारत एक विशाल देश है।", "input_text": "भारत एक विशाल देश है"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics = compute_evaluation_metrics(ref, pred, reports_dir=Path(tmp_dir))
            self.assertEqual(metrics["exact_punctuation_accuracy"], 1.0)

    def test_exact_sequence_accuracy_fails_on_word_mismatch(self):
        # Punctuation matches exactly (terminal '।'), but words differ ('महान' vs 'विशाल')
        ref = [{"sample_id": "T01", "reference_text": "भारत एक विशाल देश है।", "input_text": "भारत एक विशाल देश है"}]
        pred = [{"sample_id": "T01", "prediction_text": "भारत एक महान देश है।", "input_text": "भारत एक विशाल देश है"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics = compute_evaluation_metrics(ref, pred, reports_dir=Path(tmp_dir))
            # Exact sequence accuracy MUST compare both words AND punctuation, in order.
            self.assertEqual(metrics["exact_punctuation_accuracy"], 0.0)

    def test_exact_sequence_accuracy_fails_on_dropped_word(self):
        ref = [{"sample_id": "T01", "reference_text": "भारत एक विशाल देश है।", "input_text": "भारत एक विशाल देश है"}]
        pred = [{"sample_id": "T01", "prediction_text": "भारत एक देश है।", "input_text": "भारत एक विशाल देश है"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics = compute_evaluation_metrics(ref, pred, reports_dir=Path(tmp_dir))
            self.assertEqual(metrics["exact_punctuation_accuracy"], 0.0)

    # -----------------------------------------------------------------------
    # 5. Inference Failure Isolation Tests
    # -----------------------------------------------------------------------
    def test_inference_failures_stored_separately_not_scored(self):
        reference = [
            {"sample_id": "S01", "reference_text": "यह सही वाक्य है।", "input_text": "यह सही वाक्य है"},
            {"sample_id": "S02", "reference_text": "यह दूसरा वाक्य है।", "input_text": "यह दूसरा वाक्य है"},
        ]
        # One valid prediction, one error message
        predictions = [
            {"sample_id": "S01", "prediction_text": "यह सही वाक्य है।", "input_text": "यह सही वाक्य है"},
            {"sample_id": "S02", "error": "RuntimeError: CUDA out of memory", "prediction_text": None, "input_text": "यह दूसरा वाक्य है"},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics = compute_evaluation_metrics(reference, predictions, reports_dir=Path(tmp_dir))
            # Must report failed count separately
            self.assertEqual(metrics["total_samples"], 2)
            self.assertEqual(metrics["valid_samples"], 1)
            self.assertEqual(metrics["failed_samples"], 1)
            # Evaluated metrics calculated ONLY on valid predictions (100% exact match on S01)
            self.assertEqual(metrics["exact_punctuation_accuracy"], 1.0)
            self.assertEqual(metrics["word_preservation_accuracy"], 1.0)

    def test_error_string_never_scored_as_words(self):
        reference = [{"sample_id": "S01", "reference_text": "यह एक वाक्य है।", "input_text": "यह एक वाक्य है"}]
        predictions = [{"sample_id": "S01", "prediction_text": "<INFERENCE_ERROR: out of memory>", "input_text": "यह एक वाक्य है"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics = compute_evaluation_metrics(reference, predictions, reports_dir=Path(tmp_dir))
            self.assertEqual(metrics["valid_samples"], 0)
            self.assertEqual(metrics["failed_samples"], 1)

    # -----------------------------------------------------------------------
    # 6. Cache Validation Tests (Rejects Duplicate IDs, Missing IDs, etc.)
    # -----------------------------------------------------------------------
    def test_verify_cached_predictions_valid(self):
        records = load_reference_dataset()
        pred_path = BASE_DIR / "evaluation" / "predictions" / "model1_predictions.jsonl"
        verified = verify_cached_predictions(pred_path, records)
        self.assertEqual(len(verified), 55)

    def test_verify_cached_predictions_duplicate_id_rejected(self):
        records = load_reference_dataset()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as tf:
            p1 = {
                "sample_id": records[0]["sample_id"],
                "prediction_text": "वाक्य।",
                "input_text": records[0]["input_text"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
            tf.write(json.dumps(p1) + "\n")
            tf.write(json.dumps(p1) + "\n")  # Duplicate ID
            temp_path = Path(tf.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                verify_cached_predictions(temp_path, records)
            self.assertIn("Duplicate sample_id", str(ctx.exception))
        finally:
            if temp_path.is_file():
                temp_path.unlink()

    def test_verify_cached_predictions_missing_id_rejected(self):
        records = load_reference_dataset()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as tf:
            # Only write 1 record out of 55
            p = {
                "sample_id": records[0]["sample_id"],
                "prediction_text": "वाक्य।",
                "input_text": records[0]["input_text"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
            tf.write(json.dumps(p) + "\n")
            temp_path = Path(tf.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                verify_cached_predictions(temp_path, records)
            self.assertIn("missing ids", str(ctx.exception).lower())
        finally:
            if temp_path.is_file():
                temp_path.unlink()

    def test_verify_cached_predictions_mismatched_input_text_rejected(self):
        records = load_reference_dataset()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as tf:
            # Write a prediction with modified input text (stale)
            for r in records:
                text = "भिन्न इनपुट टेक्स्ट" if r["sample_id"] == records[0]["sample_id"] else r["input_text"]
                p = {
                    "sample_id": r["sample_id"],
                    "prediction_text": "वाक्य।",
                    "input_text": text,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                }
                tf.write(json.dumps(p) + "\n")
            temp_path = Path(tf.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                verify_cached_predictions(temp_path, records)
            self.assertIn("Input text mismatch", str(ctx.exception))
        finally:
            if temp_path.is_file():
                temp_path.unlink()

    def test_verify_cached_predictions_wrong_model_id_rejected(self):
        records = load_reference_dataset()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as tf:
            for r in records:
                p = {
                    "sample_id": r["sample_id"],
                    "prediction_text": "वाक्य।",
                    "input_text": r["input_text"],
                    "model_id": "Wrong/Model-Name",
                    "model_revision": MODEL_REVISION,
                }
                tf.write(json.dumps(p) + "\n")
            temp_path = Path(tf.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                verify_cached_predictions(temp_path, records)
            self.assertIn("Model ID mismatch", str(ctx.exception))
        finally:
            if temp_path.is_file():
                temp_path.unlink()

    def test_verify_cached_predictions_stale_revision_rejected(self):
        records = load_reference_dataset()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as tf:
            for r in records:
                p = {
                    "sample_id": r["sample_id"],
                    "prediction_text": "वाक्य।",
                    "input_text": r["input_text"],
                    "model_id": MODEL_ID,
                    "model_revision": "stale_commit_hash_12345",
                }
                tf.write(json.dumps(p) + "\n")
            temp_path = Path(tf.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                verify_cached_predictions(temp_path, records)
            self.assertIn("Stale model revision", str(ctx.exception))
        finally:
            if temp_path.is_file():
                temp_path.unlink()

    # -----------------------------------------------------------------------
    # 7. Hardware & GPU Reporting Tests
    # -----------------------------------------------------------------------
    def test_device_info_cpu_reports_not_verified(self):
        info = get_device_info("cpu")
        self.assertEqual(info["device"], "cpu")
        self.assertEqual(info["gpu_memory_compliance"], "NOT VERIFIED")
        self.assertIsNone(info["actual_gpu_allocated_MB"])
        self.assertIsNone(info["actual_gpu_reserved_MB"])

    def test_measure_gpu_memory_cpu_reports_not_verified(self):
        mem = measure_gpu_memory("cpu")
        self.assertEqual(mem["gpu_memory_compliance"], "NOT VERIFIED")
        self.assertIsNone(mem["compliant_1_5gb"])
        self.assertNotEqual(mem["gpu_memory_compliance"], "PASS")
        self.assertNotEqual(mem["gpu_memory_compliance"], "FAIL")
        self.assertIsNone(mem["actual_gpu_allocated_MB"])
        self.assertIsNone(mem["actual_gpu_reserved_MB"])

    def test_reference_dataset_integrity(self):
        records = load_reference_dataset()
        self.assertEqual(len(records), 55)
        seen_ids = set()
        for r in records:
            self.assertNotIn(r["sample_id"], seen_ids)
            seen_ids.add(r["sample_id"])
            self.assertTrue(bool(r["input_text"].strip()))
            self.assertTrue(bool(r["reference_text"].strip()))
            src_path = BASE_DIR / "data" / r["source_file"]
            self.assertTrue(src_path.is_file(), f"Missing source file: {r['source_file']}")

    def test_reference_dataset_csv_jsonl_consistency(self):
        jsonl_path = BASE_DIR / "evaluation" / "dataset" / "reference_dataset.jsonl"
        csv_path = BASE_DIR / "evaluation" / "dataset" / "reference_dataset.csv"
        self.assertTrue(jsonl_path.is_file())
        self.assertTrue(csv_path.is_file())

        with jsonl_path.open("r", encoding="utf-8") as jf:
            jsonl_records = [json.loads(line) for line in jf if line.strip()]

        with csv_path.open("r", encoding="utf-8") as cf:
            reader = csv.DictReader(cf)
            csv_records = list(reader)

        self.assertEqual(len(jsonl_records), len(csv_records))
        for j_rec, c_rec in zip(jsonl_records, csv_records):
            self.assertEqual(j_rec["sample_id"], c_rec["sample_id"])
            self.assertEqual(j_rec["input_text"].strip(), c_rec["input_text"].strip())
            self.assertEqual(j_rec["reference_text"].strip(), c_rec["reference_text"].strip())


if __name__ == "__main__":
    unittest.main()
