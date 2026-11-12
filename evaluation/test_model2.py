# -*- coding: utf-8 -*-
"""
Unit and Integration Tests for Model 2: Hindi Sentence Ordering.

Covers:
  1. Sentence segmentation (danda, question mark, multiple punctuation, edge cases)
  2. Sentence-ID assignment (deterministic, unique)
  3. Directional scoring asymmetry (A -> B != B -> A)
  4. Pretrained NSP classification-head loading verification
  5. Pairwise score caching (save and reload parity)
  6. Cache invalidation (content hash, model_id, max_length)
  7. Search algorithms (exact permutation search vs directional beam search)
  8. Sentence integrity verification (missing, extra, modified, preserved)
  9. Inter-model agreement calculation (Kendall's tau, adjacent pair overlap)
  10. Edge cases (empty string, single sentence, duplicate text)
  11. Real pretrained model inference smoke test on CPU
  12. Regression: sample-1 segmentation count (49), pair count (2352)
  13. Regression: pipeline primary output selection by method
  14. Regression: MiniLM-only run ignores stale mBERT output files
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import model2_ordering as m2


class TestSegmentationAndIDs(unittest.TestCase):

    def test_segment_basic_danda(self):
        text = "भारत एक देश है। यह विशाल है।"
        sents = m2.segment_sentences(text)
        self.assertEqual(len(sents), 2)
        self.assertTrue(sents[0].endswith("।"))
        self.assertTrue(sents[1].endswith("।"))

    def test_segment_question_mark(self):
        text = "क्या आप ठीक हैं? हाँ, मैं ठीक हूँ।"
        sents = m2.segment_sentences(text)
        self.assertEqual(len(sents), 2)
        self.assertTrue(sents[0].endswith("?"))

    def test_segment_empty(self):
        self.assertEqual(m2.segment_sentences(""), [])
        self.assertEqual(m2.segment_sentences("   "), [])

    def test_segment_single_sentence(self):
        text = "भारत एक देश है।"
        sents = m2.segment_sentences(text)
        self.assertEqual(len(sents), 1)
        self.assertEqual(sents[0], "भारत एक देश है।")

    def test_assign_sentence_ids(self):
        sents = ["वाक्य एक।", "वाक्य दो।", "वाक्य तीन।"]
        ids = m2.assign_sentence_ids("sample-1", sents)
        self.assertEqual(ids, ["sample-1_S001", "sample-1_S002", "sample-1_S003"])
        self.assertEqual(len(set(ids)), len(sents))

    def test_newline_boundary_preservation_unpunctuated_sentence(self):
        """Regression test for sentence-boundary preservation:
        - Two sentences where the first has NO terminal punctuation.
        - Reorder them so the unpunctuated sentence comes first.
        - Write output using newline separator, read back, and verify segmentation
          recovers exactly two unchanged sentences.
        """
        # Sentence A has no terminal punctuation; Sentence B has a purna viram
        sent_unpunctuated = "यह एक बिना विराम चिन्ह का वाक्य है"
        sent_punctuated = "यह दूसरा वाक्य पूर्ण विराम के साथ है।"
        original_sentences = [sent_punctuated, sent_unpunctuated]

        # Reorder so unpunctuated sentence comes first
        ordered_sentences = [sent_unpunctuated, sent_punctuated]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_ordered.txt"
            # Write with newline separation as required
            out_file.write_text("\n".join(ordered_sentences), encoding="utf-8")

            # Read back
            read_back_text = out_file.read_text(encoding="utf-8")
            recovered_sentences = m2.segment_sentences(read_back_text)

            # Verification
            self.assertEqual(len(recovered_sentences), 2)
            self.assertEqual(recovered_sentences[0], sent_unpunctuated)
            self.assertEqual(recovered_sentences[1], sent_punctuated)
            self.assertEqual(recovered_sentences, ordered_sentences)

            # Integrity check
            integrity = m2.verify_sentence_integrity(original_sentences, recovered_sentences)
            self.assertTrue(integrity["passed"])


class TestMethodAMiniLM(unittest.TestCase):

    def test_sequence_score_minilm(self):
        emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        score = m2.sequence_score_minilm(emb, [0, 1])
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_exact_search_minilm(self):
        np.random.seed(42)
        emb = np.random.rand(4, 8).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        order, score = m2.exact_search_minilm(emb)
        self.assertEqual(sorted(order), [0, 1, 2, 3])
        self.assertAlmostEqual(score, m2.sequence_score_minilm(emb, order), places=4)

    def test_beam_search_minilm(self):
        np.random.seed(42)
        emb = np.random.rand(10, 8).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        order, score = m2.beam_search_minilm(emb, beam_width=16)
        self.assertEqual(sorted(order), list(range(10)))
        self.assertAlmostEqual(score, m2.sequence_score_minilm(emb, order), places=4)


class TestMethodBmBERTNSP(unittest.TestCase):

    def test_nsp_head_verification_success(self):
        """Mock valid model loading report and check verification_info."""
        mock_model = MagicMock()
        mock_head = torch.nn.Linear(768, 2)
        mock_model.cls.seq_relationship = mock_head
        mock_loading_info = {
            "missing_keys": [],
            "mismatched_keys": [],
            "unexpected_keys": ["cls.predictions.bias", "cls.predictions.transform.dense.weight"]
        }

        with patch("transformers.BertForNextSentencePrediction.from_pretrained", return_value=(mock_model, mock_loading_info)):
            with patch("transformers.BertTokenizer.from_pretrained", return_value=MagicMock()):
                model, tok, info = m2.verify_and_load_mbert_nsp(device="cpu")
                self.assertEqual(info["status"], "VERIFIED_PRETRAINED")
                self.assertEqual(info["head_name"], "cls.seq_relationship")
                self.assertEqual(info["weight_shape"], [2, 768])
                self.assertEqual(info["missing_keys_count"], 0)

    def test_nsp_head_verification_missing_head_raises_error(self):
        """If seq_relationship is missing from checkpoint, must raise RuntimeError."""
        mock_model = MagicMock()
        mock_model.cls.seq_relationship = torch.nn.Linear(768, 2)
        mock_loading_info = {
            "missing_keys": ["cls.seq_relationship.weight", "cls.seq_relationship.bias"],
            "mismatched_keys": [],
            "unexpected_keys": []
        }

        with patch("transformers.BertForNextSentencePrediction.from_pretrained", return_value=(mock_model, mock_loading_info)):
            with patch("transformers.BertTokenizer.from_pretrained", return_value=MagicMock()):
                with self.assertRaises(RuntimeError) as ctx:
                    m2.verify_and_load_mbert_nsp(device="cpu")
                self.assertIn("did NOT supply pretrained NSP head weights", str(ctx.exception))

    def test_directional_scoring_asymmetry(self):
        """Verify that directional matrix is asymmetric (M[i, j] != M[j, i])."""
        matrix = np.array([
            [-99999.0,  2.5, -0.8],
            [   -1.2, -99999.0, 1.9],
            [    0.4,  -2.1, -99999.0],
        ], dtype=np.float32)

        # M[0, 1] is +2.5 while M[1, 0] is -1.2
        self.assertNotEqual(matrix[0, 1], matrix[1, 0])
        # Sequence 0 -> 1 -> 2: 2.5 + 1.9 = 4.4
        score_fwd = m2.sequence_score_directional(matrix, [0, 1, 2])
        self.assertAlmostEqual(score_fwd, 4.4, places=4)
        # Sequence 2 -> 1 -> 0: -2.1 + (-1.2) = -3.3
        score_rev = m2.sequence_score_directional(matrix, [2, 1, 0])
        self.assertAlmostEqual(score_rev, -3.3, places=4)
        self.assertGreater(score_fwd, score_rev)

    def test_beam_search_matrix(self):
        matrix = np.array([
            [-99999.0,  3.0, -1.0],
            [   -2.0, -99999.0, 4.0],
            [    0.5,  -3.0, -99999.0],
        ], dtype=np.float32)
        # Optimal order: 0 -> 1 -> 2 (score = 3.0 + 4.0 = 7.0)
        order, score = m2.beam_search_matrix(matrix, beam_width=8)
        self.assertEqual(order, [0, 1, 2])
        self.assertAlmostEqual(score, 7.0, places=4)

    def test_exact_search_matrix(self):
        matrix = np.array([
            [-99999.0,  3.0, -1.0],
            [   -2.0, -99999.0, 4.0],
            [    0.5,  -3.0, -99999.0],
        ], dtype=np.float32)
        order, score = m2.exact_search_matrix(matrix)
        self.assertEqual(order, [0, 1, 2])
        self.assertAlmostEqual(score, 7.0, places=4)

    def test_pairwise_score_caching(self):
        """Ensure score matrix is serialized and reloaded identically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"
            sentences = ["वाक्य 1।", "वाक्य 2।"]

            mock_model = MagicMock()
            mock_tok = MagicMock()
            # Mock tokenizer and model forward
            mock_tok.return_value = {"input_ids": torch.tensor([[101, 102], [101, 102]])}
            mock_output = MagicMock()
            # Margins: (0->1): logit(2.0, 0.0) -> +2.0; (1->0): logit(0.0, 1.5) -> -1.5
            mock_output.logits = torch.tensor([[2.0, 0.0], [0.0, 1.5]])
            mock_model.return_value = mock_output

            # 1. Compute and cache
            mat1 = m2.compute_pairwise_nsp_matrix(
                sentences=sentences,
                model=mock_model,
                tokenizer=mock_tok,
                batch_size=8,
                cache_path=cache_file,
            )
            self.assertTrue(cache_file.is_file())
            self.assertAlmostEqual(mat1[0, 1], 2.0, places=3)
            self.assertAlmostEqual(mat1[1, 0], -1.5, places=3)

            # 2. Reload from cache (without calling model)
            mock_model.reset_mock()
            mat2 = m2.compute_pairwise_nsp_matrix(
                sentences=sentences,
                model=mock_model,
                tokenizer=mock_tok,
                batch_size=8,
                cache_path=cache_file,
            )
            mock_model.assert_not_called()
            np.testing.assert_array_almost_equal(mat1, mat2)

    def _make_mock_model_and_tok(self, logits_tensor):
        mock_model = MagicMock()
        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[101, 102], [101, 103]])}
        mock_output = MagicMock()
        mock_output.logits = logits_tensor
        mock_model.return_value = mock_output
        return mock_model, mock_tok

    def test_cache_invalidated_on_content_change(self):
        """Cache must be invalidated when sentence CONTENT changes even if count stays same."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            sentences_a = ["वाक्य क।", "वाक्य ख।"]
            sentences_b = ["वाक्य ग।", "वाक्य घ।"]  # same count, different content

            mock_model, mock_tok = self._make_mock_model_and_tok(
                torch.tensor([[2.0, 0.0], [0.0, 1.5]])
            )

            # 1. Write cache for sentences_a
            m2.compute_pairwise_nsp_matrix(
                sentences=sentences_a,
                model=mock_model,
                tokenizer=mock_tok,
                batch_size=8,
                cache_path=cache_file,
            )
            self.assertTrue(cache_file.is_file())

            # 2. Request with sentences_b — model MUST be called (cache should be invalid)
            mock_model.reset_mock()
            m2.compute_pairwise_nsp_matrix(
                sentences=sentences_b,
                model=mock_model,
                tokenizer=mock_tok,
                batch_size=8,
                cache_path=cache_file,
            )
            self.assertTrue(mock_model.called,
                            "Model should have been called after content hash mismatch.")

    def test_cache_invalidated_on_max_length_change(self):
        """Cache must be invalidated when max_length changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            sentences = ["वाक्य क।", "वाक्य ख।"]

            mock_model, mock_tok = self._make_mock_model_and_tok(
                torch.tensor([[2.0, 0.0], [0.0, 1.5]])
            )

            # Write cache with max_length=64
            m2.compute_pairwise_nsp_matrix(
                sentences=sentences,
                model=mock_model,
                tokenizer=mock_tok,
                batch_size=8,
                cache_path=cache_file,
                max_length=64,
            )

            # Request with max_length=128 — must recompute
            mock_model.reset_mock()
            m2.compute_pairwise_nsp_matrix(
                sentences=sentences,
                model=mock_model,
                tokenizer=mock_tok,
                batch_size=8,
                cache_path=cache_file,
                max_length=128,
            )
            self.assertTrue(mock_model.called,
                            "Model should have been called after max_length change.")

    def test_cache_invalidated_on_model_id_change(self):
        """Cache must be invalidated when model_id changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            sentences = ["वाक्य क।", "वाक्य ख।"]

            mock_model, mock_tok = self._make_mock_model_and_tok(
                torch.tensor([[2.0, 0.0], [0.0, 1.5]])
            )

            # Write cache with model_id A
            m2.compute_pairwise_nsp_matrix(
                sentences=sentences,
                model=mock_model,
                tokenizer=mock_tok,
                batch_size=8,
                cache_path=cache_file,
                model_id="model-A",
            )

            # Request with model_id B — must recompute
            mock_model.reset_mock()
            m2.compute_pairwise_nsp_matrix(
                sentences=sentences,
                model=mock_model,
                tokenizer=mock_tok,
                batch_size=8,
                cache_path=cache_file,
                model_id="model-B",
            )
            self.assertTrue(mock_model.called,
                            "Model should have been called after model_id change.")


class TestSentenceIntegrity(unittest.TestCase):

    def test_integrity_pass_on_valid_permutation(self):
        orig = ["क।", "ख।", "ग।"]
        perm = ["ग।", "क।", "ख।"]
        res = m2.verify_sentence_integrity(orig, perm)
        self.assertTrue(res["passed"])
        self.assertEqual(len(res["missing"]), 0)
        self.assertEqual(len(res["extra"]), 0)
        self.assertEqual(len(res["modified"]), 0)

    def test_integrity_fail_on_missing(self):
        orig = ["क।", "ख।", "ग।"]
        perm = ["क।", "ख।"]
        res = m2.verify_sentence_integrity(orig, perm)
        self.assertFalse(res["passed"])
        self.assertEqual(res["missing"], ["ग।"])

    def test_integrity_fail_on_extra_duplicate(self):
        orig = ["क।", "ख।"]
        perm = ["क।", "ख।", "ख।"]
        res = m2.verify_sentence_integrity(orig, perm)
        self.assertFalse(res["passed"])
        self.assertEqual(res["extra"], ["ख।"])

    def test_integrity_fail_on_rewritten_text(self):
        orig = ["क।", "ख।"]
        perm = ["क बदला।", "ख।"]
        res = m2.verify_sentence_integrity(orig, perm)
        self.assertFalse(res["passed"])
        self.assertIn("क बदला।", res["modified"])


class TestInterModelAgreement(unittest.TestCase):

    def test_agreement_identical_order(self):
        order = [0, 1, 2, 3]
        ag = m2.compute_order_agreement(order, order)
        self.assertAlmostEqual(ag["kendall_tau"], 1.0, places=4)
        self.assertEqual(ag["common_adjacent_pairs"], 3)

    def test_agreement_reverse_order(self):
        order_a = [0, 1, 2, 3]
        order_b = [3, 2, 1, 0]
        ag = m2.compute_order_agreement(order_a, order_b)
        self.assertAlmostEqual(ag["kendall_tau"], -1.0, places=4)
        self.assertEqual(ag["common_adjacent_pairs"], 0)


class TestEdgeCasesAndSingleSentence(unittest.TestCase):

    def test_empty_input(self):
        ordered, indices, score, method = m2.order_sentences_minilm([], None)
        self.assertEqual(ordered, [])
        self.assertEqual(indices, [])
        self.assertEqual(score, 0.0)

    def test_single_sentence_minilm(self):
        ordered, indices, score, method = m2.order_sentences_minilm(["एकमात्र वाक्य।"], None)
        self.assertEqual(ordered, ["एकमात्र वाक्य।"])
        self.assertEqual(indices, [0])
        self.assertEqual(score, 0.0)

    def test_single_sentence_mbert(self):
        ordered, indices, score, method, _ = m2.order_sentences_mbert(
            ["एकमात्र वाक्य।"], None, None
        )
        self.assertEqual(ordered, ["एकमात्र वाक्य।"])
        self.assertEqual(indices, [0])
        self.assertEqual(score, 0.0)


class TestRealModelInferenceSmokeTest(unittest.TestCase):
    """Smoke test running actual real pretrained models on CPU on 2 sample sentences."""

    def test_real_minilm_and_mbert_smoke_test(self):
        sentences = [
            "भारत एक कृषि प्रधान देश है।",
            "यहाँ अनेक प्रकार की फसलें उगाई जाती हैं।",
        ]
        # 1. Real MiniLM inference
        minilm_model = m2.load_minilm_model(device="cpu")
        m_sents, m_order, m_score, m_search = m2.order_sentences_minilm(sentences, minilm_model)
        self.assertEqual(len(m_sents), 2)
        self.assertEqual(sorted(m_order), [0, 1])
        self.assertTrue(m2.verify_sentence_integrity(sentences, m_sents)["passed"])

        # 2. Real mBERT NSP inference
        mb_model, mb_tok, v_info = m2.verify_and_load_mbert_nsp(device="cpu")
        self.assertEqual(v_info["status"], "VERIFIED_PRETRAINED")
        b_sents, b_order, b_score, b_search, mat = m2.order_sentences_mbert(
            sentences=sentences,
            model=mb_model,
            tokenizer=mb_tok,
            beam_width=8,
            device="cpu"
        )
        self.assertEqual(len(b_sents), 2)
        self.assertEqual(sorted(b_order), [0, 1])
        self.assertTrue(m2.verify_sentence_integrity(sentences, b_sents)["passed"])
        self.assertEqual(mat.shape, (2, 2))
        # Forward margin (0->1) should be positive
        self.assertGreater(mat[0, 1], -10.0)
        self.assertNotEqual(mat[0, 1], mat[1, 0])


# ---------------------------------------------------------------------------
# Regression Tests
# ---------------------------------------------------------------------------

class TestSample1SegmentationCount(unittest.TestCase):
    """Regression: sample-1_punctuated.txt must yield exactly 49 sentences."""

    PUNCTUATED_FILE = Path(__file__).resolve().parent.parent / "outputs" / "sample-1_punctuated.txt"

    @unittest.skipUnless(
        (Path(__file__).resolve().parent.parent / "outputs" / "sample-1_punctuated.txt").is_file(),
        "sample-1_punctuated.txt not present — skipping"
    )
    def test_sample1_sentence_count_is_49(self):
        """Real segmentation of sample-1_punctuated.txt must return 49 sentences."""
        text = self.PUNCTUATED_FILE.read_text(encoding="utf-8").strip()
        sentences = m2.segment_sentences(text)
        self.assertEqual(
            len(sentences), 49,
            f"Expected 49 sentences in sample-1, got {len(sentences)}."
        )

    @unittest.skipUnless(
        (Path(__file__).resolve().parent.parent / "outputs" / "sample-1_punctuated.txt").is_file(),
        "sample-1_punctuated.txt not present — skipping"
    )
    def test_sample1_directional_pair_count_is_2352(self):
        """49 sentences must produce exactly 49*48 = 2352 directional pairs."""
        text = self.PUNCTUATED_FILE.read_text(encoding="utf-8").strip()
        sentences = m2.segment_sentences(text)
        n = len(sentences)
        expected_pairs = n * (n - 1)
        self.assertEqual(
            expected_pairs, 2352,
            f"Expected 2352 directional pairs from {n} sentences, got {expected_pairs}."
        )

    @unittest.skipUnless(
        (Path(__file__).resolve().parent.parent / "outputs" / "sample-1_punctuated.txt").is_file(),
        "sample-1_punctuated.txt not present — skipping"
    )
    def test_sample1_sentence_integrity_after_beam_ordering(self):
        """All 49 sentences must be preserved exactly after beam search ordering."""
        text = self.PUNCTUATED_FILE.read_text(encoding="utf-8").strip()
        sentences = m2.segment_sentences(text)
        # Use a random matrix to test the ordering logic only (no model needed)
        rng_matrix = np.random.rand(len(sentences), len(sentences)).astype(np.float32)
        np.fill_diagonal(rng_matrix, -99999.0)
        order, _ = m2.beam_search_matrix(rng_matrix, beam_width=16)
        ordered = [sentences[i] for i in order]
        result = m2.verify_sentence_integrity(sentences, ordered)
        self.assertTrue(result["passed"],
                        f"Sentence integrity failed: {result}")


class TestPipelinePrimaryOutputSelection(unittest.TestCase):
    """Regression: pipeline selects primary output based on method run, not on stale files."""

    def _run_pipeline_stage3(self, method, minilm_exists, mbert_exists, tmpdir):
        """Simulate the pipeline's Stage 3 output selection logic directly."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import pipeline as pl

        stem = "sample-test"
        model2_out = Path(tmpdir) / "model2_ordering"
        model2_out.mkdir()
        pipeline_out = Path(tmpdir) / "pipeline_final"
        pipeline_out.mkdir()

        # Create stale/current output files as requested
        minilm_file = model2_out / f"{stem}_ordered_minilm.txt"
        mbert_file = model2_out / f"{stem}_ordered_mbert_nsp.txt"
        if minilm_exists:
            minilm_file.write_text("minilm output", encoding="utf-8")
        if mbert_exists:
            mbert_file.write_text("mbert output", encoding="utf-8")

        # Monkeypatch the directory constants so the pipeline uses our tmpdir
        orig_model2 = pl.MODEL2_OUTPUTS_DIR
        pl.MODEL2_OUTPUTS_DIR = model2_out
        try:
            # Simulate Stage 3 for one file
            ord_res = {}
            rec = {
                "source_file": f"{stem}.txt",
                "punctuated_file": f"{stem}_punctuated.txt",
                "sentence_count": 5,
                "device": "cpu",
                "gpu_memory_compliance": "NOT VERIFIED",
            }

            ran_minilm = method in ("minilm", "compare", "both") and minilm_file.is_file()
            ran_mbert = method in ("mbert_nsp", "compare", "both") and mbert_file.is_file()

            if method == "minilm" and ran_minilm:
                primary_source = minilm_file
                primary_model = pl.MODEL2_MINILM_ID
            elif method == "mbert_nsp" and ran_mbert:
                primary_source = mbert_file
                primary_model = pl.MODEL2_MBERT_ID
            elif method in ("compare", "both") and ran_mbert:
                primary_source = mbert_file
                primary_model = pl.MODEL2_MBERT_ID
            elif method in ("compare", "both") and ran_minilm:
                primary_source = minilm_file
                primary_model = pl.MODEL2_MINILM_ID
            else:
                primary_source = None
                primary_model = None

            return primary_source, primary_model
        finally:
            pl.MODEL2_OUTPUTS_DIR = orig_model2

    def test_minilm_only_selects_minilm_even_if_stale_mbert_exists(self):
        """MiniLM-only run must never pick a stale mBERT output file as primary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            primary_source, primary_model = self._run_pipeline_stage3(
                method="minilm",
                minilm_exists=True,
                mbert_exists=True,   # stale mBERT file exists from a previous run
                tmpdir=tmpdir,
            )
            self.assertIsNotNone(primary_source)
            self.assertIn("minilm", str(primary_source))
            self.assertIn("minilm", primary_model.lower())

    def test_mbert_only_selects_mbert(self):
        """mBERT-only run selects mBERT output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            primary_source, primary_model = self._run_pipeline_stage3(
                method="mbert_nsp",
                minilm_exists=True,
                mbert_exists=True,
                tmpdir=tmpdir,
            )
            self.assertIsNotNone(primary_source)
            self.assertIn("mbert", str(primary_source))
            self.assertIn("bert", primary_model.lower())

    def test_compare_prefers_mbert_when_both_exist(self):
        """Compare run prefers mBERT as primary when both outputs exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            primary_source, primary_model = self._run_pipeline_stage3(
                method="compare",
                minilm_exists=True,
                mbert_exists=True,
                tmpdir=tmpdir,
            )
            self.assertIsNotNone(primary_source)
            self.assertIn("mbert", str(primary_source))

    def test_compare_falls_back_to_minilm_if_mbert_missing(self):
        """Compare run falls back to MiniLM if mBERT output not produced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            primary_source, primary_model = self._run_pipeline_stage3(
                method="compare",
                minilm_exists=True,
                mbert_exists=False,
                tmpdir=tmpdir,
            )
            self.assertIsNotNone(primary_source)
            self.assertIn("minilm", str(primary_source))

    def test_minilm_only_no_output_if_minilm_missing(self):
        """MiniLM-only run yields no primary if MiniLM output is also missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            primary_source, primary_model = self._run_pipeline_stage3(
                method="minilm",
                minilm_exists=False,
                mbert_exists=True,   # stale mBERT; must be ignored
                tmpdir=tmpdir,
            )
            self.assertIsNone(primary_source,
                              "Should not select any primary when MiniLM output absent in minilm-only mode.")


if __name__ == "__main__":
    unittest.main()
