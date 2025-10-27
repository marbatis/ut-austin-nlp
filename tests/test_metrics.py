import math

import pytest

from ut_austin_nlp.metrics import BleuResult, compute_bleu


def test_compute_bleu_perfect_match():
    reference_corpus = [[["the", "cat", "sat", "on", "the", "mat"]]]
    translation_corpus = [["the", "cat", "sat", "on", "the", "mat"]]

    result = compute_bleu(reference_corpus, translation_corpus)

    assert isinstance(result, BleuResult)
    assert math.isclose(result.bleu, 1.0)
    assert result["bleu"] == pytest.approx(1.0)
    assert result.translation_length == 6
    assert result.reference_length == 6


def test_compute_bleu_brevity_penalty():
    reference_corpus = [[["the", "cat", "is"]]]
    translation_corpus = [["the", "cat"]]

    result = compute_bleu(reference_corpus, translation_corpus)

    expected_bp = math.exp(1 - 3 / 2)
    assert result.brevity_penalty == pytest.approx(expected_bp)
    assert result.precisions[0] == pytest.approx(1.0)
    assert result.precisions[1] == pytest.approx(1.0)
    assert result.bleu == pytest.approx(expected_bp)


def test_compute_bleu_smoothing():
    reference_corpus = [[["alpha", "beta"]]]
    translation_corpus = [["gamma", "delta"]]

    unsmoothed = compute_bleu(reference_corpus, translation_corpus)
    smoothed = compute_bleu(reference_corpus, translation_corpus, smooth=True)

    assert unsmoothed.bleu == 0.0
    assert smoothed.bleu > 0.0


def test_compute_bleu_accepts_strings_and_generators():
    references = [["a b c", (token for token in ["a", "b"])]]
    translations = ["a b c"]

    result = compute_bleu(references, translations)

    assert result.bleu > 0.0


def test_compute_bleu_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_bleu([[["a"]]], [])


def test_bleu_result_mapping_behaviour():
    reference_corpus = [[["hello", "world"]]]
    translation_corpus = [["hello", "world"]]

    result = compute_bleu(reference_corpus, translation_corpus)

    assert dict(result)["bleu"] == pytest.approx(1.0)
    assert float(result) == pytest.approx(1.0)
    assert len(result) == 5


def test_compute_bleu_rejects_invalid_order():
    with pytest.raises(ValueError):
        compute_bleu([[["token"]]], [["token"]], max_order=0)
