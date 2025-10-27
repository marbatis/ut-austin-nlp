"""Evaluation utilities used by several course assignments.

This module intentionally avoids third-party dependencies so that it can be
used in minimal execution environments (e.g., CI or autograders).  The main
entry point is :func:`compute_bleu`, which mirrors the behaviour of the BLEU
score implementation that ships with `tensor2tensor` and `sacrebleu` while
remaining easy to unit test.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Iterator, Tuple

__all__ = ["BleuResult", "compute_bleu"]


_TOKEN_ERROR = "Segments must be provided as sequences of tokens or strings."


@dataclass(frozen=True)
class BleuResult(Mapping[str, object]):
    """Immutable container that exposes BLEU components.

    The object behaves like both a dataclass and a mapping.  It provides
    attribute access (``result.bleu``) as well as dictionary-style access
    (``result["bleu"]``) to make it convenient for different styles of
    downstream code and testing utilities.
    """

    bleu: float
    precisions: Tuple[float, ...]
    brevity_penalty: float
    reference_length: int
    translation_length: int

    _ORDERED_KEYS: Tuple[str, ...] = (
        "bleu",
        "precisions",
        "brevity_penalty",
        "reference_length",
        "translation_length",
    )

    def __iter__(self) -> Iterator[str]:
        return iter(self._ORDERED_KEYS)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._ORDERED_KEYS)

    def __getitem__(self, key: str) -> object:
        if key == "bleu":
            return self.bleu
        if key == "precisions":
            return self.precisions
        if key == "brevity_penalty":
            return self.brevity_penalty
        if key == "reference_length":
            return self.reference_length
        if key == "translation_length":
            return self.translation_length
        raise KeyError(key)

    def __float__(self) -> float:  # pragma: no cover - sugar
        return self.bleu

    def as_dict(self) -> dict[str, object]:
        """Return a standard dictionary representation."""

        return {key: getattr(self, key) for key in self._ORDERED_KEYS}

    # Dataclasses supply a helpful ``__repr__`` automatically.


def _as_token_sequence(segment: Sequence[str] | str | Iterable[str]) -> Tuple[str, ...]:
    """Convert ``segment`` into a tuple of tokens.

    ``segment`` can already be a sequence of tokens or a plain string that is
    tokenised on whitespace.  The helper intentionally supports generic
    iterables so that callers can pass generators without forcing an
    intermediate list allocation.
    """

    if isinstance(segment, str):
        return tuple(segment.split())
    if isinstance(segment, Sequence):
        return tuple(str(token) for token in segment)
    if isinstance(segment, Iterable):
        return tuple(str(token) for token in segment)
    raise TypeError(_TOKEN_ERROR)


def _prepare_corpus(
    reference_corpus: Iterable[Iterable[Sequence[str] | str | Iterable[str]]],
    translation_corpus: Iterable[Sequence[str] | str | Iterable[str]],
) -> tuple[tuple[Tuple[str, ...], ...], tuple[Tuple[str, ...], ...]]:
    prepared_references = []
    for references in reference_corpus:
        candidate_refs = [_as_token_sequence(reference) for reference in references]
        if not candidate_refs:
            raise ValueError("Each translation requires at least one reference.")
        prepared_references.append(tuple(candidate_refs))

    prepared_translations = [
        _as_token_sequence(translation) for translation in translation_corpus
    ]

    if len(prepared_references) != len(prepared_translations):
        raise ValueError(
            "reference_corpus and translation_corpus must contain the same number of segments."
        )

    return tuple(prepared_references), tuple(prepared_translations)


def _extract_ngrams(segment: Tuple[str, ...], max_order: int) -> Counter[Tuple[str, ...]]:
    ngrams: Counter[Tuple[str, ...]] = Counter()
    segment_length = len(segment)
    for order in range(1, max_order + 1):
        max_index = segment_length - order + 1
        if max_index <= 0:
            break
        for start in range(max_index):
            ngram = segment[start : start + order]
            ngrams[ngram] += 1
    return ngrams


def _closest_reference_length(references: Tuple[Tuple[str, ...], ...], translation_len: int) -> int:
    best_length = len(references[0])
    best_diff = abs(best_length - translation_len)
    for reference in references[1:]:
        length = len(reference)
        diff = abs(length - translation_len)
        if diff < best_diff or (diff == best_diff and length < best_length):
            best_diff = diff
            best_length = length
    return best_length


def compute_bleu(
    reference_corpus: Iterable[Iterable[Sequence[str] | str | Iterable[str]]],
    translation_corpus: Iterable[Sequence[str] | str | Iterable[str]],
    *,
    max_order: int = 4,
    smooth: bool = False,
) -> BleuResult:
    """Compute the corpus-level BLEU score.

    Parameters
    ----------
    reference_corpus:
        Iterable where each element is a collection of reference translations
        for a candidate translation.  References can be provided as strings or
        sequences of tokens.
    translation_corpus:
        Iterable containing the candidate translations.  Each translation is a
        string or sequence of tokens.
    max_order:
        Maximum *n*-gram order to consider.  Defaults to 4 (BLEU-4).
    smooth:
        If ``True``, apply add-one smoothing to precision values that would
        otherwise be zero.
    """

    if max_order < 1:
        raise ValueError("max_order must be at least 1.")

    references, translations = _prepare_corpus(reference_corpus, translation_corpus)

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0

    for refs, translation in zip(references, translations):
        translation_length += len(translation)
        reference_length += _closest_reference_length(refs, len(translation))

        merged_ref_ngram_counts: Counter[Tuple[str, ...]] = Counter()
        for reference in refs:
            merged_ref_ngram_counts |= _extract_ngrams(reference, max_order)

        translation_ngram_counts = _extract_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts

        for ngram, count in overlap.items():
            matches_by_order[len(ngram) - 1] += count

        for order in range(1, max_order + 1):
            possible_matches_by_order[order - 1] += max(len(translation) - order + 1, 0)

    precisions: list[float] = []
    smooth_value = 1.0

    for matches, possible in zip(matches_by_order, possible_matches_by_order):
        if possible == 0:
            precisions.append(0.0)
            continue
        if matches > 0:
            precisions.append(matches / possible)
        else:
            precisions.append((matches + smooth_value) / (possible + smooth_value) if smooth else 0.0)

    effective_order = sum(1 for possible in possible_matches_by_order if possible > 0)

    if effective_order == 0:
        geo_mean = 0.0
    else:
        log_precision_sum = 0.0
        for order, (precision, possible) in enumerate(
            zip(precisions, possible_matches_by_order), start=1
        ):
            if possible == 0:
                continue
            if precision > 0:
                log_precision_sum += math.log(precision)
            else:
                geo_mean = 0.0
                break
        else:
            geo_mean = math.exp(log_precision_sum / effective_order)
    if effective_order == 0:
        geo_mean = 0.0

    if translation_length == 0:
        brevity_penalty = 0.0
    elif translation_length > reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - reference_length / translation_length)

    bleu = geo_mean * brevity_penalty

    return BleuResult(
        bleu=bleu,
        precisions=tuple(precisions),
        brevity_penalty=brevity_penalty,
        reference_length=reference_length,
        translation_length=translation_length,
    )
