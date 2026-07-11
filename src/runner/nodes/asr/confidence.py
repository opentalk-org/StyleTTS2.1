from __future__ import annotations

import math
from typing import Any


def confidence_from_avg_logprob(avg_logprob: Any) -> float | None:
    """Map a mean per-token log-probability to a ``[0, 1]`` confidence.

    Whisper reports a per-segment ``avg_logprob`` (mean log-probability of the
    tokens in the segment); ``exp`` turns it into a probability. Returns ``None``
    when the value is missing or non-finite.
    """
    if avg_logprob is None:
        return None
    value = float(avg_logprob)
    if not math.isfinite(value):
        return None
    return _clamp01(math.exp(value))


def nemo_hypothesis_confidence(output: Any) -> float | None:
    """Per-utterance confidence for a NeMo ``Hypothesis`` as ``exp(mean token log-prob)``.

    NeMo's ``Hypothesis.score`` is the summed log-probability of the decoded
    token sequence; dividing by the token count yields a mean per-token log-prob,
    and ``exp`` maps it to a probability in ``[0, 1]``. Both Parakeet (RNNT) and
    Canary (AED) hypotheses expose this. Returns ``None`` when the score or token
    count is unavailable (e.g. when ``transcribe`` yields a plain string).
    """
    score = getattr(output, "score", None)
    if score is None:
        return None
    token_count = _sequence_length(getattr(output, "y_sequence", None))
    if token_count <= 0:
        return None
    mean_logprob = float(score) / token_count
    return confidence_from_avg_logprob(mean_logprob)


def mean_word_confidence(words: list[dict[str, Any]] | None) -> float | None:
    """Mean of the per-word ``score`` values, ignoring words whose score is ``None``.

    Segments carry a single ``confidence`` (the average over their words) rather
    than a score on every alignment entry. Returns ``None`` when there are no
    words or none of them has a score.
    """
    if not words:
        return None
    scores = [word["score"] for word in words if word["score"] is not None]
    if not scores:
        return None
    return _clamp01(sum(scores) / len(scores))


def _sequence_length(sequence: Any) -> int:
    if sequence is None:
        return 0
    return int(len(sequence))


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
