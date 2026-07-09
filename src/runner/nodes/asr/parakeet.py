from __future__ import annotations

from pathlib import Path
from typing import Any

from runner.nodes.accelerator_memory import maybe_cuda_half
from runner.nodes.asr.confidence import nemo_hypothesis_confidence
from runner.nodes.assets.model_downloads import single_checkpoint_file


def load_parakeet_model(checkpoint_dir: Path) -> Any:
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise RuntimeError("nemo_toolkit_not_installed") from exc
    weights = single_checkpoint_file(checkpoint_dir, (".nemo",))
    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(weights))
    model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
    _set_cuda_graph_decoder(model, enabled=False)
    _enable_word_confidence(model)
    model.eval()
    return maybe_cuda_half(model)


def _enable_word_confidence(model: Any) -> None:
    """Turn on NeMo per-token/word confidence so each segment carries a real score.

    With ``preserve_word_confidence`` set, greedy decoding attaches a ``word_confidence``
    list (one probability per recognized word, ``max_prob`` of the token softmax,
    aggregated over sub-tokens) to every hypothesis. We later average those per segment.
    """
    from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig, ConfidenceMethodConfig
    from omegaconf import OmegaConf, open_dict

    decoding = getattr(getattr(model, "cfg", None), "decoding", None)
    assert decoding is not None, "parakeet model has no decoding config to enable confidence on"
    confidence_cfg = OmegaConf.structured(
        ConfidenceConfig(
            preserve_frame_confidence=True,
            preserve_token_confidence=True,
            preserve_word_confidence=True,
            exclude_blank=True,
            aggregation="mean",
            method_cfg=ConfidenceMethodConfig(name="max_prob"),
        )
    )
    with open_dict(decoding):
        decoding.confidence_cfg = confidence_cfg
    model.change_decoding_strategy(decoding, verbose=False)


def _set_cuda_graph_decoder(model: Any, *, enabled: bool) -> None:
    from omegaconf import open_dict

    decoding = getattr(getattr(model, "cfg", None), "decoding", None)
    if decoding is None:
        return
    changed = False
    if "greedy" in decoding:
        with open_dict(decoding.greedy):
            decoding.greedy.use_cuda_graph_decoder = enabled
        changed = True
    if "beam" in decoding:
        with open_dict(decoding.beam):
            decoding.beam.allow_cuda_graphs = enabled
        changed = True
    if changed:
        model.change_decoding_strategy(decoding, verbose=False)


def transcribe_wavs_to_segments(
    model: Any,
    wav_paths: list[Path],
    durations_sec: list[float],
    *,
    batch_size: int,
) -> list[list[tuple[float, float, str, float | None]]]:
    import torch

    with torch.no_grad():
        outputs = model.transcribe([str(path) for path in wav_paths], batch_size=batch_size, timestamps=True, num_workers=0)
    return [_segments_from_hypothesis(output, durations_sec[index]) for index, output in enumerate(outputs)]


def transcribe_wavs_to_aligned_segments(
    model: Any,
    wav_paths: list[Path],
    durations_sec: list[float],
    *,
    batch_size: int,
) -> list[list[tuple[float, float, str, float | None, list[dict[str, Any]] | None]]]:
    """Like :func:`transcribe_wavs_to_segments`, but also attach per-word timings.

    Each returned span is ``(start, end, text, confidence, words)`` where ``words``
    is the list of Parakeet word timestamps whose midpoint falls inside the segment
    (or ``None`` when the model emitted no word-level timestamps for it), and
    ``confidence`` is the utterance-level confidence assigned to every segment.
    """
    import torch

    with torch.no_grad():
        outputs = model.transcribe([str(path) for path in wav_paths], batch_size=batch_size, timestamps=True, num_workers=0)
    aligned = []
    for index, output in enumerate(outputs):
        duration = durations_sec[index]
        words = _words_from_hypothesis(output, duration)
        segments = [
            (start, end, text, confidence, _words_in_span(words, start, end))
            for start, end, text, confidence in _segments_from_hypothesis(output, duration)
        ]
        aligned.append(segments)
    return aligned


def _words_from_hypothesis(output: Any, duration_sec: float) -> list[dict[str, Any]]:
    timestamp = getattr(output, "timestamp", None)
    if not (isinstance(timestamp, dict) and isinstance(timestamp.get("word"), list)):
        return []
    # word_confidence (enabled in load_parakeet_model) is one probability per recognized
    # word, in the same order as the word timestamps.
    word_confidence = getattr(output, "word_confidence", None)
    if word_confidence is None:
        word_confidence = []
    elif not isinstance(word_confidence, list):
        raise TypeError("parakeet hypothesis word_confidence must be a list")
    words = []
    for index, item in enumerate(timestamp["word"]):
        if not isinstance(item, dict):
            raise TypeError("parakeet word timestamp entries must be objects")
        text = str(item.get("word", "")).strip()
        if not text:
            continue
        start = max(0.0, float(item.get("start", 0.0)))
        end = max(start, float(item.get("end", start)))
        if duration_sec > 0:
            start = min(start, duration_sec)
            end = min(max(start, end), duration_sec)
        score = _word_confidence_at(word_confidence, index)
        words.append({"word": text, "start": start, "end": end, "score": score})
    return words


def _word_confidence_at(word_confidence: list[Any], index: int) -> float | None:
    if index >= len(word_confidence):
        return None
    return max(0.0, min(1.0, float(word_confidence[index])))


def _words_in_span(words: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]] | None:
    inside = [word for word in words if start <= (word["start"] + word["end"]) / 2 <= end]
    return inside or None


def _segments_from_hypothesis(output: Any, duration_sec: float) -> list[tuple[float, float, str, float | None]]:
    words = _words_from_hypothesis(output, duration_sec)
    # Utterance-level score is the fallback when a segment has no per-word confidence.
    fallback = nemo_hypothesis_confidence(output)
    timestamp = getattr(output, "timestamp", None)
    if isinstance(timestamp, dict) and isinstance(timestamp.get("segment"), list):
        spans = []
        for item in timestamp["segment"]:
            if not isinstance(item, dict):
                raise TypeError("parakeet segment timestamp entries must be objects")
            start, end, text = _span_bounds(item, duration_sec)
            if not text:
                continue
            spans.append((start, end, text, _span_confidence(words, start, end, fallback)))
        return spans
    text = str(getattr(output, "text", output)).strip()
    if not text:
        return []
    return [(0.0, max(0.0, duration_sec), text, fallback)]


def _span_bounds(item: dict, duration_sec: float) -> tuple[float, float, str]:
    start = max(0.0, float(item.get("start", 0.0)))
    end = max(start, float(item.get("end", start)))
    if duration_sec > 0:
        start = min(start, duration_sec)
        end = min(max(start, end), duration_sec)
    return start, end, str(item.get("segment", "")).strip()


def _span_confidence(
    words: list[dict[str, Any]], start: float, end: float, fallback: float | None
) -> float | None:
    """Mean of the per-word confidences whose words fall in ``[start, end]``."""
    inside = _words_in_span(words, start, end) or []
    scores = [word["score"] for word in inside if word.get("score") is not None]
    if not scores:
        return fallback
    return sum(scores) / len(scores)
