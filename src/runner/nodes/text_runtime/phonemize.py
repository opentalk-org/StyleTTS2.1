from __future__ import annotations

import importlib
import re

DEFAULT_PUNCTUATION_MARKS = ';:,.!?¡¿—…\\"«»\\"\\"'


def phonemize_text(
    text: str,
    *,
    language: str,
    tie: bool,
    punctuation_marks: str,
    espeak_workers: int,
    align_threads: int,
) -> str:
    return phonemize_texts(
        [text],
        language=language,
        tie=tie,
        punctuation_marks=punctuation_marks,
        espeak_workers=espeak_workers,
        align_threads=align_threads,
    )[0]


def phonemize_texts(
    texts: list[str],
    *,
    language: str,
    tie: bool,
    punctuation_marks: str,
    espeak_workers: int,
    align_threads: int,
) -> list[str]:
    if not texts:
        return []

    non_empty = [text for text in texts if text.strip()]
    if not non_empty:
        return ["" for text in texts]

    phonemized = _phonemize_non_empty_texts(
        non_empty,
        language=language,
        tie=tie,
        punctuation_marks=punctuation_marks,
        espeak_workers=espeak_workers,
        align_threads=align_threads,
    )
    output: list[str] = []
    phonemized_index = 0
    for text in texts:
        if text.strip():
            output.append(phonemized[phonemized_index])
            phonemized_index += 1
        else:
            output.append("")
    return output


def _phonemize_non_empty_texts(
    texts: list[str],
    *,
    language: str,
    tie: bool,
    punctuation_marks: str,
    espeak_workers: int,
    align_threads: int,
) -> list[str]:
    try:
        engine_class = importlib.import_module("espeak_align").Engine
    except ImportError as exc:
        raise RuntimeError("espeak_align_not_installed") from exc

    engine = engine_class(language, tie, espeak_workers)
    try:
        batch = engine.align_batch(texts, punctuation_marks, align_threads)
    except Exception:
        batch = None
    if batch is None or len(batch) != len(texts):
        raise RuntimeError("phonemize_batch_length_mismatch")

    output: list[str] = []
    for _words, phoneme_words in batch:
        line = "".join(phoneme_words).strip() if phoneme_words else ""
        output.append(re.sub(r'" *"', '"', line))
    return output
