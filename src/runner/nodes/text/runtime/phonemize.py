from __future__ import annotations

import importlib
import re

DEFAULT_PUNCTUATION_MARKS = ';:,.!?¡¿—…\\"«»\\"\\"'


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
    except ImportError:
        # ``espeak_align`` is an optional native extension. The canonical StyleTTS2
        # preprocessing uses the pure-Python ``phonemizer`` package on top of the
        # espeak-ng binary, which produces the same single-character IPA (with
        # stress marks) that the StyleTTS2 symbol table expects. Fall back to it.
        return _phonemize_with_phonemizer(
            texts,
            language=language,
            punctuation_marks=punctuation_marks,
        )

    engine = engine_class(language, tie, espeak_workers)
    batch = engine.align_batch(texts, punctuation_marks, align_threads)
    if batch is None or len(batch) != len(texts):
        raise RuntimeError("phonemize_batch_length_mismatch")

    output: list[str] = []
    for _words, phoneme_words in batch:
        line = "".join(phoneme_words).strip() if phoneme_words else ""
        output.append(re.sub(r'" *"', '"', line))
    return output


_ESPEAK_LANGUAGE_ALIASES = {
    "en": "en-us",
    "english": "en-us",
    "": "en-us",
}

_PHONEMIZER_BACKENDS: dict[tuple[str, str], object] = {}


def _phonemize_with_phonemizer(
    texts: list[str],
    *,
    language: str,
    punctuation_marks: str,
) -> list[str]:
    try:
        from phonemizer.backend import EspeakBackend
    except ImportError as exc:  # pragma: no cover - both engines missing
        raise RuntimeError("phonemizer_not_installed") from exc

    espeak_language = _ESPEAK_LANGUAGE_ALIASES.get(language.lower(), language)
    backend_key = (espeak_language, punctuation_marks)
    backend = _PHONEMIZER_BACKENDS.get(backend_key)
    if backend is None:
        backend = EspeakBackend(
            espeak_language,
            preserve_punctuation=True,
            punctuation_marks=punctuation_marks,
            with_stress=True,
            language_switch="remove-flags",
        )
        _PHONEMIZER_BACKENDS[backend_key] = backend
    phonemized = backend.phonemize(texts, strip=True)
    return [str(line).strip() for line in phonemized]
