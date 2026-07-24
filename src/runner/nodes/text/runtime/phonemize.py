from __future__ import annotations

from phonemizer.backend import EspeakBackend

DEFAULT_PUNCTUATION_MARKS = ';:,.!?¡¿—…\\"«»\\"\\"'


def phonemize_texts(
    texts: list[str],
    *,
    language: str,
    punctuation_marks: str,
) -> list[str]:
    if not texts:
        return []

    non_empty = [text for text in texts if text.strip()]
    if not non_empty:
        return ["" for text in texts]

    phonemized = _phonemize_with_phonemizer(
        non_empty,
        language=language,
        punctuation_marks=punctuation_marks,
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


_ESPEAK_LANGUAGE_ALIASES = {
    "en": "en-us",
    "english": "en-us",
    "": "en-us",
}

_PHONEMIZER_BACKENDS: dict[tuple[str, str], EspeakBackend] = {}


def _phonemize_with_phonemizer(
    texts: list[str],
    *,
    language: str,
    punctuation_marks: str,
) -> list[str]:
    espeak_language = _ESPEAK_LANGUAGE_ALIASES.get(language.lower(), language)
    backend_key = (espeak_language, punctuation_marks)
    if backend_key not in _PHONEMIZER_BACKENDS:
        _PHONEMIZER_BACKENDS[backend_key] = EspeakBackend(
            espeak_language,
            preserve_punctuation=True,
            punctuation_marks=punctuation_marks,
            with_stress=True,
            language_switch="remove-flags",
        )
    backend = _PHONEMIZER_BACKENDS[backend_key]
    phonemized = backend.phonemize(texts, strip=True)
    return [str(line).strip() for line in phonemized]
