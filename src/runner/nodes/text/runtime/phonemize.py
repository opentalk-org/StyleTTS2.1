from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import pyopenjtalk
from phonemizer.backend import EspeakBackend

DEFAULT_PUNCTUATION_MARKS = ';:,.!?¡¿—…\\"«»\\"\\"'


def phonemize_texts(
    texts: list[str],
    *,
    language: str,
    punctuation_marks: str,
) -> list[str]:
    language = _normalized_language(language)
    if not texts:
        return []

    non_empty = [text for text in texts if text.strip()]
    if not non_empty:
        return ["" for text in texts]

    phonemized = _phonemize_non_empty(non_empty, language, punctuation_marks)
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
    "fr": "fr-fr",
    "no": "nb",
    "zh-yue": "yue",
    # These corpus labels have no dedicated voice in the installed eSpeak build.
    # Keep the fallback explicit so it can be audited and replaced by a dedicated
    # frontend later instead of being silently interpreted as English.
    "ary": "ar",   # Moroccan Arabic
    "arz": "ar",   # Egyptian Arabic
    "azb": "az",   # South Azerbaijani
    "gom": "kok",  # Goan Konkani
    "pcm": "en-gb",  # Nigerian Pidgin
    "pnb": "ur",   # Western Punjabi (Shahmukhi)
    "skr": "ur",   # Saraiki
}

_PHONEMIZER_BACKENDS: dict[tuple[str, str], EspeakBackend] = {}


def _normalized_language(language: str | None) -> str:
    normalized = (language or "").strip().lower().replace("_", "-")
    if not normalized or normalized in {"missing", "[missing]", "und", "unknown"}:
        raise ValueError(
            "phoneme_language_missing: set an explicit language before phonemization; "
            "missing languages are never treated as English"
        )
    return normalized


def phonemizer_backend(language: str) -> str:
    """Return the backend selected for a corpus language code."""
    language = _normalized_language(language)
    if language == "ja":
        return "pyopenjtalk"
    if language in {"zh", "zh-tw"}:
        return "g2pw"
    if language == "ko":
        return "g2pk+espeak"
    if language == "zh-yue":
        return "espeak:yue"
    return f"espeak:{_ESPEAK_LANGUAGE_ALIASES.get(language, language)}"


def _phonemize_non_empty(texts: list[str], language: str, punctuation_marks: str) -> list[str]:
    if language == "ja":
        return [pyopenjtalk.g2p(text, kana=False).strip() for text in texts]
    if language in {"zh", "zh-tw"}:
        return _phonemize_mandarin(texts, punctuation_marks)
    if language == "ko":
        pronounced = [_korean_g2p()(text)[0] for text in texts]
        return _phonemize_with_phonemizer(pronounced, language="ko", punctuation_marks=punctuation_marks)
    return _phonemize_with_phonemizer(texts, language=language, punctuation_marks=punctuation_marks)


@lru_cache(maxsize=1)
def _korean_g2p():
    from misaki.ko import KOG2P

    return KOG2P()


@lru_cache(maxsize=1)
def _mandarin_g2p():
    from g2pw import G2PWConverter

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache")) / "g2pw"
    return G2PWConverter(
        model_dir=str(cache_root),
        style="pinyin",
        enable_non_tradional_chinese=True,
    )


def _phonemize_mandarin(texts: list[str], punctuation_marks: str) -> list[str]:
    rows = _mandarin_g2p()(texts)
    pinyin_texts: list[str] = []
    for text, syllables in zip(texts, rows, strict=True):
        parts: list[str] = []
        for character, syllable in zip(text, syllables, strict=True):
            parts.append(character if syllable is None else f" {syllable} ")
        pinyin_texts.append("".join(parts).strip())
    # G2PW resolves contextual readings; eSpeak's pinyin frontend then converts
    # those resolved, tone-bearing syllables into the same IPA alphabet used by
    # the other eSpeak routes.
    return _phonemize_with_phonemizer(
        pinyin_texts,
        language="cmn-latn-pinyin",
        punctuation_marks=punctuation_marks,
    )


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
            tie=True,
            language_switch="remove-flags",
        )
    backend = _PHONEMIZER_BACKENDS[backend_key]
    phonemized = backend.phonemize(texts, strip=True)
    return [str(line).strip() for line in phonemized]
