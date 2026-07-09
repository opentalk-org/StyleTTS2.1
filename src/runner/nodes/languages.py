from __future__ import annotations

from enum import Enum


class Language(str, Enum):
    """Allowed languages for text generation and TTS synthesis.

    Values are ISO 639-1 codes (passed to engines that take a ``language_id``);
    ``display_name`` gives the human-readable name woven into generation prompts.
    """

    ENGLISH = "en"
    CHINESE = "zh"
    SPANISH = "es"
    FRENCH = "fr"
    HINDI = "hi"
    ITALIAN = "it"
    JAPANESE = "ja"
    PORTUGUESE = "pt"
    GERMAN = "de"
    KOREAN = "ko"
    RUSSIAN = "ru"
    DUTCH = "nl"
    POLISH = "pl"
    ARABIC = "ar"
    TURKISH = "tr"

    @property
    def display_name(self) -> str:
        return _DISPLAY_NAMES[self]


_DISPLAY_NAMES: dict[Language, str] = {
    Language.ENGLISH: "English",
    Language.CHINESE: "Chinese",
    Language.SPANISH: "Spanish",
    Language.FRENCH: "French",
    Language.HINDI: "Hindi",
    Language.ITALIAN: "Italian",
    Language.JAPANESE: "Japanese",
    Language.PORTUGUESE: "Portuguese",
    Language.GERMAN: "German",
    Language.KOREAN: "Korean",
    Language.RUSSIAN: "Russian",
    Language.DUTCH: "Dutch",
    Language.POLISH: "Polish",
    Language.ARABIC: "Arabic",
    Language.TURKISH: "Turkish",
}
