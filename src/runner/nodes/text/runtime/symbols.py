from __future__ import annotations

import logging

PAD_SYMBOL = "$"
PUNCTUATION_SYMBOLS = ';:,.!?¡¿—…"«»“” '
LETTER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
IPA_SYMBOLS = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœ"
    "ɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

DEFAULT_STYLETTS_SYMBOLS = [PAD_SYMBOL] + list(PUNCTUATION_SYMBOLS) + list(LETTER_SYMBOLS) + list(IPA_SYMBOLS)
DEFAULT_STYLETTS_SYMBOL_INDEX = {symbol: index for index, symbol in enumerate(DEFAULT_STYLETTS_SYMBOLS)}
symbols = DEFAULT_STYLETTS_SYMBOLS
logger = logging.getLogger(__name__)


def build_symbol_index(symbols: list[str]) -> dict[str, int]:
    return {str(symbol): index for index, symbol in enumerate(symbols)}


def build_word_index_dictionary(symbols_list: list[str]) -> dict[str, int]:
    return build_symbol_index(symbols_list)


def default_styletts_testing_phoneme_symbols() -> list[str]:
    return [str(symbol) for symbol in DEFAULT_STYLETTS_SYMBOLS]


class TextCleaner:
    def __init__(self, symbols: list[str] | None = None):
        self.symbol_index = build_symbol_index(symbols) if symbols is not None else DEFAULT_STYLETTS_SYMBOL_INDEX
        self.chars_logged: set[str] = set()

    def __call__(self, text: str) -> list[int]:
        indexes: list[int] = []
        for character in text:
            if character in self.symbol_index:
                indexes.append(self.symbol_index[character])
            elif character not in self.chars_logged:
                logger.warning("Character %s not found in dictionary", character)
                self.chars_logged.add(character)
        return indexes
