from __future__ import annotations

import logging

from traintts.model_bert_symbols import MODEL_BERT_SYMBOLS

PAD_SYMBOL = "$"
START_SYMBOL = "<start/>"
END_SYMBOL = "<end/>"
PUNCTUATION_SYMBOLS = ';:,.!?¡¿—…"«»“” '
LETTER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
IPA_SYMBOLS = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœ"
    "ɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

DEFAULT_STYLETTS_SYMBOLS = (
    [PAD_SYMBOL]
    + list(PUNCTUATION_SYMBOLS)
    + list(LETTER_SYMBOLS)
    + list(IPA_SYMBOLS)
    + [START_SYMBOL, END_SYMBOL]
)
DEFAULT_STYLETTS_SYMBOL_INDEX = {symbol: index for index, symbol in enumerate(DEFAULT_STYLETTS_SYMBOLS)}
MODEL_BERT_STYLETTS_SYMBOLS = DEFAULT_STYLETTS_SYMBOLS + [
    symbol for symbol in MODEL_BERT_SYMBOLS
    if symbol not in DEFAULT_STYLETTS_SYMBOLS
]
logger = logging.getLogger(__name__)


def build_symbol_index(symbols: list[str]) -> dict[str, int]:
    return {str(symbol): index for index, symbol in enumerate(symbols)}


def build_word_index_dictionary(symbols_list: list[str]) -> dict[str, int]:
    return build_symbol_index(symbols_list)

class TextCleaner:
    def __init__(self, symbols: list[str] | None = None):
        self.symbol_index = build_symbol_index(symbols) if symbols is not None else DEFAULT_STYLETTS_SYMBOL_INDEX
        self.multichar_symbols = sorted(
            (symbol for symbol in self.symbol_index if len(symbol) > 1),
            key=len,
            reverse=True,
        )
        self.chars_logged: set[str] = set()

    def __call__(self, text: str) -> list[int]:
        indexes: list[int] = []
        position = 0
        while position < len(text):
            symbol = next(
                (
                    candidate
                    for candidate in self.multichar_symbols
                    if text.startswith(candidate, position)
                ),
                None,
            )
            if symbol is not None:
                indexes.append(self.symbol_index[symbol])
                position += len(symbol)
                continue
            character = text[position]
            if character in self.symbol_index:
                indexes.append(self.symbol_index[character])
            elif character not in self.chars_logged:
                logger.warning("Character %s not found in dictionary", character)
                self.chars_logged.add(character)
            position += 1
        return indexes
