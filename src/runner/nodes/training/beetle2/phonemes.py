import json
from pathlib import Path


LEGACY_SYMBOLS = (
    "$"
    ';:,.!?¡¿—…"«»“” '
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘ"
    "ɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)
PHONEME_SPECIAL_COUNT = 3
PHONEME_EXTENSIONS = {
    "ɚ": ("ə", "ɹ"),
    "ᵻ": ("ɨ",),
}


class PlBertVocabulary:
    def __init__(self, directory: Path) -> None:
        payload = json.loads((directory / "phonemes.json").read_text(encoding="utf-8"))
        artifact_symbols = tuple(payload["symbols"])
        extensions = tuple(
            symbol for symbol in PHONEME_EXTENSIONS if symbol not in artifact_symbols
        )
        self.artifact_token_count = len(artifact_symbols)
        self.symbols = artifact_symbols + extensions
        self.token_ids = {
            symbol: index for index, symbol in enumerate(self.symbols)
        }
        self.pad_id = self.token_ids["[PAD]"]
        self.mask_id = self.token_ids["[MASK]"]

    def phonemes(self, text: str) -> list[int]:
        unknown_id = self.token_ids["[UNK]"]
        return [self.token_ids.get(symbol, unknown_id) for symbol in text]

    def text_bytes(self, text: str) -> list[int]:
        return [PHONEME_SPECIAL_COUNT + value for value in text.encode("utf-8")]


class PhonemeTokenizer:
    def __init__(self, vocabulary: PlBertVocabulary) -> None:
        self.vocabulary = vocabulary

    def encode(self, text: str) -> list[int]:
        return self.vocabulary.phonemes(text)


class TextTokenizer:
    def __init__(self, vocabulary: PlBertVocabulary) -> None:
        self.vocabulary = vocabulary

    def encode(self, text: str) -> list[int]:
        return self.vocabulary.text_bytes(text)
