SYMBOLS = (
    "$"
    ';:,.!?¡¿—…"«»“” '
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘ"
    "ɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)


class PhonemeTokenizer:
    def __init__(self) -> None:
        self.token_ids = {symbol: index for index, symbol in enumerate(SYMBOLS)}

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text).difference(self.token_ids))
        if unknown:
            raise ValueError(f"phonemes are outside the StyleTTS2 vocabulary: {unknown}")
        return [self.token_ids[symbol] for symbol in text]
