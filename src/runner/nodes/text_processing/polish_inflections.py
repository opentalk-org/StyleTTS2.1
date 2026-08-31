from __future__ import annotations

import morfeusz2
from num2words import num2words


GENITIVE_CARDINAL_WORDS = {
    "zero": "zera", "jeden": "jednego", "dwa": "dwóch", "trzy": "trzech",
    "cztery": "czterech", "pięć": "pięciu", "sześć": "sześciu",
    "siedem": "siedmiu", "osiem": "ośmiu", "dziewięć": "dziewięciu",
    "dziesięć": "dziesięciu", "jedenaście": "jedenastu", "dwanaście": "dwunastu",
    "trzynaście": "trzynastu", "czternaście": "czternastu", "piętnaście": "piętnastu",
    "szesnaście": "szesnastu", "siedemnaście": "siedemnastu",
    "osiemnaście": "osiemnastu", "dziewiętnaście": "dziewiętnastu",
    "dwadzieścia": "dwudziestu", "trzydzieści": "trzydziestu",
    "czterdzieści": "czterdziestu", "pięćdziesiąt": "pięćdziesięciu",
    "sześćdziesiąt": "sześćdziesięciu", "siedemdziesiąt": "siedemdziesięciu",
    "osiemdziesiąt": "osiemdziesięciu", "dziewięćdziesiąt": "dziewięćdziesięciu",
    "sto": "stu", "dwieście": "dwustu", "trzysta": "trzystu", "czterysta": "czterystu",
    "pięćset": "pięciuset", "sześćset": "sześciuset", "siedemset": "siedmiuset",
    "osiemset": "ośmiuset", "dziewięćset": "dziewięciuset", "tysiąc": "tysiąca",
    "tysiące": "tysięcy", "tysięcy": "tysięcy", "milion": "miliona",
    "miliony": "milionów", "milionów": "milionów",
}
LOCATIVE_CARDINAL_WORDS = GENITIVE_CARDINAL_WORDS | {
    "zero": "zerze", "jeden": "jednym", "tysiąc": "tysiącu",
    "tysiące": "tysiącach", "tysięcy": "tysiącach", "milion": "milionie",
    "miliony": "milionach", "milionów": "milionach",
}
DATIVE_CARDINAL_WORDS = GENITIVE_CARDINAL_WORDS | {
    "jeden": "jednemu", "dwa": "dwóm", "trzy": "trzem", "cztery": "czterem",
}
MORFEUSZ = morfeusz2.Morfeusz()


def cardinal(value: int) -> str:
    return str(num2words(value, lang="pl"))


def ordinal(value: int) -> str:
    if value == 0:
        return "zerowy"
    words = str(num2words(value, lang="pl", to="ordinal"))
    return words.replace("trzysetny", "trzechsetny").replace("czterysetny", "czterechsetny")


def genitive_cardinal(value: int) -> str:
    words = cardinal(value).split()
    inflected = [GENITIVE_CARDINAL_WORDS[word] for word in words]
    if len(words) > 1 and words[-1] == "jeden":
        inflected[-1] = "jeden"
    return " ".join(inflected)


def locative_cardinal(value: int) -> str:
    words = cardinal(value).split()
    inflected = [LOCATIVE_CARDINAL_WORDS[word] for word in words]
    if len(words) > 1 and words[-1] == "jeden":
        inflected[-1] = "jeden"
    return " ".join(inflected)


def dative_cardinal(value: int) -> str:
    words = cardinal(value).split()
    inflected = [DATIVE_CARDINAL_WORDS[word] for word in words]
    if len(words) > 1 and words[-1] == "jeden":
        inflected[-1] = "jeden"
    return " ".join(inflected)


def instrumental_cardinal(value: int) -> str:
    if value == 1:
        return "jednym"
    inflected = []
    words = cardinal(value).split()
    for word in words:
        if word == "jeden" and len(words) > 1:
            inflected.append(word)
            continue
        forms = {surface for surface, _, tag, _, _ in MORFEUSZ.generate(word) if ":inst:" in tag and ":congr" in tag}
        preferred = {surface for surface in forms if surface.endswith("oma")} or forms
        assert len(preferred) == 1, f"ambiguous Polish instrumental cardinal: {word}"
        inflected.append(preferred.pop())
    return " ".join(inflected)


def genitive_ordinal(value: int) -> str:
    return " ".join(_ordinal_ending(word, "ego", "iego") for word in ordinal(value).split())


def locative_ordinal(value: int) -> str:
    return " ".join(_ordinal_ending(word, "ym", "im") for word in ordinal(value).split())


def feminine_genitive_ordinal(value: int) -> str:
    if value == 0:
        return "zerowej"
    return _feminine_ordinal_words(value, "gen", "ej", "iej")


def feminine_nominative_ordinal(value: int) -> str:
    if value == 0:
        return "zerowa"
    return _feminine_ordinal_words(value, "nom.voc", "a", "ia")


def feminine_accusative_ordinal(value: int) -> str:
    if value == 0:
        return "zerową"
    return _feminine_ordinal_words(value, "acc", "ą", "ią")


def plural_nominative_ordinal(value: int, masculine_personal: bool) -> str:
    gender = "m1" if masculine_personal else "m2.m3.f.n"
    marker = f"adj:pl:nom.voc:{gender}:"
    inflected = []
    for word in ordinal(value).split():
        generated = [surface for surface, _, tag, _, _ in MORFEUSZ.generate(word) if marker in tag]
        assert generated, f"unsupported Polish plural ordinal: {word}"
        inflected.append(generated[0])
    return " ".join(inflected)


def _feminine_ordinal_words(value: int, case: str, y_ending: str, i_ending: str) -> str:
    inflected = []
    tag_marker = f"adj:sg:{case}:f:"
    for word in ordinal(value).split():
        generated = [surface for surface, _, tag, _, _ in MORFEUSZ.generate(word) if tag_marker in tag]
        inflected.append(generated[0] if generated else _ordinal_ending(word, y_ending, i_ending))
    return " ".join(inflected)


def _ordinal_ending(word: str, y_ending: str, i_ending: str) -> str:
    if word.endswith("y"):
        return f"{word[:-1]}{y_ending}"
    if word.endswith("i"):
        return f"{word[:-1]}{i_ending}"
    return word
