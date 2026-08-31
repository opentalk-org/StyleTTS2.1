from __future__ import annotations

import re

from runner.nodes.text_processing.polish_inflections import cardinal, genitive_cardinal
from runner.nodes.text_processing.polish_syntax import PARSER


GROUPED_SCALE_RE = re.compile(
    r"\b(?P<prefix>(?:od|do|około|blisko|poniżej|powyżej|sprzed|w wysokości|o powierzchni|na wysokości|w odległości|wartości) )?"
    r"(?P<value>\d{4,5}|\d{1,3}(?: \d{3})+) (?P<scale>tysiąc|tysiące|tysięcy|milion|miliony|milionów|miliard|miliardy|miliardów)\b",
    re.IGNORECASE,
)
SCALE_VALUES = {"tysiąc": 1_000, "tysiące": 1_000, "tysięcy": 1_000, "milion": 1_000_000,
                "miliony": 1_000_000, "milionów": 1_000_000, "miliard": 1_000_000_000,
                "miliardy": 1_000_000_000, "miliardów": 1_000_000_000}
AGE_PAIR_LABEL_RE = re.compile(r"\b(?P<first>\d+),(?P<second>\d+)\b(?=.{0,240}\b(?P=first)-lat\w+.{0,120}\b(?P=second)-lat\w+)", re.IGNORECASE | re.DOTALL)
ARITHMETIC_GOVERNOR_LEMMAS = frozenset({"działanie", "liczyć", "mnożyć", "obliczać", "wykrzykować", "wykrzykiwać"})
COORDINATED_MEASUREMENT_RE = re.compile(r"\b(?P<first>\d+) (?P<unit>metrów|kilometrów) (?P<first_dimension>[^\W\d_]+) (?P<link>i|oraz) (?P<second>\d+) (?P=unit) (?P<second_dimension>[^\W\d_]+)\b", re.IGNORECASE)


def expand_grouped_scales(text: str) -> str:
    return GROUPED_SCALE_RE.sub(grouped_scale_words, text)


def expand_age_pair_labels(text: str) -> str:
    return AGE_PAIR_LABEL_RE.sub(lambda match: f"{cardinal(int(match['first']))} {cardinal(int(match['second']))}", text)


def expand_coordinated_measurements(text: str) -> str:
    return COORDINATED_MEASUREMENT_RE.sub(lambda match: f"{cardinal(int(match['first']))} {match['unit']} {match['first_dimension']} {match['link']} {cardinal(int(match['second']))} {match['unit']} {match['second_dimension']}", text)


def demonstrative_cardinal_words(match: re.Match[str]) -> str:
    document = PARSER(match.string)
    number = next(token for token in document if token.idx == match.start("value"))
    modifier = next(token for token in document if token.idx == match.start("modifier"))
    existential = number.head.dep_ == "nsubj" and number.head.head.lemma_.split(":", 1)[0].lower() == "być"
    personal_accusative = number.morph.get("Case") == ["Acc"] and number.head.morph.get("Animacy") == ["Hum"]
    genitive = not existential and (modifier.morph.get("Case") == ["Gen"] or personal_accusative)
    return f"{match['modifier']} {(genitive_cardinal if genitive else cardinal)(int(match['value']))}"


def grouped_scale_words(match: re.Match[str]) -> str:
    value = int(match["value"].replace(" ", "")) * SCALE_VALUES[match["scale"].lower()]
    return f"{match['prefix'] or ''}{(genitive_cardinal if match['prefix'] else cardinal)(value)}"


def polish_compound_prefix(value: int) -> str:
    ones = ("", "jedno", "dwu", "trzy", "cztero", "pięcio", "sześcio", "siedmio", "ośmio", "dziewięcio")
    teens = ("dziesięcio", "jedenasto", "dwunasto", "trzynasto", "czternasto", "piętnasto", "szesnasto", "siedemnasto", "osiemnasto", "dziewiętnasto")
    tens = ("", "", "dwudziesto", "trzydziesto", "czterdziesto", "pięćdziesięcio", "sześćdziesięcio", "siedemdziesięcio", "osiemdziesięcio", "dziewięćdziesięcio")
    hundreds = ("", "stu", "dwustu", "trzystu", "czterystu", "pięciuset", "sześciuset", "siedmiuset", "ośmiuset", "dziewięciuset")
    if value >= 100:
        hundred, remainder = divmod(value, 100)
        return f"{hundreds[hundred]}{polish_compound_prefix(remainder)}"
    if value < 10:
        return ones[value]
    if value < 20:
        return teens[value - 10]
    decade, unit = divmod(value, 10)
    return f"{tens[decade]}{ones[unit]}"


def dimension_words(match: re.Match[str]) -> str:
    values = re.split(r"\s*[x×]\s*", match.group(), flags=re.IGNORECASE)
    return " na ".join(cardinal(int(value)) for value in values)


def expand_dimensions(text: str, pattern: re.Pattern[str]) -> str:
    document = PARSER(text)
    arithmetic = {
        token.idx for token in document if pattern.fullmatch(token.text) and (
            token.head.lemma_.lower() in ARITHMETIC_GOVERNOR_LEMMAS
            or any(part.lemma_.lower() in ARITHMETIC_GOVERNOR_LEMMAS for part in document[max(0, token.i - 2):token.i])
        )
    }
    return pattern.sub(lambda match: " razy ".join(cardinal(int(value)) for value in re.split(r"\s*[x×]\s*", match.group()))
                       if match.start() in arithmetic else dimension_words(match), text)


def has_ambiguous_dotted_group(text: str, pattern: re.Pattern[str]) -> bool:
    document = PARSER(text)
    for match in pattern.finditer(text):
        following = next((part for part in document if part.idx >= match.end()), None)
        preceding = text[:match.start()].lower().rstrip()
        governed = re.search(r"\b(?:od|do|około|blisko|poniżej|powyżej|sprzed|w wysokości|o powierzchni|na wysokości|w odległości|wartości)$", preceding)
        measured = following is not None and (following.pos_ == "NOUN" or following.lower_ in {"g", "kg", "km", "m", "ml", "zł"})
        if governed is None and not measured:
            return True
    return False


def dotted_group_words(match: re.Match[str]) -> str:
    preceding = match.string[:match.start()].lower().rstrip()
    governed = re.search(r"\b(?:od|do|około|blisko|poniżej|powyżej|sprzed|w wysokości|o powierzchni|na wysokości|w odległości|wartości)$", preceding)
    value = int(match.group().replace(".", ""))
    return (genitive_cardinal if governed is not None else cardinal)(value)


def age_adjective_words(match: re.Match[str]) -> str:
    prefix = match["prefix"] or ""
    suffix = "latka" if prefix and match["suffix"].lower() == "latek" else match["suffix"]
    return f"{prefix}{polish_compound_prefix(int(match['value']))}{suffix}"


def currency_noun_phrase(match: re.Match[str]) -> str:
    value = int(match["value"])
    symbol = match.group()[-1]
    forms = {"$": ("dolar", "dolary", "dolarów"), "£": ("funt", "funty", "funtów")}
    if symbol == "€":
        return f"{value} euro"
    singular, few, many = forms[symbol]
    noun = singular if value == 1 else few if value % 10 in {2, 3, 4} and value % 100 not in {12, 13, 14} else many
    return f"{value} {noun}"
