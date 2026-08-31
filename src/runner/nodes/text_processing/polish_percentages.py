from __future__ import annotations

import re
from collections.abc import Callable

from runner.nodes.text_processing.polish_inflections import cardinal, genitive_cardinal, instrumental_cardinal, locative_cardinal
from runner.nodes.text_processing.polish_syntax import GENITIVE_VERB_LEMMAS, MORFEUSZ, PARSER
from runner.nodes.text_processing.polish_valency import GENITIVE_PERCENT_NOUN_LEMMAS


DECIMAL_RE = re.compile(r"(?P<whole>\d+)[,.](?P<fraction>\d+)")
CHANGE_AMOUNT_VERB_LEMMAS = frozenset({"maleć", "obniżać", "obniżyć", "podnieść", "podnosić", "rosnąć", "spadać", "spaść", "wydłużać", "wydłużyć", "wzrastać", "wzrosnąć", "zwiększać", "zwiększyć", "zmniejszać", "zmniejszyć"})
ACCUSATIVE_O_VERB_LEMMAS = CHANGE_AMOUNT_VERB_LEMMAS | frozenset({"apelować", "błagać", "chodzić", "dbać", "martwić", "modlić", "prosić", "pytać", "starać", "troszczyć", "ubiegać", "walczyć", "wnosić", "zabiegać", "zadbać"})
COORDINATED_LOCATIVE_RE = re.compile(
    r"\b(?P<prefix>w|przy) (?P<first>\d+(?:[,.]\d+)?)\s*%(?P<middle>\s+[^0-9%.,;!?]*?\b(?:i|oraz|lub|albo)\s+)(?P<second>\d+(?:[,.]\d+)?)\s*%",
    re.IGNORECASE,
)
NOMINAL_GENITIVE_RE = re.compile(r"\b(?P<head>[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+) (?P<modifier>(?:blisko|niemal|około|ponad|prawie|zaledwie) )?(?P<value>\d+)% (?P<complement>[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)\b")
NOMINAL_Z_PERCENT_RE = re.compile(r"\b(?P<head>[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+) z (?P<value>\d+(?:[,.]\d+)?)\s*%", re.IGNORECASE)
APPROXIMATE_Z_RE = re.compile(r"\b(?P<prefix>tak) z (?P<value>\d+(?:[,.]\d+)?)\s*%", re.IGNORECASE)
COPULAR_APPROXIMATE_Z_RE = re.compile(r"\b(?P<copula>\w+) (?P<modifier>nawet|może) z (?P<value>\d+(?:[,.]\d+)?)\s*%", re.IGNORECASE)
VERBAL_APPROXIMATE_Z_RE = re.compile(r"\bz (?P<value>\d+(?:[,.]\d+)?)\s*%", re.IGNORECASE)
GENITIVE_VERBAL_PERCENT_RE = re.compile(r"\b(?P<governor>[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+) (?P<modifier>(?:blisko|nawet|niemal|około|ponad|prawie) )?(?P<value>\d+(?:[,.]\d+)?)\s*%", re.IGNORECASE)
APPROXIMATE_DECIMAL_RE = re.compile(r"\bw okolicach (?P<value>\d+,\d+)\b(?!\s*%)", re.IGNORECASE)
GENITIVE_DECIMAL_RE = re.compile(r"\b(?P<prefix>blisko|do|od|około|poniżej|powyżej|sprzed|wobec) (?P<value>\d+[,.]\d+)\b(?!\s*%)", re.IGNORECASE)


def expand_contextual_percentages(text: str) -> str:
    text = NOMINAL_Z_PERCENT_RE.sub(_nominal_z_percent_words, text)
    text = APPROXIMATE_Z_RE.sub(lambda match: f"{match['prefix']} z {number_words(match['value'])} procent", text)
    text = COPULAR_APPROXIMATE_Z_RE.sub(_copular_approximation_words, text)
    text = VERBAL_APPROXIMATE_Z_RE.sub(_verbal_approximation_words, text)
    text = GENITIVE_VERBAL_PERCENT_RE.sub(_genitive_verbal_percent_words, text)
    text = APPROXIMATE_DECIMAL_RE.sub(lambda match: f"w okolicach {number_words(match['value'], genitive_cardinal)}", text)
    text = GENITIVE_DECIMAL_RE.sub(lambda match: f"{match['prefix']} {number_words(match['value'], genitive_cardinal)}", text)
    text = COORDINATED_LOCATIVE_RE.sub(
        lambda match: f"{match['prefix']} {_locative_percent(match['first'])}{match['middle']}{_locative_percent(match['second'])}",
        text,
    )
    return NOMINAL_GENITIVE_RE.sub(_nominal_genitive_words, text)


def _nominal_z_percent_words(match: re.Match[str]) -> str:
    analyses = [item for _, _, item in MORFEUSZ.analyse(match["head"])]
    nominal = any(item[2].startswith(("subst:sg:nom", "subst:sg:acc", "subst:pl:nom", "subst:pl:acc")) for item in analyses) and not any(item[2].startswith("ger:") for item in analyses)
    return f"{match['head']} z {number_words(match['value'], instrumental_cardinal)} procentami" if nominal else match.group()


def has_ambiguous_negated_percent(text: str) -> bool:
    document = PARSER(text)
    if any(token.text == "%" and token.i >= 2 and token.nbor(-2).lemma_.lower() == "o" and re.search(r"[,.]", token.nbor(-1).text) and not any(ancestor.pos_ in {"AUX", "VERB"} for ancestor in token.ancestors) for token in document):
        return True
    for token in document:
        if token.text != "%" or token.head.pos_ not in {"AUX", "VERB"}:
            continue
        negated = any(child.dep_ == "advmod:neg" for child in token.head.children)
        copular = token.head.lemma_.split(":", 1)[0].lower() == "być"
        governed = any(child.dep_ == "case" for child in token.children)
        if negated and not copular and not governed:
            return True
    return False


def _copular_approximation_words(match: re.Match[str]) -> str:
    copular = any(interpretation[1].split(":", 1)[0] == "być" for _, _, interpretation in MORFEUSZ.analyse(match["copula"]))
    return f"{match['copula']} {match['modifier']} z {number_words(match['value'])} procent" if copular else match.group()


def _verbal_approximation_words(match: re.Match[str]) -> str:
    before = re.search(r"([A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)\s+$", match.string[:match.start()])
    after = re.match(r"\s+([A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)", match.string[match.end():])
    preceding_noun = before is None or any(analysis[2][2].startswith(("depr:", "subst:")) for analysis in MORFEUSZ.analyse(before[1]))
    following_verb = after is not None and any(analysis[2][2].startswith(("bedzie:", "fin:", "impt:", "praet:")) for analysis in MORFEUSZ.analyse(after[1]))
    return f"z {number_words(match['value'])} procent" if following_verb and not preceding_noun else match.group()


def _genitive_verbal_percent_words(match: re.Match[str]) -> str:
    analyses = MORFEUSZ.analyse(match["governor"])
    lemmas = {analysis[2][1].split(":", 1)[0].lower() for analysis in analyses}
    tags = {analysis[2][2].split(":", 1)[0] for analysis in analyses}
    genitive = bool(lemmas & GENITIVE_VERB_LEMMAS) or any(analysis[2][2].startswith("ger:sg:") for analysis in analyses) and not tags & {"bedzie", "fin", "impt", "praet"}
    return f"{match['governor']} {match['modifier'] or ''}{number_words(match['value'], genitive_cardinal)} {'procenta' if match['value'] == '1' else 'procent'}" if genitive else match.group()


def _nominal_genitive_words(match: re.Match[str]) -> str:
    document = PARSER(match.string)
    head_token = next(token for token in document if token.idx == match.start("head"))
    percent_token = next(token for token in document if token.text == "%" and token.idx == match.end("value"))
    head_analyses = MORFEUSZ.analyse(match["head"])
    head_numbers = {analysis[2][2].split(":")[1] for analysis in head_analyses if analysis[2][2].startswith("subst:")}
    head_plural = head_numbers == {"pl"} and all(analysis[2][2].startswith(("subst:", "depr:")) for analysis in head_analyses)
    complement_genitive = any(re.match(r"subst:(?:sg|pl):gen", analysis[2][2]) for analysis in MORFEUSZ.analyse(match["complement"]))
    dependency_genitive = head_plural and percent_token.head == head_token and percent_token.dep_ in {"nmod", "nmod:arg"}
    if not complement_genitive or not (head_token.lemma_.lower() in GENITIVE_PERCENT_NOUN_LEMMAS or dependency_genitive):
        return match.group()
    return f"{match['head']} {match['modifier'] or ''}{genitive_cardinal(int(match['value']))} {'procenta' if match['value'] == '1' else 'procent'} {match['complement']}"


def _locative_percent(value: str) -> str:
    if value == "1":
        return "jednym procencie"
    words = number_words(value) if re.search(r"[,.]", value) else genitive_cardinal(int(value))
    return f"{words} procentach"


def governed_percent_words(match: re.Match[str]) -> str:
    prefix = match["prefix"]
    modifier = match["modifier"] or ""
    value = match["value"]
    accusative = prefix.lower().endswith(" o") and prefix.lower() != "o" or prefix.lower() == "na około" or prefix.lower() == "o" and _is_accusative_o(match)
    if value == "1":
        if accusative:
            return f"{prefix} {modifier}jeden procent"
        if prefix.lower() in {"o", "przy", "w"}:
            return f"{prefix} {modifier}jednym procencie"
        return f"{prefix} {modifier}jednego procenta"
    suffix = "procent" if accusative else "procentach" if prefix.lower() in {"o", "przy", "w"} else "procent"
    inflector = cardinal if accusative else locative_cardinal if prefix.lower() in {"o", "przy", "w"} else genitive_cardinal
    words = number_words(value, inflector) if re.search(r"[,.]", value) else inflector(int(value))
    return f"{prefix} {modifier}{words} {suffix}"


def _is_accusative_o(match: re.Match[str]) -> bool:
    document = PARSER(match.string)
    number = next(token for token in document if token.idx == match.start("value"))
    return any(token.lemma_.lower() in ACCUSATIVE_O_VERB_LEMMAS or any(item[2][1].split(":", 1)[0].lower() in ACCUSATIVE_O_VERB_LEMMAS for item in MORFEUSZ.analyse(token.text)) for token in number.ancestors)


def decimal_words(match: re.Match[str], inflector: Callable[[int], str] = cardinal) -> str:
    raw_fraction = match["fraction"]
    fraction = inflector(int(raw_fraction)) if len(raw_fraction) <= 2 and not raw_fraction.startswith("0") else " ".join(cardinal(int(digit)) for digit in raw_fraction)
    return f"{inflector(int(match['whole']))} przecinek {fraction}"


def number_words(value: str, inflector: Callable[[int], str] = cardinal) -> str:
    match = DECIMAL_RE.fullmatch(value)
    return decimal_words(match, inflector) if match is not None else inflector(int(value))


def negated_percent_words(match: re.Match[str]) -> str:
    verbal_tags = {"bedzie", "fin", "impt", "inf", "praet", "winien"}
    is_verb = any(
        interpretation[2].split(":", 1)[0] in verbal_tags
        for _, _, interpretation in MORFEUSZ.analyse(match["governor"])
    )
    if not is_verb:
        return match.group()
    copular = any(
        interpretation[1].split(":", 1)[0] == "być"
        for _, _, interpretation in MORFEUSZ.analyse(match["governor"])
    )
    inflector = cardinal if copular else genitive_cardinal
    return f"{match['prefix']}{inflector(int(match['value']))} procent"
