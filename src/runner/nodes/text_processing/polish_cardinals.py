from __future__ import annotations

import re
from collections.abc import Callable

import morfeusz2
from num2words import num2words


COUNTED_NOUNS = (
    "lat|lata|rok|roku|miesiąc|miesiące|miesięcy|tydzień|tygodnie|tygodni|dni|dzień|"
    "godzina|godziny|godzin|minuta|minuty|minut|sekunda|sekundy|sekund|"
    "osoba|osoby|osób|ludzi|dzieci|kobiet|mężczyzn|pracowników|"
    "zł|złoty|złote|złotych|dolar|dolary|dolarów|euro|funt|funty|funtów|franków|"
    "tysiąc|tysiące|tysięcy|milion|miliony|milionów|miliard|miliardy|miliardów|"
    "mm|cm|m|km|mg|g|kg|gram|gramy|gramów|metr|metry|metrów|kilometr|kilometry|kilometrów|"
    "stopień|stopnie|stopni|procent|procenty|procentów|punkt|punkty|punktów|"
    "strona|strony|stron|odcinek|odcinki|odcinków|rozdział|rozdziały|rozdziałów|"
    "mecz|mecze|meczów|bramka|bramki|bramek|miejsce|miejsca|miejsc|"
    "razy|sztuka|sztuki|sztuk|egzemplarz|egzemplarze|egzemplarzy"
)
COUNTED_CARDINAL_RE = re.compile(rf"\b(?P<value>\d+)\s+(?P<noun>{COUNTED_NOUNS})\b", re.IGNORECASE)
ISOLATED_CARDINAL_RE = re.compile(r"^(?P<prefix>\s*)(?P<value>\d+)(?P<suffix>[.,;:!?]?\s*)$")
DOTTED_CARDINAL_RE = re.compile(r"(?<![\w.])(?P<value>\d+)[.](?!\d)")
MORPH_CARDINAL_RE = re.compile(r"\b(?P<value>\d+)\s+(?P<noun>[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)\b")
GROUPED_NOUN_RE = re.compile(r"\b(?P<value>\d{1,3}(?: \d{3})+)\s+(?P<noun>[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)\b")
SCALED_NOUN_RE = re.compile(r"\b\d+ (?:tysiąc|tysiące|tysięcy|milion|miliony|milionów|miliard|miliardy|miliardów) (?P<noun>[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)\b", re.IGNORECASE)
GOVERNED_CONTEXT_RE = re.compile(r"(?:od|do|około|z|roku|rzędu|długości|głębokości|mocy|objętości|odległości|powierzchni|szerokości|wartości|wysokości|w ciągu|w kontekście|po(?: około)?)(?:,\s*(?:nie wiem|powiedzmy|załóżmy),?)?\s+$", re.IGNORECASE)
GOVERNED_NOUN_RE = re.compile(r"\b(?P<governor>od|do|(?<!w )około|z|zamiast|dla|poniżej|powyżej|rzędu|długości|głębokości|mocy|objętości|odległości|powierzchni|szerokości|wartości|wysokości|w ciągu|w kontekście) (?P<value>\d+) (?P<noun>[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)\b", re.IGNORECASE)
MORFEUSZ = morfeusz2.Morfeusz()


def expand_counted_cardinals(text: str, genitive_cardinal: Callable[[int], str]) -> str:
    expanded = MORPH_CARDINAL_RE.sub(
        lambda match: _morphological_words(match, genitive_cardinal),
        text,
    )
    expanded = COUNTED_CARDINAL_RE.sub(_counted_words, expanded)
    expanded = ISOLATED_CARDINAL_RE.sub(_isolated_words, expanded)
    return expanded


def has_incompatible_cardinal_noun(text: str, ordinal_lemmas: frozenset[str]) -> bool:
    scaled_mismatch = any(
        tags and not any(_tag_has(tag, "pl", "gen") for tag in tags)
        for match in SCALED_NOUN_RE.finditer(text)
        if _is_unambiguously_substantive(match["noun"]) and (tags := _noun_tags(match["noun"]))
    )
    grouped_mismatch = any(
        not _grouped_noun_agrees(int(match["value"].replace(" ", "")), tags)
        for match in GROUPED_NOUN_RE.finditer(text)
        if _is_unambiguously_substantive(match["noun"]) and (tags := _noun_tags(match["noun"]))
    )
    incompatible_count = any(
        not _grouped_noun_agrees(int(match["value"]), tags)
        and not _has_supported_ordinal_case(text[:match.start()], match["noun"], tags, ordinal_lemmas)
        and not any(case in tag for tag in tags for case in (":dat", ":inst", ":loc"))
        and GOVERNED_CONTEXT_RE.search(text[:match.start()]) is None
        for match in MORPH_CARDINAL_RE.finditer(text)
        if _is_unambiguously_substantive(match["noun"]) and (tags := _noun_tags(match["noun"]))
    )
    governed_mismatch = any(
        tags and not _governed_noun_agrees(int(match["value"]), tags)
        for match in GOVERNED_NOUN_RE.finditer(text)
        if (tags := _noun_tags(match["noun"]))
        and (match["governor"].lower() in {"rzędu", "w ciągu"}
             or not _has_supported_ordinal_case(match["governor"], match["noun"], tags, ordinal_lemmas))
    )
    ambiguous_substantive = any(
        _has_ambiguous_substantive_gender(match["noun"])
        for match in MORPH_CARDINAL_RE.finditer(text)
    )
    return scaled_mismatch or grouped_mismatch or incompatible_count or governed_mismatch or ambiguous_substantive


def article_number_is_quantity(text: str, number_end: int) -> bool:
    following = re.match(r"\s+(?P<noun>[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)", text[number_end:])
    return following is not None and _only_plural_genitive(_noun_tags(following["noun"]))


def _grouped_noun_agrees(value: int, tags: tuple[str, ...]) -> bool:
    if _takes_plural_nominative(value):
        return any(_tag_has(tag, "pl", "nom") or _tag_has(tag, "pl", "acc") for tag in tags)
    return any(_tag_has(tag, "pl", "gen") for tag in tags)


def _governed_noun_agrees(value: int, tags: tuple[str, ...]) -> bool:
    number = "sg" if value == 1 else "pl"
    return any(_tag_has(tag, number, "gen") for tag in tags)


def _tag_has(tag: str, number: str, case: str) -> bool:
    parts = tag.split(":")
    return parts[0] == "subst" and number in parts[1].split(".") and case in parts[2].split(".")


def _counted_words(match: re.Match[str]) -> str:
    return f"{num2words(int(match['value']), lang='pl')} {match['noun']}"


def _morphological_words(
    match: re.Match[str],
    genitive_cardinal: Callable[[int], str],
) -> str:
    value = int(match["value"])
    tags = _noun_tags(match["noun"])
    if tags and all(":loc:" in tag for tag in tags):
        return f"{genitive_cardinal(value)} {match['noun']}"
    if value < 1000 and tags and all(tag.endswith(":m1") for tag in tags):
        return f"{genitive_cardinal(value)} {match['noun']}"
    if any(":pl:nom" in tag and tag.endswith(":f") for tag in tags):
        return f"{_feminine_cardinal(value)} {match['noun']}"
    if _takes_plural_nominative(value) and any(":pl:gen:" in tag for tag in tags):
        return match.group()
    return match.group()


def _noun_tags(word: str) -> tuple[str, ...]:
    interpretations = [interpretation for _, _, interpretation in MORFEUSZ.analyse(word)
                       if interpretation[2].startswith("subst:")]
    exact = [interpretation for interpretation in interpretations
             if interpretation[1].split(":", 1)[0].lower() == word.lower()]
    return tuple(interpretation[2] for interpretation in exact or interpretations)


def _has_ambiguous_substantive_gender(word: str) -> bool:
    tags = [interpretation[2] for _, _, interpretation in MORFEUSZ.analyse(word)]
    genders = {tag.split(":")[3] for tag in tags if tag.startswith("subst:")}
    return any(tag.startswith("adj:") for tag in tags) and {"f", "m1"} <= genders


def _is_ordinal_noun(word: str, ordinal_lemmas: frozenset[str]) -> bool:
    return any(
        interpretation[1].split(":", 1)[0].lower() in ordinal_lemmas
        for _, _, interpretation in MORFEUSZ.analyse(word)
    )


def _has_supported_ordinal_case(prefix: str, word: str, tags: tuple[str, ...], ordinal_lemmas: frozenset[str]) -> bool:
    if not _is_ordinal_noun(word, ordinal_lemmas):
        return False
    governor = prefix.lower().split()[-1:] or [""]
    cases = {"acc", "loc"} if governor[0] in {"na", "o"} else {"gen"} if governor[0] in {"bez", "dla", "do", "od", "z"} else {"loc"} if governor[0] in {"po", "przy", "w"} else {"acc", "nom"}
    return any(tag.startswith("subst:sg:") and cases & set(tag.split(":")[2].split(".")) for tag in tags)


def _is_unambiguously_substantive(word: str) -> bool:
    interpretations = MORFEUSZ.analyse(word)
    return not any(
        interpretation[2].startswith(("comp", "conj", "part", "pred", "prep"))
        for _, _, interpretation in interpretations
    )


def _only_plural_genitive(tags: tuple[str, ...]) -> bool:
    return bool(tags) and all(":pl:gen:" in tag for tag in tags)


def _takes_plural_nominative(value: int) -> bool:
    return value % 10 in (2, 3, 4) and value % 100 not in (12, 13, 14)


def _feminine_cardinal(value: int) -> str:
    words = str(num2words(value, lang="pl")).split()
    if words[-1] == "jeden":
        words[-1] = "jedna"
    elif words[-1] == "dwa":
        words[-1] = "dwie"
    return " ".join(words)


def _isolated_words(match: re.Match[str]) -> str:
    return f"{match['prefix']}{num2words(int(match['value']), lang='pl')}{match['suffix']}"
