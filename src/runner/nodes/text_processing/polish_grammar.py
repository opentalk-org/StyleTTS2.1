from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from runner.nodes.text_processing.polish_inflections import (
    cardinal, dative_cardinal, feminine_accusative_ordinal, feminine_genitive_ordinal,
    feminine_nominative_ordinal, genitive_cardinal, genitive_ordinal,
    locative_cardinal, locative_ordinal, ordinal, plural_nominative_ordinal,
)
from runner.nodes.text_processing.polish_scales import expand_governed_scale_coordination
from runner.nodes.text_processing.polish_syntax import DURATION_LEMMAS, MORFEUSZ, NOMINAL_DURATION_GOVERNORS, NumberSyntax, has_unsafe_number_syntax, infer_number_syntax
from runner.nodes.text_processing.polish_valency import GENITIVE_VERB_LEMMAS

WORD = r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+"
NUMBER_NOUN_RE = re.compile(
    rf"(?P<prefix>\b(?:{WORD}\s+){{0,8}})(?P<value>\d+(?: \d{{3}})*)\s+(?P<noun>{WORD})\b", re.IGNORECASE,
)
MODIFIED_NUMBER_NOUN_RE = re.compile(
    rf"(?P<prefix>\b(?:{WORD}\s+){{0,8}})(?P<value>\d+(?: \d{{3}})*)\s+(?P<modifier>{WORD})\s+(?P<noun>{WORD})\b",
    re.IGNORECASE,
)
class NumberRole(StrEnum):
    CARDINAL = "cardinal"
    ORDINAL = "ordinal"
@dataclass(frozen=True)
class Morphology:
    lemma: str
    number: str
    case: str
    gender: str
@dataclass(frozen=True)
class Governor:
    lemma: str
    requires_genitive: bool
    identifier: bool
ORDINAL_LEMMAS = frozenset({
    "armia", "artykuł", "batalion", "brygada", "dywizja", "edycja", "forum", "godzina",
    "lekcja", "miejsce", "minuta", "odcinek", "pozycja", "poziom", "pułk", "rocznica",
    "rozdział", "rok", "strona", "stopień", "tydzień", "ustęp", "wersja", "wiek",
    "występ",
})
GENITIVE_GOVERNORS = frozenset({
    "bez", "blisko", "dla", "do", "granicach", "ilość", "ilości", "koło", "od", "około", "okolicach", "poniżej",
    "powyżej", "rzędu", "spośród", "sprzed", "według", "wobec", "wokoło", "wokół", "zamiast", "ciągu", "długości", "głębokości", "kontekście", "objętości", "odległości", "ostatnich", "powierzchni", "szerokości", "wartości", "wysokości", "władzach",
}) | GENITIVE_VERB_LEMMAS
LOCATIVE_GOVERNORS = frozenset({"na", "o", "po", "przy", "w"})
QUANTITY_MODIFIERS = frozenset({"blisko", "chyba", "kolejnych", "maksymalnie", "następnych", "nieco", "niespełna", "niemal", "około", "ponad", "prawie", "powiedzmy", "przeszło", "tam", "tylko", "zaledwie"})
IDENTIFIER_GOVERNOR_LEMMAS = frozenset({"numer"})
GENITIVE_NOUN_GOVERNOR_LEMMAS = frozenset({"autor", "budżet", "cykl", "granica", "kwestia", "kwota", "liczba", "lista", "obsługa", "perspektywa", "plan", "poziom", "promień", "prędkość", "próg", "przeciąg", "przewaga", "różnica", "rynek", "termin", "tło", "udział", "zbiór"})

def expand_morphological_numbers(text: str) -> str:
    if re.search(r"\d", text) is None:
        return text
    syntax = infer_number_syntax(text, ORDINAL_LEMMAS, _quantity_shape)
    modified = [
        match for match in MODIFIED_NUMBER_NOUN_RE.finditer(text)
        if (number_syntax := syntax.get(match.start("value")))
        and number_syntax.head_offset == match.start("noun")
        and match.start("modifier") in number_syntax.modifier_offsets
    ]
    direct = [
        match for match in NUMBER_NOUN_RE.finditer(text)
        if not any(match.start() < item.end() and item.start() < match.end() for item in modified)
    ]
    parts = []
    cursor = 0
    for match in sorted((*modified, *direct), key=lambda item: item.start()):
        parts.extend((text[cursor:match.start()], _expand_span(match, syntax)))
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts)


def expand_number_roles(text: str) -> str:
    text = expand_governed_scale_coordination(re.sub(rf"\b(?P<governor>blisko|do|od|około|poniżej|powyżej),(?P<space>\s+)(?P<value>\d+(?: \d{{3}})*) (?P<noun>{WORD})\b", lambda match: f"{match['governor']},{match['space']}{genitive_cardinal(int(match['value'].replace(' ', '')))} {match['noun']}", text, flags=re.IGNORECASE), _requires_genitive)
    text = re.sub(rf"\b(?P<governor>{WORD}) m[.]in[.] (?P<value>\d+(?: \d{{3}})*) (?P<noun>{WORD})\b", lambda match: f"{match['governor']} m.in. {genitive_cardinal(int(match['value'].replace(' ', '')))} {match['noun']}" if any(item[2][2].startswith("ger:") for item in MORFEUSZ.analyse(match['governor'])) else match.group(), text, flags=re.IGNORECASE)
    text = re.sub(rf"\b(?P<governor>bez|blisko|dla|do|od|około|poniżej|powyżej)(?P<first>,?\s+)(?P<filler>nie wiem|powiedzmy|załóżmy)(?P<second>,?\s+)(?P<value>\d+(?: \d{{3}})*) (?P<noun>{WORD})\b", lambda match: f"{match['governor']}{match['first']}{match['filler']}{match['second']}{genitive_cardinal(int(match['value'].replace(' ', '')))} {match['noun']}", re.sub(r"\b(?P<prefix>do (?:minus )?)(?P<value>\d+) (?P<noun>potęgi)\b", lambda match: f"{match['prefix']}{feminine_genitive_ordinal(int(match['value']))} {match['noun']}", text, flags=re.IGNORECASE), flags=re.IGNORECASE)
    text = re.sub(rf"\bo mocy (?P<value>\d+(?: \d{{3}})*) (?P<unit>{WORD})\b", lambda match: f"o mocy {genitive_cardinal(int(match['value'].replace(' ', '')))} {match['unit']}", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b(?P<modifier>(?:około|zaledwie) )?(?P<value>\d+(?: \d{{3}})*) (?P<unit>{WORD}) (?P<comparator>poniżej|powyżej|dalej)\b", lambda match: f"{match['modifier'] or ''}{(genitive_cardinal if (match['modifier'] or '').lower() == 'około ' else cardinal)(int(match['value'].replace(' ', '')))} {match['unit']} {match['comparator']}", text, flags=re.IGNORECASE)
    text = re.sub(rf"(?P<prefix>\b(?:{WORD}\s+){{0,8}}?)(?P<operator>plus minus) (?P<value>\d+(?: \d{{3}})*)\b", lambda match: f"{match['prefix']}{match['operator']} {(locative_cardinal if _governor(match['prefix']).lemma in LOCATIVE_GOVERNORS else genitive_cardinal if _requires_genitive(match['prefix']) else cardinal)(int(match['value'].replace(' ', '')))}", text, flags=re.IGNORECASE)
    text = re.sub(rf"(?P<prefix>\b(?:{WORD}\s+){{0,8}}?)(?P<operator>minus) (?P<value>\d+) (?P<unit>{WORD})\b", lambda match: f"{match['prefix']}{match['operator']} {(locative_cardinal if _governor(match['prefix']).lemma in LOCATIVE_GOVERNORS else genitive_cardinal if _requires_genitive(match['prefix']) else cardinal)(int(match['value']))} {match['unit']}", text, flags=re.IGNORECASE)
    text = re.sub(rf"(?P<prefix>\b(?:{WORD}\s+){{0,8}}?)(?P<value>\d+) (?P<unit>km|kilometrów|metrów) (?P<link>od|na)\b", lambda match: f"{match['prefix']}{(genitive_cardinal if _governor(match['prefix']).requires_genitive else cardinal)(int(match['value']))} {match['unit']} {match['link']}", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b(?P<factor>{WORD}) razy (?P<distribution>po )?(?P<value>\d+) (?P<unit>{WORD})\b", lambda match: f"{match['factor']} razy {match['distribution'] or ''}{cardinal(int(match['value']))} {match['unit']}", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b(?P<prefix>między {WORD} a) (?P<day>[1-9]|[12][0-9]|3[01]) (?P<month>stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\b", lambda match: f"{match['prefix']} {locative_ordinal(int(match['day']))} {match['month']}", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?P<prefix>w uproszczeniu) (?P<value>\d+(?: \d{3})*)\b", lambda match: f"{match['prefix']} {cardinal(int(match['value'].replace(' ', '')))}", text, flags=re.IGNORECASE)
    text = re.sub(r"\bw wieku (?P<values>\d+ lat(?:\s*(?:,|albo|i|lub|oraz)\s*\d+ lat)+)", _age_list_words, text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?P<prefix>na (?:koncie|rachunku)) (?P<value>\d+) (?P<unit>złotych|dolarów|euro|funtów)\b", lambda match: f"{match['prefix']} {cardinal(int(match['value']))} {match['unit']}", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?P<prefix>w wieku) (?P<modifier>(?:mniej więcej|niespełna|około|ponad|prawie|tam|zaledwie) )?(?P<value>\d+) lat\b", lambda match: f"{match['prefix']} {match['modifier'] or ''}{genitive_cardinal(int(match['value']))} lat", text, flags=re.IGNORECASE)
    text = re.sub(rf"(?P<prefix>\b(?:{WORD}\s+){{0,8}}?)(?P<modifier>prawie|czyli) (?P<value>\d+) (?P<unit>lat|dni|godzin|minut|sekund|tygodni|miesięcy)\b", lambda match: f"{match['prefix']}{match['modifier']} {(genitive_cardinal if match['modifier'].lower() == 'prawie' and _requires_genitive(match['prefix']) else cardinal)(int(match['value']))} {match['unit']}", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?P<prefix>w )?wieku lat (?P<value>\d+)\b", lambda match: f"{match['prefix'] or ''}wieku lat {genitive_cardinal(int(match['value']))}", text, flags=re.IGNORECASE)
    return re.sub(r"\b(?P<label>top) (?P<value>\d+)\b", lambda match: f"{match['label']} {cardinal(int(match['value']))}", text, flags=re.IGNORECASE)
def _age_list_words(match: re.Match[str]) -> str:
    values = re.sub(r"\d+(?= lat\b)", lambda item: genitive_cardinal(int(item.group())), match["values"])
    return f"{match.group()[0]} wieku {values}"

def nominative_ordinal_for_noun(value: int, noun: str) -> str:
    analyses = {
        morphology
        for _, _, interpretation in MORFEUSZ.analyse(noun)
        for morphology in _parse_substantive(interpretation)
        if morphology.case == "nom"
    }
    shapes = {(analysis.number, analysis.gender) for analysis in analyses}
    assert len(shapes) == 1, f"ambiguous Polish nominative noun: {noun}"
    number, gender = shapes.pop()
    if number == "pl":
        return plural_nominative_ordinal(value, gender == "m1")
    morphology = Morphology(lemma=noun, number=number, case="nom", gender=gender)
    return _ordinal_words(value, morphology)


def _expand_span(match: re.Match[str], syntax_by_offset: dict[int, NumberSyntax]) -> str:
    prefix = match["prefix"] or ""
    governor = _governor(prefix)
    value = int(match["value"].replace(" ", ""))
    syntax = syntax_by_offset.get(match.start("value"))
    morphology = _select_morphology(match["noun"], governor, value, syntax)
    if morphology is None:
        return match.group()
    modifier = match.groupdict().get("modifier")
    plural_modifier = modifier and any(interpretation[2].startswith("adj:pl:") for _, _, interpretation in MORFEUSZ.analyse(modifier))
    role = NumberRole.ORDINAL if morphology.lemma in ORDINAL_LEMMAS and (morphology.number == "sg" or syntax and syntax.ordinal_identifier) and not plural_modifier and governor.lemma != "przez" and not (syntax and (syntax.cardinal_subject or syntax.case == "acc" and morphology.lemma in DURATION_LEMMAS)) else NumberRole.CARDINAL
    words = _ordinal_words(value, morphology) if role is NumberRole.ORDINAL else _cardinal_words(value, morphology, governor, prefix, syntax)
    return f"{prefix}{words} {f'{modifier} ' if modifier else ''}{match['noun']}"

def _governor(prefix: str) -> Governor:
    words = prefix.strip().lower().split()
    if words[-2:] in (["mniej", "więcej"], ["co", "najmniej"]):
        words = words[:-2]
    while (len(words) > 1 and words[-1] in QUANTITY_MODIFIERS
           and (words[-1] not in GENITIVE_GOVERNORS
                or words[-2] in GENITIVE_GOVERNORS | LOCATIVE_GOVERNORS | {"przez", "za"})):
        words.pop()
    if not words:
        return Governor("", False, False)
    surface = words[-1]
    recognized_lemmas = GENITIVE_VERB_LEMMAS | GENITIVE_NOUN_GOVERNOR_LEMMAS | IDENTIFIER_GOVERNOR_LEMMAS
    recognized = next(((lemma, analysis[2][2]) for analysis in MORFEUSZ.analyse(surface)
                       if (lemma := analysis[2][1].split(":", 1)[0]) in recognized_lemmas), None)
    lemma = recognized[0] if recognized else surface
    fixed_participation = lemma == "udział" and len(words) > 1 and any(
        analysis[2][1].split(":", 1)[0] in {"brać", "wziąć"} for analysis in MORFEUSZ.analyse(words[-2])
    )
    interpretations = [item for _, _, item in MORFEUSZ.analyse(surface)]
    adjectival_genitive = (not any(item[2].startswith(("adv:", "pact:", "ppas:")) for item in interpretations)
                           and any(item[2].startswith(("adj:pl:gen", "adj:sg:gen")) for item in interpretations))
    requires_genitive = lemma in GENITIVE_GOVERNORS or adjectival_genitive or bool(
        recognized and lemma in GENITIVE_NOUN_GOVERNOR_LEMMAS and not fixed_participation
    )
    return Governor(lemma, requires_genitive, lemma in IDENTIFIER_GOVERNOR_LEMMAS or surface == "nr")

def _requires_genitive(prefix: str) -> bool:
    governor = _governor(prefix)
    negated = re.search(r"\b(?:ani|nie)\s+\w+\s+$", prefix, re.IGNORECASE) is not None
    preposition = any(analysis[2][2].startswith("prep:") for analysis in MORFEUSZ.analyse(governor.lemma))
    copular = any(interpretation[1] == "być" for _, _, interpretation in MORFEUSZ.analyse(governor.lemma))
    return governor.requires_genitive or negated and not preposition and not copular

def _select_morphology(word: str, governor: Governor, value: int, syntax: NumberSyntax | None) -> Morphology | None:
    if governor.lemma in {"z", "ze"}:
        return None
    if syntax and syntax.case == "dat" and value >= 1_000:
        return None
    interpretations = [interpretation for _, _, interpretation in MORFEUSZ.analyse(word)]
    if any(interpretation[2].startswith(("conj", "prep")) for interpretation in interpretations):
        return None
    analyses = {
        morphology
        for interpretation in interpretations
        for morphology in _parse_substantive(interpretation)
    } | ({Morphology(word.lower(), "sg", "acc", "m3")} if word.lower() in {"dolar", "funt"} and value == 1 else set())
    substantive_lemmas = [interpretation[1].split(":", 1)[0] for interpretation in interpretations
                          if interpretation[2].startswith("subst:")]
    if word.islower() and substantive_lemmas and all(lemma[0].isupper() for lemma in substantive_lemmas):
        analyses = {Morphology(analysis.lemma, analysis.number, analysis.case, "m3") for analysis in analyses}
    if syntax and syntax.gender:
        gender_prefix = {"fem": "f", "masc": "m", "neut": "n"}[syntax.gender]
        analyses = {analysis for analysis in analyses if analysis.gender.startswith(gender_prefix)}
    if syntax and syntax.animacy:
        personal = {analysis for analysis in analyses if analysis.gender == "m1"}
        if syntax.animacy == "hum" and personal and analyses - personal: return None
        analyses = personal or analyses if syntax.animacy == "hum" else analyses - personal or analyses
    syntax_case = "acc" if governor.lemma == "przez" else syntax.case if syntax else None
    scale_genitive = syntax_case == "loc" and value >= 1_000 and value % 1_000 == 0
    noun_case = "gen" if scale_genitive or syntax_case in {"acc", "nom"} and _quantity_shape(value) == frozenset({("pl", "gen")}) else syntax_case
    expected_case = "gen" if governor.requires_genitive and not (syntax_case == "nom" and governor.lemma not in GENITIVE_GOVERNORS | GENITIVE_NOUN_GOVERNOR_LEMMAS or syntax_case == "acc" and (governor.lemma not in GENITIVE_GOVERNORS | GENITIVE_NOUN_GOVERNOR_LEMMAS or any(analysis.lemma in {"dolar", "euro", "funt", "jen", "złoty"} for analysis in analyses))) else "loc" if governor.lemma in LOCATIVE_GOVERNORS and syntax_case != "acc" and any(analysis.case == "loc" for analysis in analyses) else noun_case
    expected_case = None if governor.lemma == "po" and any(analysis.lemma == "raz" for analysis in analyses) else expected_case
    matching = {analysis for analysis in analyses if analysis.case == expected_case} if expected_case else analyses
    all_personal = analyses and {analysis.gender for analysis in analyses} == {"m1"} and (syntax is None or syntax.animacy != "inan")
    human_accusative = {analysis for analysis in analyses
                        if all_personal and expected_case == "acc" and analysis.case == "gen" and analysis.number == "pl"}
    matching |= human_accusative
    if expected_case and not matching:
        return None
    supported = {analysis for analysis in matching if analysis.case in {"acc", "dat", "gen", "loc", "nom"}}
    ordinal_singular = {analysis for analysis in supported if analysis.lemma in ORDINAL_LEMMAS and analysis.number == "sg"}
    quantity_shape = (frozenset({("sg" if value == 1 else "pl", expected_case)})
                      if expected_case in {"dat", "gen", "loc"} else _quantity_shape(value))
    compatible = {analysis for analysis in supported if (analysis.number, analysis.case) in quantity_shape}
    if not ordinal_singular and not compatible and not any(analysis.lemma in ORDINAL_LEMMAS for analysis in supported):
        return None
    candidates = compatible if compatible and syntax_case in {None, "acc", "nom"} else ordinal_singular or compatible or supported
    features = {(analysis.number, analysis.case, analysis.gender) for analysis in candidates}
    if len(features) > 1 and {(analysis.number, analysis.gender) for analysis in candidates} == {("pl", "m1")}:
        return next(iter(candidates))
    if len(features) > 1 and len({(analysis.number, analysis.case) for analysis in candidates}) == 1:
        nonpersonal = {analysis for analysis in candidates if analysis.gender != "m1"}
        if len(nonpersonal) == 1 or {analysis.gender for analysis in candidates} <= {"m2", "m3"}:
            return next(iter(nonpersonal))
        return None
    syncretic_cases = {analysis.case for analysis in candidates}
    syncretic_shapes = {(analysis.number, analysis.gender) for analysis in candidates}
    if syncretic_cases <= {"acc", "nom"} and len(syncretic_shapes) == 1:
        return next((analysis for analysis in candidates if analysis.case == "nom"), next(iter(candidates)))
    return next(iter(candidates)) if len(features) == 1 else None

def _quantity_shape(value: int) -> frozenset[tuple[str, str]]:
    final = value % 100
    if value == 1:
        return frozenset({("sg", "acc"), ("sg", "nom")})
    if value % 10 in {2, 3, 4} and final not in {12, 13, 14}:
        return frozenset({("pl", "acc"), ("pl", "nom")})
    return frozenset({("pl", "gen")})


def _parse_substantive(interpretation: tuple[str, str, str, list[str], list[str]]) -> tuple[Morphology, ...]:
    lemma, tag = interpretation[1], interpretation[2]
    if "przest." in interpretation[4] or "nazwisko" in interpretation[3]:
        return ()
    parts = tag.split(":")
    if parts[0] != "subst" or len(parts) < 4:
        return ()
    return tuple(
        Morphology(lemma=lemma.split(":", 1)[0], number=number, case=case, gender=parts[3])
        for number in parts[1].split(".") for case in parts[2].split(".")
    )


def _ordinal_words(value: int, morphology: Morphology) -> str:
    if morphology.case == "acc" and morphology.gender == "f":
        return feminine_accusative_ordinal(value)
    if morphology.case == "gen":
        return feminine_genitive_ordinal(value) if morphology.gender == "f" else genitive_ordinal(value)
    if morphology.case == "loc":
        return _feminine_locative_ordinal(value) if morphology.gender == "f" else _neuter_locative_ordinal(value) if morphology.gender == "n" else locative_ordinal(value)
    if morphology.case == "nom":
        return feminine_nominative_ordinal(value) if morphology.gender == "f" else _neuter_nominative_ordinal(value) if morphology.gender == "n" else ordinal(value)
    return ordinal(value)

def _cardinal_words(value: int, morphology: Morphology, governor: Governor, prefix: str, syntax: NumberSyntax | None) -> str:
    negated_governor = _requires_genitive(prefix) and not (syntax and (syntax.case == "nom" or syntax.case == "acc" and governor.lemma not in GENITIVE_GOVERNORS | GENITIVE_NOUN_GOVERNOR_LEMMAS))
    if governor.identifier or governor.lemma == "blisko" and syntax and syntax.case == "acc" or re.search(r"\b(?:limit(?:\s+\w+){0,6}|te\s+około)\s+$", prefix, re.IGNORECASE):
        return feminine_cardinal(value) if morphology.gender == "f" and re.search(r"\bte\s+około\s+$", prefix, re.IGNORECASE) else cardinal(value)
    if governor.lemma == "przez" and morphology.lemma in ORDINAL_LEMMAS:
        return feminine_cardinal(value) if morphology.gender == "f" else cardinal(value)
    if morphology.lemma == "raz" and (not governor.requires_genitive or syntax and syntax.case == "acc"): return cardinal(value)
    if syntax and syntax.case == "acc" and morphology.lemma in DURATION_LEMMAS and governor.lemma in GENITIVE_NOUN_GOVERNOR_LEMMAS - NOMINAL_DURATION_GOVERNORS: return cardinal(value)
    if morphology.case == "gen" and morphology.lemma in {"tysiąc", "milion", "miliard"}:
        return genitive_cardinal(value) if syntax and syntax.case == "gen" or _requires_genitive(prefix) and (not (syntax and syntax.case == "acc") or governor.lemma in GENITIVE_GOVERNORS | GENITIVE_NOUN_GOVERNOR_LEMMAS) else cardinal(value)
    if morphology.case == "gen" and re.search(r"\b(?:na|przez|w|za)\s+około\s+$", prefix, re.IGNORECASE):
        return cardinal(value)
    if morphology.case in {"dat", "loc"} or syntax and syntax.case in {"dat", "loc"}:
        return ("jednej" if morphology.gender == "f" else "jednemu") if value == 1 and (syntax and syntax.case or morphology.case) == "dat" else dative_cardinal(value) if (syntax and syntax.case or morphology.case) == "dat" else "jednej" if value == 1 and morphology.gender == "f" else locative_cardinal(value)
    if syntax and syntax.case == "gen" or governor.requires_genitive and not (syntax and (syntax.case == "nom" and governor.lemma not in GENITIVE_GOVERNORS | GENITIVE_NOUN_GOVERNOR_LEMMAS or syntax.case == "acc" and governor.lemma not in GENITIVE_GOVERNORS | GENITIVE_NOUN_GOVERNOR_LEMMAS)) or negated_governor:
        return "jednej" if value == 1 and morphology.gender == "f" else genitive_cardinal(value)
    if value < 1_000 and morphology.gender == "m1" and morphology.number == "pl":
        return genitive_cardinal(value)
    if morphology.gender == "f" and syntax and syntax.case == "acc": return "jedną" if value == 1 else feminine_cardinal(value)
    if morphology.gender == "f" and morphology.case in {"acc", "nom"}:
        return feminine_cardinal(value)
    if syntax and syntax.case == "acc" and syntax.animacy != "hum":
        return cardinal(value)
    return cardinal(value)

def feminine_cardinal(value: int) -> str:
    words = cardinal(value).split()
    if words[-1] == "jeden":
        words[-1] = "jedna"
    elif words[-1] == "dwa":
        words[-1] = "dwie"
    return " ".join(words)


def _feminine_locative_ordinal(value: int) -> str:
    return feminine_genitive_ordinal(value)


def _neuter_nominative_ordinal(value: int) -> str:
    return _neuter_ordinal(value)


def _neuter_locative_ordinal(value: int) -> str:
    return locative_ordinal(value)


def _neuter_ordinal(value: int) -> str:
    words = ordinal(value).split()
    return " ".join(f"{word[:-1]}e" if word.endswith("y") else f"{word}e" if word.endswith("i") else word for word in words)
