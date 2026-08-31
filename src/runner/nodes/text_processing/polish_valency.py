"""Lexical government and grammatical evidence for quantity case."""

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, Iterable
import morfeusz2


MORFEUSZ = morfeusz2.Morfeusz()
ACCUSATIVE_QUANTITY_VERB_LEMMAS = frozenset({"dostać", "kosztować", "kupić", "kupować", "płacić", "wydać", "wydawać", "wypłacać", "wypłacić", "zapłacić"})
DURATION_OBJECT_VERB_LEMMAS = frozenset({"dać", "dawać", "dostać", "mieć", "odbyć", "odsiedzieć", "otrzymać", "przeżyć", "przesiedzieć", "spędzać", "spędzić", "stracić", "wytrzymać", "zyskać", "zmarnować"})
DISTANCE_ADJUNCT_VERB_LEMMAS = frozenset({"biec", "chodzić", "iść", "jechać", "lecieć", "płynąć", "wracać", "wędrować", "wychodzić"})
NEGATION_PROPAGATING_VERB_LEMMAS = frozenset({"chcieć", "mieć", "móc", "musieć", "próbować", "trzeba", "zamierzać"})
GENITIVE_VERB_LEMMAS = frozenset({
    "braknąć", "brakować", "dokonać", "dokonywać", "doliczać", "doliczyć", "domagać", "dotyczyć", "dożyć", "dożywać", "oczekiwać", "potrzeba", "potrzebować", "pragnąć",
    "sięgać", "słuchać", "szukać", "unikać", "używać", "wymagać",
})
GENITIVE_PERCENT_NOUN_LEMMAS = frozenset({"emisja", "wypłata"})
CURRENCY_LEMMAS = frozenset({"dolar", "euro", "funt", "jen", "złoty"})
DISTANCE_LEMMAS = frozenset({"kilometr", "km", "metr"})
NEGATED_DISTANCE_OBJECT_VERB_LEMMAS = frozenset({"pokonać", "przejechać"})
QUANTITY_PARTICLE_LEMMAS = frozenset({"aż", "jeszcze", "maksymalnie", "nawet", "ponad", "tylko"})
CITED_WORK_LEMMAS = frozenset({"album", "film", "gra", "książka", "powieść", "serial", "spektakl", "utwór"})


class QuantityRole(StrEnum):
    OBJECT = "object"
    ADJUNCT = "adjunct"
    NOMINAL_COMPLEMENT = "nominal_complement"
    SUBJECT = "subject"
    UNKNOWN = "unknown"


class CaseConfidence(IntEnum):
    """Reliability of evidence used to choose a spoken numeral form."""

    PARSER = 10
    DEPENDENCY = 40
    AGREEMENT = 60
    GOVERNMENT = 80


@dataclass(frozen=True)
class CaseEvidence:
    case: str
    confidence: CaseConfidence
    source: str


def resolve_quantity_case(evidence: Iterable[CaseEvidence]) -> str:
    """Choose the strongest case, abstaining when equally strong evidence conflicts."""

    candidates = tuple(evidence)
    assert candidates, "quantity case requires at least one evidence item"
    confidence = max(candidate.confidence for candidate in candidates)
    strongest = {candidate.case for candidate in candidates if candidate.confidence == confidence}
    return strongest.pop() if len(strongest) == 1 else "ambiguous"


def accusative_quantity_governor(token: Any) -> Any | None:
    ancestors = (token.head, *token.ancestors)
    candidates = (*ancestors, *(child for ancestor in ancestors for child in ancestor.children if child.dep_ == "xcomp"))
    return next((part for part in candidates if part.lemma_.lower().split()[0] in ACCUSATIVE_QUANTITY_VERB_LEMMAS or any(
        analysis[2][1].split(":", 1)[0].lower() in ACCUSATIVE_QUANTITY_VERB_LEMMAS for analysis in MORFEUSZ.analyse(part.text)
    )), None)


def has_accusative_quantity_government(token: Any, duration_lemmas: frozenset[str]) -> bool:
    """Recognize marked or limiter-owned quantity phrases governed in the accusative."""

    inherited_owner = token.head.head if token.head.head.lemma_.lower() in {"maksimum", "minimum"} else None
    owners = (token.head, token.head.head) if token.head.dep_ == "conj" else (token.head,) if inherited_owner is None else (token.head, inherited_owner)
    markers = {
        child.lemma_.lower()
        for owner in owners
        for child in owner.children
        if child.dep_ == "case"
    }
    previous = token.nbor(-1).lemma_.lower() if token.i else ""
    lemma = token.head.lemma_.split(":", 1)[0].lower()
    transparent_limiter = (inherited_owner is not None and inherited_owner.head.lemma_.lower() not in GENITIVE_VERB_LEMMAS
                           and not any(child.dep_ == "advmod:neg" for child in inherited_owner.head.children))
    return ("przez" in markers or lemma in duration_lemmas and transparent_limiter
            or lemma in CURRENCY_LEMMAS and ({"o", "po"} & markers or previous == "o")
            or lemma in duration_lemmas and token.head.morph.get("Case") != ["Ins"]
            and ("za" in markers or previous == "za"))


def has_locative_quantity_government(token: Any) -> bool:
    """Recognize fixed reflexive predicates whose `na` complement is locative."""

    markers = {child.lemma_.lower() for child in token.head.children if child.dep_ == "case"}
    governor = next((part for part in token.ancestors if part.pos_ == "VERB"), None)
    return ("na" in markers and governor is not None and (governor.lemma_.lower() in {"kończyć", "skończyć"} or any(analysis[2][1].split(":", 1)[0].lower() in {"kończyć", "skończyć"} for analysis in MORFEUSZ.analyse(governor.text)))
            and any(child.dep_ == "expl:pv" for child in governor.children))


def is_accusative_distance_quantity(token: Any) -> bool:
    """Identify bare distance adjuncts of movement rather than governed complements."""

    lemma = token.head.lemma_.split(":", 1)[0].lower()
    has_marker = any(child.dep_ == "case" and child.pos_ == "ADP" for owner in (token, token.head) for child in owner.children)
    local_motion = any(part.pos_ == "VERB" and part.lemma_.lower().split()[0] in DISTANCE_ADJUNCT_VERB_LEMMAS for part in token.doc[token.head.i + 1:token.head.i + 4] if not part.is_punct)
    destination = (token.head.i + 1 < len(token.doc) and token.head.nbor().lemma_.lower() == "stąd"
                   or token.head.dep_ in {"nmod", "nmod:arg"}
                   and any(child.dep_ == "case" and child.lemma_.lower() in {"do", "od"} for child in token.head.head.children))
    return (lemma in DISTANCE_LEMMAS and (token.morph.get("Case") == ["Acc"] or local_motion) and not has_marker
            and (token.head.head.lemma_.lower() in DISTANCE_ADJUNCT_VERB_LEMMAS or destination or local_motion))


def is_negated_distance_object(token: Any) -> bool:
    governor = token.head.head
    return (token.head.lemma_.split(":", 1)[0].lower() in DISTANCE_LEMMAS
            and governor.lemma_.lower().split()[0] in NEGATED_DISTANCE_OBJECT_VERB_LEMMAS
            and any(child.dep_ == "advmod:neg" for child in governor.children))


def classify_quantity_role(token: Any) -> QuantityRole:
    """Classify the quantity phrase before deciding how its numeral is inflected."""

    noun = token.head
    if noun.dep_ == "obj":
        return QuantityRole.OBJECT
    if noun.dep_ in {"obl", "advmod"}:
        return QuantityRole.ADJUNCT
    if noun.dep_ in {"nmod", "nmod:arg"} and noun.head.pos_ in {"NOUN", "PROPN"}:
        return QuantityRole.NOMINAL_COMPLEMENT
    if noun.dep_.startswith("nsubj"):
        return QuantityRole.SUBJECT
    governor = noun.head.lemma_.split(":", 1)[0].lower()
    if governor in DURATION_OBJECT_VERB_LEMMAS:
        return QuantityRole.OBJECT
    if governor in DISTANCE_ADJUNCT_VERB_LEMMAS:
        return QuantityRole.ADJUNCT
    return QuantityRole.UNKNOWN


def quantity_particles(token: Any) -> frozenset[str]:
    """Return discourse particles attached to the number phrase, independent of word order."""

    owners = (token, token.head, token.head.head)
    attached = {
        child.lemma_.lower()
        for owner in owners
        for child in owner.children
        if child.lemma_.lower() in QUANTITY_PARTICLE_LEMMAS
        and not any(part.is_punct for part in token.doc[min(child.i, token.i):max(child.i, token.i)])
    }
    attached.update(owner.lemma_.lower() for owner in owners if owner.lemma_.lower() in QUANTITY_PARTICLE_LEMMAS
                    and not any(part.is_punct for part in token.doc[min(owner.i, token.i):max(owner.i, token.i)]))
    if token.i and token.nbor(-1).lemma_.lower() in QUANTITY_PARTICLE_LEMMAS:
        attached.add(token.nbor(-1).lemma_.lower())
    return frozenset(attached)


def particle_governed_case(token: Any, particles: frozenset[str]) -> str | None:
    """Inherit case only from a marked nominal that structurally owns the quantity."""

    if not particles & {"aż", "nawet"}:
        return None
    governor = next((ancestor for ancestor in token.head.ancestors if ancestor.pos_ in {"NOUN", "PROPN"}
                     and any(child.dep_ == "case" for child in ancestor.children)), None)
    cases = governor.morph.get("Case") if governor is not None else []
    return cases[0].lower() if len(cases) == 1 else None


def is_cited_work_title_quantity(token: Any) -> bool:
    """Keep an unquoted numeric work title in its citation form."""

    noun = token.head
    work = noun.head
    return (noun.dep_ in {"nmod", "nmod:arg"} and work.lemma_.lower() in CITED_WORK_LEMMAS
            and noun.i > work.i and not any(child.dep_ == "case" for child in noun.children))
