from collections.abc import Sequence
from typing import Any


def coordinated_surface_case(token: Any, genitive_governed: bool, negated: bool, previous_case: str | None) -> str | None:
    """Recover clause case when a coordinated quantity noun is surface-genitive."""
    noun = token.head
    if noun.dep_ != "conj":
        return None
    if token.i >= 2 and token.nbor(-1).lower_ == "nie" and token.nbor(-2).lower_ in {"a", "ale"}:
        return "acc"
    omitted_measure = any(part.text.isdigit() and part.head.morph.get("Number") == ["Plur"] and part.head.morph.get("Case") == ["Gen"]
                          for part in token.doc[:token.i])
    if (previous_case and omitted_measure and noun.morph.get("Number") == ["Sing"] and noun.morph.get("Case") == ["Gen"]
            and noun.head.dep_ in {"nmod", "nmod:arg"} and any(part.pos_ == "CCONJ" for part in token.doc[noun.head.i + 1:token.i])):
        return "gen" if genitive_governed or negated else previous_case
    if genitive_governed or negated:
        return None
    appositive = (noun.head.lemma_.lower() == noun.lemma_.lower() or any(part.lemma_.lower() == noun.lemma_.lower() and part.head == noun.head for part in token.doc[:noun.i])) and any(
        part.is_punct for part in token.doc[noun.head.i + 1:token.i]
    )
    parsed_case = token.morph.get("Case")
    return parsed_case[0].lower() if appositive and parsed_case in (["Acc"], ["Nom"]) else None


def coordinated_quantity_case(
    token: Any,
    prior_syntax: Sequence[Any],
    genitive_governed: bool,
    negated: bool,
    earlier_genitive: bool,
) -> str | None:
    """Resolve a coordinated quantity from its shared dependency owner."""

    noun = token.head
    previous_case = "gen" if earlier_genitive else prior_syntax[-1].case if prior_syntax else None
    surface_case = coordinated_surface_case(token, genitive_governed, negated, previous_case)
    repeated_case = next((item.case for item in prior_syntax if token.dep_ == "flat"
                          and noun.text.isdigit() and item.head_offset == noun.idx), None)
    if repeated_case:
        return repeated_case
    if surface_case or noun.dep_ != "conj":
        return surface_case
    if noun.head.dep_ == "obj" and not (genitive_governed or negated):
        return "acc"
    shared_case = next((item.case for item in prior_syntax if item.head_offset == noun.head.idx), None)
    if shared_case:
        return shared_case
    morphology = noun.morph.get("Case")
    same_case = noun.head.morph.get("Case") == morphology or any(
        child.dep_ in {"nmod", "nmod:arg"} and child.morph.get("Case") == morphology
        for child in noun.head.children
    )
    return morphology[0].lower() if morphology and same_case else None


def is_elliptical_quantity(token: Any) -> bool:
    clause_start = max((part.i + 1 for part in token.doc[max(0, token.i - 8):token.i] if part.is_punct), default=max(0, token.i - 8))
    clause = token.doc[clause_start:token.i]
    return any(part.lower_ == "a" and part.i + 1 < token.i and part.nbor().lower_ in {"ten", "ta", "to", "ci", "te"} for part in clause) and not any(part.pos_ in {"AUX", "VERB"} for part in clause)


def is_copular_quantity_subject(token: Any) -> bool:
    """Recognize a quantity predicated through Polish copular forms or discourse `to`."""

    noun = token.head
    return any(
        child.dep_ == "cop" or child.pos_ == "AUX" and child.lemma_.lower() == "być"
        for owner in (noun, noun.head)
        for child in owner.children
    )
