from __future__ import annotations
from collections.abc import Callable, Collection
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import morfeusz2
import pl_core_news_sm
from runner.nodes.text_processing.polish_coordination import coordinated_quantity_case, is_copular_quantity_subject, is_elliptical_quantity
from runner.nodes.text_processing.polish_valency import accusative_quantity_governor, CaseConfidence, CaseEvidence, classify_quantity_role, DISTANCE_ADJUNCT_VERB_LEMMAS, DURATION_OBJECT_VERB_LEMMAS, GENITIVE_VERB_LEMMAS, has_accusative_quantity_government, has_locative_quantity_government, is_accusative_distance_quantity, is_cited_work_title_quantity, is_negated_distance_object, NEGATION_PROPAGATING_VERB_LEMMAS, particle_governed_case, quantity_particles, QuantityRole, resolve_quantity_case
MORFEUSZ = morfeusz2.Morfeusz(); PARSER = pl_core_news_sm.load(disable=["ner"])
DURATION_LEMMAS = frozenset({"dekada", "dzień", "godzina", "miesiąc", "minuta", "noc", "rok", "sekunda", "tydzień"}); MEASUREMENT_LEMMAS = frozenset({"długość", "głębokość", "ilość", "objętość", "powierzchnia", "prędkość", "szerokość", "wartość", "wysokość"})
NOMINAL_DURATION_GOVERNORS = frozenset({"czas", "efekt", "granica", "kara", "okres", "perspektywa", "podstawa", "różnica", "staż", "wyrok"})
@lru_cache(maxsize=8)
def _parse(text: str) -> Any: return PARSER(text)
@dataclass(frozen=True)
class NumberSyntax:
    case: str
    gender: str
    animacy: str
    head_offset: int; modifier_offsets: tuple[int, ...]
    cardinal_subject: bool; ordinal_identifier: bool
def has_unsafe_number_syntax(text: str) -> bool:
    document = _parse(text)
    bare_adjectival_percent = any(
            token.text == "%" and (
            any(child.dep_.startswith("det") and child.morph.get("Number") == ["Sing"] for child in token.children) or token.i >= 2 and token.nbor(-2).pos_ == "VERB" and token.nbor(-2).morph.get("Tense") == ["Past"] and token.nbor(-2).morph.get("Number") == ["Plur"] or token.i + 1 < len(document) and (token.nbor().pos_ == "NOUN" and token.nbor().morph.get("Case") != ["Gen"] and not any(child.dep_ == "case" for child in token.children) or any(analysis[2][2].startswith(("subst:sg:acc", "subst:sg:dat", "subst:sg:inst", "subst:sg:loc")) for analysis in MORFEUSZ.analyse(token.nbor().text))) or token.i >= 2 and token.nbor(-2).pos_ == "ADJ"
            or token.i + 2 < len(document) and token.nbor().pos_ == "DET" and token.nbor(2).pos_ == "NOUN" and (token.nbor().morph.get("Case") == ["Acc"] and token.nbor(2).morph.get("Case") == ["Acc"] or token.nbor().morph.get("Number") and token.nbor().morph.get("Number") != token.nbor(2).morph.get("Number")) or token.i + 1 < len(document) and token.nbor().dep_ == "amod" and token.nbor().head == token or (token.i >= 2 and token.nbor(-2).pos_ in {"AUX", "NOUN", "PRON"}
                and any(part.pos_ == "ADJ" for part in document[token.i + 1:token.i + 3])) or any(part.pos_ in {"AUX", "ADJ"} and part.morph.get("Number") == ["Sing"] and part.morph.get("Gender") in (["Fem"], ["Masc"]) for part in document[token.i + 1:token.i + 4] if not part.is_punct) or (not any(child.dep_ == "case" for child in token.children) and any(predicate.pos_ == "VERB" and not any(part.is_punct or part.pos_ == "CCONJ" for part in document[token.i + 1:predicate.i]) and (predicate.morph.get("Number") != ["Sing"] or predicate.morph.get("Tense") == ["Past"] and predicate.morph.get("Gender") != ["Neut"]) for predicate in document[token.i + 1:token.i + 5]))
        )
        for token in document
    )
    mismatched_number_modifier = any(
        token.text.isdigit()
        and (
            token.morph.get("Case")
            and any(child.dep_ in {"amod", "det"}
                    and (token.morph.get("NumType") == ["Ord"] or child.morph.get("Number") == ["Sing"])
                    and child.morph.get("Case")
                    and (child.morph.get("Case") != token.morph.get("Case") or int(token.text) >= 5 and child.morph.get("Number") == ["Sing"])
                    for child in token.children)
            or (token.i and token.nbor(-1).pos_ == "ADJ" and token.nbor(-1).head == token.head and token.nbor(-1).morph.get("Gender") and token.head.morph.get("Gender") and token.nbor(-1).morph.get("Gender") != token.head.morph.get("Gender")) or any(child.dep_ == "case" and child.lemma_.lower() == "przez" for child in token.head.children) and any(modifier.pos_ == "ADJ" and modifier.head == token.head and modifier.morph.get("Case") == ["Gen"] for modifier in document[max(0, token.i - 3):token.i]) and token.head.morph.get("Animacy") != ["Hum"] or token.head.lemma_.lower() == "rok" and token.head.text.lower() == "rok" and token.head.head.morph.get("Number") == ["Plur"] or token.i and len(token.text) == 4 and token.nbor(-1).text.lower() == "rok" and (token.head != token.nbor(-1) or any(child != token and child.pos_ in {"ADJ", "DET"} for child in token.nbor(-1).children)) or token.i and len(token.text) in {3, 4} and token.nbor(-1).pos_ == "NOUN" and token.nbor(-1).morph.get("Case") == ["Nom"] and token.i + 1 < len(document) and token.nbor().lemma_.lower() == "rok" and token.nbor().morph.get("Case") == ["Nom"] or sum(child.text.isdigit() for child in token.head.children) > 1 or _has_mismatched_age_head(token)
            or token.i >= 2 and token.nbor(-2).lemma_.lower() == "na" and token.nbor(-1).head == token and token.nbor(-1).morph.get("Animacy") == ["Hum"] and token.head.morph.get("Gender") in (["Fem"], ["Neut"])
            or token.i >= 2 and token.nbor(-2).lemma_.lower() in {"na", "o", "przez", "za"} and token.nbor(-1).pos_ in {"ADJ", "DET"} and "Acc" not in token.nbor(-1).morph.get("Case") and token.head.morph.get("Case") == ["Gen"] and token.head.morph.get("Animacy") != ["Hum"]
            or token.head.lemma_.split(":", 1)[0].lower() in {"dolar", "euro", "funt", "jen", "złoty"} and (any(child.dep_ == "case" and child.lemma_.lower() == "w" for child in token.head.children) or any(child.dep_ == "cop" and child.lemma_.lower() == "być" for child in token.head.children) and any(child.pos_ == "PRON" and child.morph.get("Case") == ["Gen"] for child in token.head.children))
            or token.head.text.lower() == "roku" and (token.head.head.lemma_.lower() == "mieć" and not any(child.dep_ == "case" for child in token.head.children) or token.i and token.nbor(-1).lemma_.lower() == "za")
        )
        for token in document
    )
    bare_adverbial_percent = any(
        token.text == "%"
        and token.head.pos_ == "VERB" and not any(child.dep_ == "case" for child in token.children)
        and token.i + 1 < len(document)
        and any(item[2][2].startswith("adv") for item in MORFEUSZ.analyse(token.nbor().text))
        for token in document
    )
    malformed_root_infinitive = any(
        token.pos_ == "ADJ" and token.morph.get("Voice") == ["Pass"] and (any(child.dep_ == "obj" and child.morph.get("Case") == ["Acc"] for child in token.children) or token.head.pos_ == "NOUN" and token.morph.get("Gender") and token.head.morph.get("Gender") and token.morph.get("Gender") != token.head.morph.get("Gender") and any(part.text.isdigit() for part in token.head.subtree)) or token.pos_ == "VERB" and token.morph.get("VerbForm") == ["Inf"]
        and any(child.dep_ == "nsubj" for child in token.children)
        and any(child.dep_ == "mark" and child.lemma_.lower() == "że" for child in token.children)
        for token in document
    )
    locative_scale_mismatch = any(
        token.text.isdigit()
        and int(token.text) >= 1_000
        and int(token.text) % 1_000 == 0
        and token.morph.get("Case") == ["Loc"]
        and token.head.morph.get("Case") == ["Loc"] and token.head.lemma_.lower() != "rok"
        for token in document
    )
    malformed_copular_years = any(
        token.text.isdigit()
        and token.head.lemma_.lower() == "rok"
        and token.head.text.lower() == "lat"
        and token.head.head.lemma_.lower() == "być"
        and any(child.dep_ == "case" and child.lemma_.lower() == "po" for child in token.head.children)
        for token in document
    )
    parallel_case_conflict = any(
        first.head.pos_ == "NOUN" and first.head.lemma_ == second.head.lemma_
        and not (second.head.dep_ == "conj" and second.head.head == first.head)
        and not _has_case_marker(first) and not _has_case_marker(second)
        and first.head.morph.get("Case") == second.head.morph.get("Case")
        and (first.morph.get("Case") and second.morph.get("Case") and first.morph.get("Case") != second.morph.get("Case") or next((part.lemma_ for part in document[first.head.i + 1:] if not part.is_punct), None) == next((part.lemma_ for part in document[second.head.i + 1:] if not part.is_punct), None))
        and any(part.text == "," for part in document[first.i + 1:second.i])
        and all(part.pos_ not in {"AUX", "VERB"} for part in document[first.i + 1:second.i])
        for first in document if first.text.isdigit()
        for second in document[first.i + 1:] if second.text.isdigit()
    )
    unknown_nominal_head = any(
        token.text.isdigit() and token.head.pos_ in {"NOUN", "PROPN"} and all(analysis[2][2] == "ign" for analysis in MORFEUSZ.analyse(token.head.text))
        for token in document
    )
    return bare_adjectival_percent or mismatched_number_modifier or any(token.text == "%" and token.i >= 2 and token.nbor(-2).pos_ in {"NOUN", "PROPN"} and any(analysis[2][2].startswith("subst:pl:gen") for analysis in MORFEUSZ.analyse(token.nbor(-2).text)) and not any(child.dep_ == "case" for child in token.nbor(-2).children) and not any(part.is_punct for part in document[token.i - 2:token.i]) for token in document) or any(token.text == "%" and token.i + 1 < len(document) and any(analysis[2][2].startswith("subst:sg:nom") for analysis in MORFEUSZ.analyse(token.nbor().text)) and not any(analysis[2][2].startswith("subst:sg:gen") for analysis in MORFEUSZ.analyse(token.nbor().text)) for token in document) or any(token.text.isdigit() and token.head.pos_ == "ADJ" and token.head.morph.get("Case") == ["Gen"] and token.head.morph.get("Number") == ["Plur"] and token.head.head.pos_ == "NOUN" and token.head.head.morph.get("Case") == ["Ins"] for token in document) or any(token.text.isdigit() and token.head.lemma_.lower() == "osoba" and (any(ancestor.pos_ == "NOUN" and ancestor.morph.get("Animacy") == ["Hum"] and ancestor.morph.get("Number") == ["Plur"] and not any(part.is_punct for part in document[ancestor.i + 1:token.i]) for ancestor in token.head.ancestors) or any(sibling != token.head and sibling.pos_ == "NOUN" and sibling.morph.get("Case") == ["Gen"] and sibling.morph.get("Number") == ["Plur"] and sibling.i < token.i and not any(part.is_punct for part in document[sibling.i + 1:token.i]) for sibling in token.head.head.children)) for token in document) or any(token.text.isdigit() and token.dep_ == "obl" and token.head.pos_ == "ADJ" and any(child.dep_ == "case" for child in token.head.children) and token.i + 1 < len(document) and any(character.isdigit() for character in token.nbor().text) for token in document) or any(token.text.isdigit() and token.head.lemma_.lower() == "rok" and token.head.text.lower() == "roku" and (token.head.dep_ == "obj" or token.head.dep_ == "conj" and token.head.head.pos_ == "VERB") and not _has_direct_case_marker(token) and not (_has_genitive_governor(token) or _has_negated_verb_ancestor(token)) for token in document) or any(token.pos_ == "NOUN" and any(child.dep_ == "case" and child.lemma_.lower() in {"bez", "dla", "do", "od", "spod", "sprzed", "wobec", "zza"} for child in token.children) and not any(analysis[2][2].startswith("subst:") and ":gen" in analysis[2][2] for analysis in MORFEUSZ.analyse(token.text)) for token in document) or any(token.text.isdigit() and token.head.dep_ == "conj" and token.head.head.pos_ in {"NOUN", "PROPN"} and token.head.morph.get("Case") and token.head.head.morph.get("Case") and token.head.morph.get("Case") != token.head.head.morph.get("Case") for token in document) or any(token.text.isdigit() and token.i and token.nbor(-1) != token.head and token.nbor(-1).lemma_ == token.head.lemma_ for token in document) or any(any(character.isdigit() for character in token.text) and token.head.pos_ == "NOUN" and any(candidate.pos_ == "NOUN" and candidate.lemma_.lower() == token.head.lemma_.lower() for candidate in document[token.head.i + 1:token.head.i + 4]) for token in document) or any(first.text.isdigit() and first.head.pos_ in {"NOUN", "PROPN"} and second.text.isdigit() and second.head.pos_ in {"NOUN", "PROPN"} and first.head.i + 1 == second.i and second.head.i == second.i + 1 for first in document for second in document[first.i + 1:first.i + 4]) or not all(not any(character.isdigit() for character in token.text) or len(token.text) in {3, 4} and (token.i and token.nbor(-1).lower_ in {"rok", "roku"} or token.i + 1 < len(document) and token.nbor().lower_ in {"rok", "roku"}) for token in document) and any((bare_adverbial_percent, locative_scale_mismatch, any(token.text.isdigit() and token.pos_ == "X" and any(child.dep_ == "flat" for child in token.children) for token in document),
                malformed_copular_years, malformed_root_infinitive, mismatched_number_modifier, any(token.text == "%" and any(ancestor.lemma_.lower() in {"spadać", "spaść", "wzrastać", "wzrosnąć", "zwiększać", "zwiększyć", "zmniejszać", "zmniejszyć"} for ancestor in token.ancestors) and not any(child.dep_ == "case" for owner in (token, *token.ancestors) for child in owner.children) for token in document), any(token.text.isdigit() and (token.dep_ == "nsubj" and token.head.morph.get("Number") == ["Plur"] and any(child.lemma_.lower() == "być" and child.morph.get("Number") == ["Sing"] for child in token.head.children) or token.i + 2 < len(document) and token.nbor(2).pos_ == "VERB" and token.nbor(2).morph.get("Number") == ["Plur"] and any(part.lemma_.lower() == "być" and part.morph.get("Number") == ["Sing"] for part in document[max(0, token.i - 3):token.i])) for token in document),
                parallel_case_conflict, unknown_nominal_head, any(token.text.isdigit() and token.head.lemma_.lower() in DURATION_LEMMAS and token.head.i + 1 < len(document) and token.head.nbor().pos_ == "NOUN" and token.head.nbor().morph.get("Case") == ["Nom"] and all(part.is_punct for part in document[token.head.i + 2:]) for token in document), any(token.text.isdigit() and int(token.text) >= 5 and token.head.dep_ == "nsubj" and token.head.head.lemma_.lower() == "być" and token.head.head.morph.get("Gender") not in ([], ["Neut"]) for token in document), any(token.text.isdigit() and (any(child != token and child.dep_ == "det:numgov" for child in token.head.children) or token.i and (token.nbor(-1).pos_ == "DET" and token.nbor(-1).morph.get("NumType") == ["Card"] or token.nbor(-1).lower_ in {"kilkadziesiąt", "kilkadziesięć", "kilkudziesiąt", "kilkudziesięć"} or any(analysis[2][2].startswith("num:") and analysis[2][1].split(":", 1)[0].lower() not in {"miliard", "milion", "tysiąc"} for analysis in MORFEUSZ.analyse(token.nbor(-1).text)))) for token in document), any(token.text.isdigit() and token.head.lemma_.lower() in DURATION_LEMMAS and token.head.head.lemma_.lower() == "znajdować" and any(child.dep_ == "expl:pv" for child in token.head.head.children) and any(child.dep_ == "case" and child.lemma_.lower() == "w" for child in token.head.children) for token in document), any(token.text.isdigit() and int(token.text) <= 31 and (token.i == 0 or token.nbor(-1).text in ".!?") and token.i + 1 < len(document) and token.nbor().lemma_.lower() == "mieć" and token.nbor().morph.get("VerbForm") == ["Fin"] for token in document)))
def infer_number_syntax(
    text: str,
    ordinal_lemmas: Collection[str],
    quantity_shape: Callable[[int], frozenset[tuple[str, str]]],
) -> dict[int, NumberSyntax]:
    syntax = {}
    for token in _parse(text):
        if not token.text.isdigit():
            continue
        case_source = token.head
        if case_source.lemma_.lower() not in ordinal_lemmas and token.i + 1 < len(token.doc):
            case_source = token.nbor()
        ordinal_noun = case_source.lemma_.lower() in ordinal_lemmas and case_source.morph.get("Number") == ["Sing"]
        quantity_role = classify_quantity_role(token); particles = quantity_particles(token); particle_case = particle_governed_case(token, particles)
        features = case_source.morph.to_dict() if ordinal_noun else token.morph.to_dict(); features = {**features, "Case": "Gen"} if "Case" not in features and _is_nominal_genitive_complement(token) else features
        if "Case" not in features:
            continue
        evidence = [CaseEvidence(features["Case"].lower(), CaseConfidence.PARSER, "parser morphology")]
        modifier_cases = {
            value.lower()
            for child in token.children if child.dep_ in {"amod", "det"}
            for value in child.morph.get("Case")
        }
        if token.i and (token.nbor(-1).pos_ in {"ADJ", "DET"} or token.i >= 2 and token.nbor(-1).lemma_.lower() in {"chyba", "około", "ponad", "prawie", "zaledwie"} and token.nbor(-2).pos_ in {"ADJ", "DET"} or token.i >= 3 and token.nbor(-3).pos_ in {"ADJ", "DET"} and token.nbor(-2).lower_ == "na" and token.nbor(-1).lower_ == "przykład"):
            modifier_cases.update(
                value.lower()
                for child in token.doc[max(0, token.i - 3):token.i]
                if child.pos_ in {"ADJ", "DET"} and (child.head == token.head or token.i >= 2 and token.nbor(-1).lemma_.lower() in {"chyba", "około", "ponad", "prawie", "zaledwie"} and child == token.nbor(-2) and (child.head in {token, token.head} or any(marker.dep_ == "case" for owner in (child, token.head) for marker in owner.children)) or token.i >= 3 and child == token.nbor(-3) and token.nbor(-2).lower_ == "na" and token.nbor(-1).lower_ == "przykład")
                for value in child.morph.get("Case")
            )
        case_from_modifier = len(modifier_cases) == 1
        if case_from_modifier:
            modifier_case = modifier_cases.pop()
            evidence.append(CaseEvidence(modifier_case, CaseConfidence.AGREEMENT, "agreeing modifier"))
        if token.dep_ == "conj" and token.head.morph.get("Case"): evidence.append(CaseEvidence(token.head.morph.get("Case")[0].lower(), CaseConfidence.DEPENDENCY, "coordinated head"))
        if token.head.dep_ == "dep" and _has_case_marker(token): evidence.append(CaseEvidence("ambiguous", CaseConfidence.AGREEMENT, "unresolved prepositional attachment"))
        case = resolve_quantity_case(evidence)
        if token.pos_ == "NUM" and token.dep_ in {"dep", "nummod:gov", "obj"} and case == "gen" and not case_from_modifier:
            case = "ambiguous"
        if token.pos_ == "NUM" and token.dep_ == "nummod" and case == "gen" and not _has_direct_case_marker(token) and not case_from_modifier:
            case = "ambiguous"
        if quantity_role is QuantityRole.OBJECT:
            evidence.append(CaseEvidence("gen" if _has_negated_verb_ancestor(token) else "acc", CaseConfidence.DEPENDENCY, "verbal object"))
        parser_object_error = token.morph.get("NumType") == ["Ord"] and token.head.morph.get("Number") == ["Plur"]
        if parser_object_error and not _has_negated_verb_ancestor(token):
            evidence.append(CaseEvidence("acc", CaseConfidence.DEPENDENCY, "misparsed cardinal object"))
        if _has_genitive_governor(token) and not _has_direct_case_marker(token): evidence.append(CaseEvidence("gen", CaseConfidence.GOVERNMENT, "lexical government"))
        case = resolve_quantity_case(evidence)
        if (token.i and token.head.idx > token.idx and not parser_object_error
                and token.nbor(-1).pos_ == "NOUN"
                and token.nbor(-1) in token.head.ancestors
                and token.nbor(-1).morph.get("Case") in (["Dat"], ["Ins"], ["Loc"])
                and not any(child.dep_ == "case" for child in token.nbor(-1).children)):
            case = "gen"
        if token.i and not parser_object_error and (_is_verbal_noun_form(token.nbor(-1)) or token.i >= 2 and token.nbor(-1).lower_ in {"około", "ponad", "prawie", "zaledwie"} and _is_verbal_noun_form(token.nbor(-2)) or token.nbor(-1).pos_ == "NOUN" and token.nbor(-1) in token.head.ancestors and token.nbor(-1).head.pos_ != "ADP" and token.nbor(-1).morph.get("Case") in (["Acc"], ["Dat"], ["Ins"], ["Loc"]) and token.head.morph.get("Case") == ["Gen"] and not any(child.dep_ == "case" for child in token.nbor(-1).children) and not _has_case_marker(token)):
            case = "gen"
        if not case_from_modifier and _is_nominal_genitive_complement(token):
            case = "invalid" if token.i and token.nbor(-1).pos_ == "NOUN" and token.nbor(-1).morph.get("Case") == ["Acc"] and any(child.dep_ == "case" for child in token.nbor(-1).children) and token.head != token.nbor(-1) else "gen"
        postposed = token.doc[token.head.i + 1:token.head.i + 4]
        if any(part.morph.get("Clitic") == ["Yes"] for part in postposed) and any(part.pos_ == "VERB" for part in postposed) and not any(part.dep_ == "advmod:neg" for part in postposed) and not _is_nominal_genitive_complement(token) and not _has_case_marker(token): case = "acc"
        if is_copular_quantity_subject(token) and not _is_nominal_genitive_complement(token):
            case = "nom"
        if _is_elliptical_subject(token):
            case = "nom"
        if is_elliptical_quantity(token):
            case = "acc"
        if _is_measurement_predicate(token):
            case = "nom"
        if "maksymalnie" in particles and token.head.lemma_.lower() in DURATION_LEMMAS or token.head.lemma_.lower() in {"kilometr", "metr"} and (token.head.i + 1 < len(token.doc) and token.head.nbor().lemma_.lower() == "po" or any(child.dep_ == "case" and child.lemma_.lower() == "na" for child in token.head.children) and (token.head.head.lemma_.lower() in {"być", "pozostawać", "znajdować"} or any(analysis[2][1].split(":", 1)[0].lower() in {"być", "pozostawać", "znajdować"} for analysis in MORFEUSZ.analyse(token.head.head.text)))): case = "acc" if "maksymalnie" in particles or token.head.i + 1 < len(token.doc) and token.head.nbor().lemma_.lower() == "po" else "loc"
        if (token.i >= 2 and token.nbor(-2).lemma_.lower() == "mniej" and token.nbor(-1).lemma_.lower() == "dużo"
                and (token.i < 3 or token.nbor(-3).pos_ != "ADP")
                and not _has_case_marker(token) and not _has_genitive_governor(token)):
            case = "acc"
        if (quantity_role is QuantityRole.SUBJECT and not (token.i >= 2 and token.nbor(-1).lower_ in {"około", "ponad", "prawie", "zaledwie"} and _is_verbal_noun_form(token.nbor(-2))) or token.head.dep_ == "conj" and (any(child.dep_ == "aux:pass" for ancestor in token.head.ancestors for child in ancestor.children) or any(child.dep_ == "cop" for child in token.head.head.children) and any(part.is_punct for part in token.doc[token.head.head.i + 1:token.i]) or token.i and token.nbor(-1).is_punct and all(part.is_punct for part in token.doc[token.head.i + 1:]) and not _has_genitive_governor(token)) or _is_existential_subject(token) or _is_subject_after_fixed_adverbial(token) or _is_participation_subject(token)):
            case = "nom"
        instrumental_time = (token.nbor(-1) if token.i and token.nbor(-1).lemma_.lower() == "czas"
                             and token.nbor(-1).morph.get("Case") == ["Ins"] else None)
        if (instrumental_time and instrumental_time.morph.get("Number") == ["Plur"]
                and any(child.dep_ == "case" for child in instrumental_time.children)):
            case = "gen"
        elif (token.dep_ != "amod:flat" and (case_source.lemma_.lower() in DURATION_LEMMAS and quantity_role is QuantityRole.ADJUNCT or case_source.lemma_.lower() in {"kilometr", "metr"} and quantity_role is QuantityRole.ADJUNCT) and not case_from_modifier and not _has_direct_case_marker(token)
                and (instrumental_time is not None or not _is_nominal_genitive_complement(token) or token.head.i + 1 < len(token.doc) and token.head.nbor().lemma_.lower() == "do")
                and (not _has_genitive_governor(token) or token.morph.get("Case") == ["Acc"] and token.head.head.lemma_.lower() not in DURATION_OBJECT_VERB_LEMMAS | GENITIVE_VERB_LEMMAS or token.head.head.pos_ == "NOUN" and token.head.head.lemma_.lower() not in NOMINAL_DURATION_GOVERNORS and token.head.head.head.lemma_.lower() not in GENITIVE_VERB_LEMMAS)):
            parsed_case = token.morph.get("Case")
            case = parsed_case[0].lower() if parsed_case in (["Acc"], ["Nom"]) else "acc"
        coordinated_case = coordinated_quantity_case(token, tuple(syntax.values()), _has_genitive_governor(token), _has_negated_verb_ancestor(token), any(part.text.isdigit() and (_has_genitive_governor(part) or _has_negated_verb_ancestor(part)) for part in token.doc[:token.i]))
        precision_case = next(reversed(syntax.values())).case if syntax and token.i >= 2 and token.nbor(-1).lemma_.lower() == "dokładnie" and token.nbor(-2).lower_ == "a" else None
        if (coordinated_case or precision_case) and not case_from_modifier and not (token.head.dep_ == "conj" and (any(child.dep_ == "aux:pass" for ancestor in token.head.ancestors for child in ancestor.children) or any(child.dep_ == "cop" for child in token.head.head.children) and any(part.is_punct for part in token.doc[token.head.head.i + 1:token.i]) or token.i and token.nbor(-1).is_punct and all(part.is_punct for part in token.doc[token.head.i + 1:]) and not _has_genitive_governor(token) or any(ancestor.pos_ in {"AUX", "VERB"} and ancestor.morph.get("VerbForm") == ["Fin"] for ancestor in token.head.ancestors) and any(part.lower_ == "to" for part in token.doc[token.head.i + 1:token.head.i + 5]) and any(part.lemma_.lower() == "być" for part in token.doc[token.head.i + 1:token.head.i + 5]))) and not (token.morph.get("Case") == ["Acc"] and (_has_direct_case_marker(token) or case_source.lemma_.lower() in DURATION_LEMMAS and (token.head.dep_ == "conj" or token.head.i + 1 < len(token.doc) and token.head.nbor().lemma_.lower() in {"do", "po", "przed"} or any(part.lower_ == "za" for part in token.doc[max(0, token.i - 4):token.i])))):
            case = coordinated_case or precision_case
        nested_quantity_case = next((item.case for item in syntax.values() if token.head.dep_ in {"nmod", "nmod:arg"} and item.head_offset == token.head.head.idx), None) or next((item.case for item in syntax.values() if item.case == "gen" and case_source.lemma_.lower() in DURATION_LEMMAS and token.i and token.nbor(-1).lemma_.lower() in DURATION_LEMMAS), None)
        if nested_quantity_case: case = token.morph.get("Case")[0].lower() if token.head.head.lemma_.lower() == token.head.lemma_.lower() and token.morph.get("Case") in (["Nom"], ["Acc"]) else nested_quantity_case if not _is_nominal_genitive_complement(token) else case
        if token.i and any(analysis[2][2].startswith(("bedzie:", "fin:", "impt:", "praet:")) for analysis in MORFEUSZ.analyse(token.nbor(-1).text)) and not _has_genitive_governor(token) and not (case_source.lemma_.lower() in DURATION_LEMMAS and not (token.head.head.lemma_.lower() in DURATION_OBJECT_VERB_LEMMAS | GENITIVE_VERB_LEMMAS or any(analysis[2][1].split(":", 1)[0].lower() in DURATION_OBJECT_VERB_LEMMAS | GENITIVE_VERB_LEMMAS for analysis in MORFEUSZ.analyse(token.head.head.text)))): case = "gen" if _has_negated_verb_ancestor(token) or token.nbor(-1).lemma_.lower() == "być" and any(child.dep_ == "advmod:neg" for child in token.nbor(-1).children) else "acc"
        if any(ancestor.lemma_.lower() == "mieć" for ancestor in token.ancestors) and (not _is_nominal_genitive_complement(token) or token.head.i < token.head.head.i and _is_verbal_noun_form(token.head.head) or token.head.head.morph.get("Case") == ["Loc"] and any(child.dep_ == "case" for child in token.head.head.children)) and not (token.i and token.nbor(-1).morph.get("VerbForm") == ["Fin"] and not (token.head.i < token.head.head.i and _is_verbal_noun_form(token.head.head))): case = "gen" if _has_negated_verb_ancestor(token) else "acc"
        if not _is_nominal_genitive_complement(token) and (quantity_governor := accusative_quantity_governor(token)) is not None: case = "gen" if _has_negated_verb_ancestor(token) or any(child.dep_ == "advmod:neg" for child in quantity_governor.children) else "acc"
        if has_accusative_quantity_government(token, DURATION_LEMMAS): case = "acc"
        if has_locative_quantity_government(token) or token.head.lemma_.lower() == "pozycja" and any(child.dep_ == "case" and child.lemma_.lower() == "na" for child in token.head.children): case = "loc" if token.head.lemma_.lower() != "pozycja" or any(ancestor.lemma_.lower() in DISTANCE_ADJUNCT_VERB_LEMMAS | {"być", "plasować", "stać", "znajdować"} or any(analysis[2][1].split(":", 1)[0].lower() in {"być", "plasować", "stać", "znajdować"} for analysis in MORFEUSZ.analyse(ancestor.text)) for ancestor in token.ancestors) else "acc"
        if token.i and token.nbor(-1).is_punct and token.head.head.dep_ == "acl:relcl" and token.head.head.head.head.pos_ == "VERB": case = "gen" if any(child.dep_ == "advmod:neg" for child in token.head.head.head.head.children) else "acc"
        if case_from_modifier: internal_quantity_genitive = modifier_case == "gen" and quantity_shape(int(token.text)) == frozenset({("pl", "gen")}) and not (_has_case_marker(token) or _has_genitive_governor(token) or _has_negated_verb_ancestor(token) or _is_nominal_genitive_complement(token) or token.i >= 2 and token.nbor(-1).lemma_.lower() in {"chyba", "około", "ponad", "prawie", "zaledwie"} and any(marker.dep_ == "case" for owner in (token.nbor(-2), token.head) for marker in owner.children) or token.i >= 3 and token.nbor(-2).lower_ == "na" and token.nbor(-1).lower_ == "przykład"); case = "acc" if internal_quantity_genitive or modifier_case == "gen" and token.head.lemma_.lower() in DURATION_LEMMAS and token.head.morph.get("Case") == ["Gen"] and any(child.dep_ == "case" and child.lemma_.lower() == "na" for child in token.head.children) else modifier_case
        if token.i >= 2 and token.head.text.lower() == "lat" and len(token.nbor(-1).text) == 1 and token.nbor(-1).is_alpha and token.nbor(-2).is_punct: case = "nom"
        if "aż" in particles and particle_case: case = particle_case
        if token.morph.get("Case") == ["Nom"] and any(child.pos_ == "VERB" and child.morph.get("VerbForm") == ["Fin"] for child in token.head.children): case = "nom"
        if token.i and not case_from_modifier and token.head.dep_ in {"nmod", "nmod:arg"} and (token.nbor(-1).is_punct and token.head.head.morph.get("Case") in (["Nom"], ["Acc"]) or token.morph.get("Case") in (["Nom"], ["Acc"]) and quantity_role is QuantityRole.NOMINAL_COMPLEMENT and not _is_nominal_genitive_complement(token)): case = token.morph.get("Case")[0].lower() if token.morph.get("Case") in (["Nom"], ["Acc"]) else token.head.head.morph.get("Case")[0].lower()
        if token.i and not case_from_modifier and not _is_nominal_genitive_complement(token) and token.morph.get("Case") in (["Nom"], ["Acc"]) and (token.nbor(-1).is_punct or "nawet" in particles and not (_has_direct_case_marker(token) or _has_genitive_governor(token) or _has_negated_verb_ancestor(token)) or token.head.head.lemma_.lower() == token.head.lemma_.lower()): case = particle_case or token.morph.get("Case")[0].lower()
        if token.i >= 2 and token.nbor(-1).lemma_.lower() == "łącznie" and token.nbor(-2).lemma_.lower() == "czyli" and not _has_direct_case_marker(token): case = "nom"
        if token.i and ("ponad" in particles and (token.head.lemma_.lower() in DURATION_LEMMAS and not (_has_direct_case_marker(token) or _has_genitive_governor(token) or _has_negated_verb_ancestor(token)) and (not _is_nominal_genitive_complement(token) or token.head.i + 1 < len(token.doc) and token.head.nbor().lemma_.lower() == "po") or any(ancestor.morph.get("VerbForm") == ["Fin"] for ancestor in token.ancestors) and not _has_genitive_governor(token)) or "tylko" in particles and token.head.head.lemma_.lower() in {"dystans", "droga", "podróż", "trasa"} or "jeszcze" in particles and token.head.lemma_.lower() in DURATION_LEMMAS and token.morph.get("Case") == ["Acc"] and not (_has_direct_case_marker(token) or _has_genitive_governor(token) or _has_negated_verb_ancestor(token)) or token.head.lemma_.lower() == "punkt" and token.head.i + 1 < len(token.doc) and token.head.nbor().text.lower() in {"minus"} or token.head.lemma_.lower() == "raz" and token.head.i + 1 < len(token.doc) and token.head.nbor().text.lower() in {"za", "tak", "nie"} and not _has_direct_case_marker(token)) or token.head.lemma_.lower() in DURATION_LEMMAS and token.head.morph.get("Case") == ["Gen"] and any(child.dep_ == "case" and child.lemma_.lower() == "na" for child in token.head.children) or token.head.i + 1 < len(token.doc) and token.head.nbor().dep_ == "ROOT" and token.head.nbor().morph.get("Voice") == ["Pass"] or token.i >= 3 and token.nbor(-3).pos_ == "VERB" and token.nbor(-2).lemma_.lower() == "nie" and token.nbor(-1).lemma_.lower() == "wiedzieć" or token.i >= 2 and token.nbor(-2).lower_ == "na" and token.nbor(-1).lemma_.lower() == "przykład" and not (_has_direct_case_marker(token) or _is_nominal_genitive_complement(token)) or token.i == 0 and token.head.lemma_.lower() in DURATION_LEMMAS and token.head.i + 1 < len(token.doc) and token.head.nbor().is_punct: case = "acc"
        if token.i and token.head.morph.get("Case") == ["Gen"] and token.nbor(-1).pos_ == "NOUN" and token.nbor(-1) in token.head.ancestors and token.nbor(-1).morph.get("Case") == ["Gen"] and any(child.dep_ == "case" for child in token.nbor(-1).children): case = "gen"
        if token.i >= 2 and token.head.morph.get("Case") == ["Gen"] and token.nbor(-1).lemma_.lower() in {"około", "ponad", "prawie", "zaledwie"} and token.nbor(-2).pos_ == "NOUN" and token.nbor(-2).morph.get("Case") == ["Gen"] and any(child.dep_ == "case" for child in token.nbor(-2).children): case = "gen" if token.nbor(-2) in token.head.ancestors else "invalid"
        case = features["Case"].lower() if token.i and token.nbor(-1).pos_ in {"DET", "PRON"} and token.nbor(-1).morph.get("Gender") != token.head.morph.get("Gender") and any(child.dep_ == "case" and child.lemma_.lower() == "dla" for child in token.head.children) and (any(child.dep_ == "nsubj" and child != token.head for child in token.head.head.children) or any(part.pos_ in {"NOUN", "PRON"} and part.morph.get("Case") == ["Nom"] for part in token.doc[token.head.head.i + 1:token.head.head.i + 8])) or token.head.dep_ == "nmod" and token.head.head.morph.get("Case") == ["Ins"] and (token.morph.get("Case") == ["Acc"] and not any(child.dep_ == "cop" for child in token.head.head.children) or token.morph.get("Case") == ["Gen"] and any(child.dep_ == "cop" for child in token.head.head.children)) or token.dep_ == "obl:cmpr" and token.i and token.nbor(-1).lemma_.lower() == "niż" or ordinal_noun and token.head.dep_ == "conj" and token.head.head.lemma_.lower() == token.head.lemma_.lower() else "nom" if is_cited_work_title_quantity(token) or _is_measurement_predicate(token) else "gen" if (is_negated_distance_object(token) or quantity_role is QuantityRole.OBJECT and _has_negated_verb_ancestor(token) and not _has_direct_case_marker(token) or _has_genitive_governor(token) and not _has_direct_case_marker(token)) and not any(child.dep_ == "case" and child.lemma_.lower() == "temu" for child in token.head.children) else "acc" if is_accusative_distance_quantity(token) or token.head.lemma_.lower() in DURATION_LEMMAS and (any(child.dep_ == "case" and child.lemma_.lower() == "temu" for child in token.head.children) and not any(child.dep_ == "case" and child.lemma_.lower() != "temu" for child in token.head.children) or token.head.dep_ == "nmod" and token.head.head.dep_ == "nsubj" and token.head.head.head.lemma_.lower() == "być" or token.head.i + 1 < len(token.doc) and (token.head.nbor().lemma_.lower() == "wstecz" or token.head.i + 2 < len(token.doc) and token.head.nbor().lemma_.lower() == "do" and token.head.nbor(2).lemma_.lower() == "przód")) else case
        if any(child.dep_ in {"aux", "cop"} and child.lemma_.lower() == "być" for child in token.head.children): case = "nom"
        case = "invalid" if _has_incompatible_modifier(token, quantity_shape) else "dat" if case_source == token.head and case_source.morph.get("Case") == ["Dat"] else case
        noun_features = case_source.morph.to_dict()
        syntax[token.idx] = NumberSyntax(
            case=case,
            gender=noun_features.get("Gender", features.get("Gender", "")).lower(),
            animacy=noun_features.get("Animacy", features.get("Animacy", "")).lower(), cardinal_subject=token.head.dep_.startswith("nsubj") and any(part.morph.get("Number") == ["Sing"] and part.morph.get("Gender") == ["Neut"] for part in (token.head.head, *token.head.head.children)), ordinal_identifier=token.morph.get("NumType") == ["Ord"] or token.i and token.nbor(-1).pos_ == "NOUN" and token.head.lemma_.lower() in ordinal_lemmas - DURATION_LEMMAS and any(part.pos_ == "VERB" and part.morph.get("Number") == token.nbor(-1).morph.get("Number") for part in token.doc[token.head.i + 1:]),
            head_offset=token.head.idx,
            modifier_offsets=tuple(
                child.idx for child in token.head.children if child.dep_ in {"amod", "det"}
            ),
        )
    return syntax
def _has_case_marker(token: Any) -> bool:
    return any(child.dep_ == "case" and child.pos_ == "ADP" for ancestor in (token.head, token.head.head) for child in ancestor.children)
def _has_direct_case_marker(token: Any) -> bool:
    return any(child.dep_ == "case" and child.pos_ == "ADP" for owner in (token, token.head) for child in owner.children)
def _has_negated_verb_ancestor(token: Any) -> bool:
    governor = next((ancestor for ancestor in token.ancestors if ancestor.pos_ == "VERB"), token.head if token.head.pos_ == "VERB" else token.head.head)
    while governor.pos_ == "VERB":
        if any(child.dep_ == "advmod:neg" and child.i + 1 == governor.i for child in governor.children) or any(child.pos_ == "AUX" and any(grandchild.dep_ == "advmod:neg" and grandchild.i + 1 == child.i for grandchild in child.children) for child in governor.children): return True
        if governor.dep_ != "xcomp" or governor.head.lemma_.lower() not in NEGATION_PROPAGATING_VERB_LEMMAS: return False
        governor = governor.head
    return False
def _has_genitive_governor(token: Any) -> bool:
    governor = token.head.head
    if governor.lemma_.lower() in GENITIVE_VERB_LEMMAS or token.head.lemma_.lower() in DURATION_LEMMAS and (governor.lemma_.lower() in NOMINAL_DURATION_GOVERNORS or any(part.lemma_.lower() in NOMINAL_DURATION_GOVERNORS for part in token.doc[max(0, token.i - 3):token.i])) or governor.pos_ == "NOUN" and governor.head.lemma_.lower() in GENITIVE_VERB_LEMMAS: return True
    return any(interpretation[1].split(":", 1)[0].lower() in GENITIVE_VERB_LEMMAS for _, _, interpretation in MORFEUSZ.analyse(governor.text))
def _has_mismatched_age_head(token: Any) -> bool:
    if token.i + 1 >= len(token.doc) or token.nbor().text != "-" or not token.morph.get("Gender"): return False
    if token.head.pos_ in {"NOUN", "PROPN"} and token.head.morph.get("Number") and token.nbor().morph.get("Number") and token.head.morph.get("Number") != token.nbor().morph.get("Number"): return True
    candidates = [
        part for part in reversed(token.doc[max(0, token.i - 4):token.i])
        if part.pos_ in {"NOUN", "PROPN"}
    ]
    return bool(candidates and candidates[0].morph.get("Gender")
                and candidates[0].morph.get("Gender") != token.morph.get("Gender"))
def _is_verbal_noun_form(token: Any) -> bool:
    return any(interpretation[2].startswith("ger:") for _, _, interpretation in MORFEUSZ.analyse(token.text)) and not any(interpretation[2].startswith(("fin:", "inf:", "praet:")) for _, _, interpretation in MORFEUSZ.analyse(token.text))
def _is_existential_subject(token: Any) -> bool:
    noun = token.head
    return noun.dep_ == "nmod:arg" and noun.head.dep_ == "obl" and noun.head.head.lemma_.lower() == "być"
def _is_participation_subject(token: Any) -> bool:
    noun = token.head
    return noun.head.lemma_.lower() == "udział" and noun.head.head.lemma_.split(":", 1)[0].lower() in {"brać", "wziąć"}
def _is_subject_after_fixed_adverbial(token: Any) -> bool:
    noun = token.head
    return noun.dep_ == "obl" and token.morph.get("Case") == ["Nom"] and any(
        child.dep_ == "case" and any(part.dep_ == "fixed" and part.i < token.i for part in child.children)
        for child in noun.children
    )
def _is_elliptical_subject(token: Any) -> bool:
    clause_start = max(
        (part.i + 1 for part in token.doc[:token.i] if part.is_punct or part.pos_ in {"CCONJ", "SCONJ"}),
        default=0,
    )
    prefix = token.doc[clause_start:token.i]
    noun = token.head
    return clause_start > 0 and (
        noun.dep_ == "nmod:arg"
        and any(part.dep_ == "case" and part.morph.get("Case") and part.morph.get("Case") != ["Gen"]
                for part in noun.head.children)
        and not any(part.pos_ in {"AUX", "VERB"} for part in prefix)
    )
def _is_measurement_predicate(token: Any) -> bool:
    clause_start = max((part.i + 1 for part in token.doc[:token.i] if part.is_punct), default=0); prefix = token.doc[clause_start:token.i]
    return (not any(part.pos_ in {"AUX", "VERB"} for part in prefix) and (any(part.dep_ == "ROOT" and part.lemma_.lower() in MEASUREMENT_LEMMAS for part in prefix) or token.head.lemma_.lower() in {"kilometr", "metr"} and token.head.head.dep_ == "ROOT" and token.head.head.lemma_.lower() in {"jesień", "lato", "wiosna", "zima"} and any(child.dep_ == "det" for child in token.head.head.children)))
def _is_nominal_genitive_complement(token: Any) -> bool:
    noun = token.head
    parent = noun.head; role = classify_quantity_role(token)
    if any(child.dep_ == "case" and child.pos_ == "NOUN" and child.morph.get("Case") == ["Ins"] for child in noun.children): return True
    if noun.pos_ == "X" and noun.dep_ == "nmod" and parent.lemma_.lower() in MEASUREMENT_LEMMAS or parent.lemma_.lower() in DURATION_LEMMAS and noun.i > parent.i: return noun.pos_ == "X"
    if noun.lemma_.lower() in DURATION_LEMMAS and not any(child.dep_ == "case" for child in noun.children):
        return (parent.lemma_.lower() in NOMINAL_DURATION_GOVERNORS or _is_verbal_noun_form(parent)) and not (noun.i < parent.i and any(child.dep_ == "case" and child.lemma_.lower() == "do" for child in parent.children))
    if token.morph.get("NumType") == ["Ord"] and noun.morph.get("Number") == ["Plur"]: return False
    if role is not QuantityRole.NOMINAL_COMPLEMENT and noun.dep_ not in {"nsubj", "obj"}:
        return False
    verbal_noun = _is_verbal_noun_form(parent)
    return parent.lemma_.lower() in {"efekt", "seria"} and noun.dep_ == "nmod" and noun.morph.get("Case") == ["Gen"] or verbal_noun and not any(child.dep_ == "case" for child in noun.children) or (
        (role is QuantityRole.NOMINAL_COMPLEMENT or noun.dep_ == "nsubj")
        and noun.morph.get("Case") == ["Gen"]
        and not any(child.dep_ == "case" for child in noun.children)
        and parent.pos_ in {"NOUN", "PROPN"}
        and parent.dep_ not in {"advcl", "conj"}
        and parent.morph.get("Case") != ["Nom"]
        and (
            noun.dep_ != "nmod"
            or parent.morph.get("Case") in (["Dat"], ["Gen"], ["Ins"], ["Loc"])
            or parent.lemma_.lower() in MEASUREMENT_LEMMAS | {"efekt", "seria"}
            or parent.dep_ in {"nmod", "nmod:arg"} and not any(child.dep_ == "case" for child in parent.children)
        )
    )
def _has_incompatible_modifier(token: Any, quantity_shape: Callable[[int], frozenset[tuple[str, str]]]) -> bool:
    previous = token.nbor(-1) if token.i else None
    return (previous is not None and previous.pos_ in {"ADJ", "DET"}
        and quantity_shape(int(token.text)) == frozenset({("pl", "gen")})
        and (int(token.text) == 1000 and not any(analysis[2][2].startswith(("adj:sg:nom", "adj:sg:acc:m3")) for analysis in MORFEUSZ.analyse(previous.text)) or int(token.text) != 1000 and not any(analysis[2][2].startswith(("adj:pl:", "num:pl:")) for analysis in MORFEUSZ.analyse(previous.text))) or any(modifier != token and modifier.pos_ in {"ADJ", "DET"} and modifier.morph.get("Number") == ["Sing"] and token.head.morph.get("Number") == ["Plur"] for modifier in token.head.children))
