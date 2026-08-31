"""Inflection of productive Polish cardinal scale nouns."""

import re
from collections.abc import Callable
from typing import Any

from runner.nodes.text_processing.polish_inflections import cardinal, genitive_cardinal, instrumental_cardinal, locative_cardinal
from runner.nodes.text_processing.polish_syntax import MORFEUSZ

LOCATIVE_SCALE_CONTEXT_RE = re.compile(r"\b(?P<prefix>skończ\w*(?: się)?(?: tylko)? na) (?P<value>\d+) (?P<scale>tysięcy|milionów|miliardów)\b", re.IGNORECASE)
INSTRUMENTAL_SCALE_CONTEXT_RE = re.compile(r"\b(?P<prefix>przed) (?P<value>\d+) (?P<scale>tysięcy|milionów|miliardów)(?= lat\b)", re.IGNORECASE)
GOVERNED_SCALE_COORDINATION_RE = re.compile(r"(?P<prefix>\b(?:[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+\s+){1,4})(?P<first>\d+) (?P<first_scale>tysięcy|milionów|miliardów) (?P<link>i|oraz) (?P<second>\d+) (?P<second_scale>tysięcy|milionów|miliardów)\b", re.IGNORECASE)
EMBEDDED_CARDINAL_SCALE_RE = re.compile(r"\b(?P<major>milion|miliard) (?P<value>\d+) (?P<scale>tysięcy|milionów)\b", re.IGNORECASE)


def expand_locative_scale_contexts(text: str) -> str:
    text = INSTRUMENTAL_SCALE_CONTEXT_RE.sub(lambda match: f"{match['prefix']} {_scale_words(int(match['value']), match['scale'], 'inst', instrumental_cardinal, MORFEUSZ)}", text)
    return LOCATIVE_SCALE_CONTEXT_RE.sub(lambda match: f"{match['prefix']} {_scale_words(int(match['value']), match['scale'], 'loc', locative_cardinal, MORFEUSZ)}", text)


def expand_governed_scale_coordination(text: str, requires_genitive: Callable[[str], bool]) -> str:
    return GOVERNED_SCALE_COORDINATION_RE.sub(lambda match: f"{match['prefix']}{genitive_cardinal(int(match['first']))} {match['first_scale']} {match['link']} {genitive_cardinal(int(match['second']))} {match['second_scale']}" if requires_genitive(match['prefix']) else match.group(), text)


def expand_embedded_cardinal_scales(text: str) -> str:
    return EMBEDDED_CARDINAL_SCALE_RE.sub(lambda match: f"{match['major']} {cardinal(int(match['value']))} {match['scale']}", text)


def _scale_words(value: int, surface: str, case: str, inflector: Callable[[int], str], morfeusz: Any) -> str:
    lemmas = {
        interpretation[1].split(":", 1)[0]
        for _, _, interpretation in morfeusz.analyse(surface)
        if interpretation[2].startswith("subst:")
    }
    assert len(lemmas) == 1, f"ambiguous Polish scale noun: {surface}"
    number = "sg" if value == 1 else "pl"
    forms = {form for form, _, tag, _, _ in morfeusz.generate(lemmas.pop()) if tag.startswith(f"subst:{number}:{case}:")}
    assert len(forms) == 1, f"ambiguous Polish scale form: {surface}"
    return f"{inflector(value)} {forms.pop()}"
