from __future__ import annotations

import re

from runner.nodes.text_processing.polish_inflections import (
    cardinal,
    genitive_cardinal,
)
from runner.nodes.text_processing.polish_syntax import MORFEUSZ


MIXED_RANGE_RE = re.compile(r"\bod (?P<first>[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+) do \d+\b", re.IGNORECASE)


def has_invalid_mixed_range(text: str) -> bool:
    return any(
        not any(analysis[2][2].startswith("num:") for analysis in MORFEUSZ.analyse(match["first"]))
        for match in MIXED_RANGE_RE.finditer(text)
    )


def range_words(match: re.Match[str]) -> str:
    prefix = match.string[:match.start()]
    start = int(match["start"])
    first = genitive_cardinal(start) if re.search(r"\bod\s*$", prefix, re.IGNORECASE) else cardinal(start)
    return f"{first} do {genitive_cardinal(int(match['end']))}"


def range_percent_words(match: re.Match[str]) -> str:
    start = int(match["start"])
    end = int(match["end"])
    assert start <= end, "descending percentage range must be rejected before expansion"
    first = genitive_cardinal(start) if match["prefix"] else cardinal(start)
    return f"{match['prefix'] or ''}{first} do {genitive_cardinal(end)} procent"


def age_range_words(match: re.Match[str]) -> str:
    start = genitive_cardinal(int(match["start"]))
    end = genitive_cardinal(int(match["end"]))
    return f"wieku od {start} do {end}"


def po_alternative_words(match: re.Match[str]) -> str:
    first = genitive_cardinal(int(match["first"]))
    second = genitive_cardinal(int(match["second"]))
    link = re.search(r"\b(czy|lub|albo)\b", match.group(), re.IGNORECASE)
    assert link is not None, "alternative numeral link missing"
    return f"po {first} {link.group()} {second}"
