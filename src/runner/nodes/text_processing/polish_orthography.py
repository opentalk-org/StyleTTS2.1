from __future__ import annotations

import re

from runner.nodes.text_processing.polish_temporal import is_continued_decade_clause


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
NON_SENTENCE_ABBREVIATIONS = ("itd.", "itp.", "m.in.", "np.", "ok.", "tj.", "tzw.", "ust.")
BIBLE_REFERENCE_RE = re.compile(r"\b(?P<label>Genesis|Rzymian|Koryntian|Galatów|Efezjan|Filipian|Kolosan|Tesaloniczan|Tymoteusza|Tytusa|Filemona|Hebrajczyków|Apokalipsy|(?:Mateusza|Marka|Łukasza|Jana)|Ewangelii (?:według )?(?:Mateusza|Marka|Łukasza|Jana)|(?:listu |księgi )(?:Rodzaju|Wyjścia|Kapłańskiej|Liczb|Powtórzonego Prawa|Jozuego|Sędziów|Rzymian|Koryntian|Galatów|Efezjan|Filipian|Kolosan|Tesaloniczan|Tymoteusza|Tytusa|Filemona|Hebrajczyków|Jakuba|Piotra|Judy|Apokalipsy)|Psalm(?:u|ie)?|wersecie [A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+|(?:Mateusza|Marka|Łukasza|Jana|Rodzaju|Wyjścia|Kapłańskiej|Liczb|Powtórzonego Prawa|Jozuego|Sędziów|Rzymian|Koryntian|Galatów|Efezjan|Filipian|Kolosan|Tesaloniczan|Tymoteusza|Tytusa|Filemona|Hebrajczyków|Jakuba|Piotra|Judy|Apokalipsy)(?= \d+[,.:]\d+\b[^.!?]{0,100}\b(?:werset\w*|Bibli\w*))) (?P<chapter>\d+)[,.:](?P<verse>\d+)\b", re.IGNORECASE)
SPORTS_RECORD_RE = re.compile(r"\b(?P<label>bilans(?:u|em|ie)?) (?P<wins>\d+)[,:](?P<losses>\d+)\b", re.IGNORECASE)


def restore_numeric_sentence_capitals(source: str, normalized: str) -> str:
    """Capitalize words expanded from digits that began a sentence."""
    source_starts = _sentence_starts(source)
    normalized_starts = _sentence_starts(normalized)
    if len(source_starts) != len(normalized_starts):
        return source
    characters = list(normalized)
    for source_start, normalized_start in zip(source_starts, normalized_starts, strict=True):
        if source[source_start:source_start + 1].isdigit() and normalized[normalized_start:normalized_start + 1].islower():
            characters[normalized_start] = characters[normalized_start].upper()
    return "".join(characters)


def _sentence_starts(text: str) -> tuple[int, ...]:
    return (0, *(match.end() for match in SENTENCE_BOUNDARY_RE.finditer(text)
                 if not text[:match.start()].lower().endswith(NON_SENTENCE_ABBREVIATIONS)
                 and not _is_numeric_nonboundary(text, match)))


def _is_numeric_nonboundary(text: str, match: re.Match[str]) -> bool:
    decade = re.search(r"\b(?:latach|lat) (?P<value>[2-9]0)[.]$", text[:match.start()], re.IGNORECASE)
    return (text[match.start() - 2:match.start() - 1].isdigit()
            and (text[match.end():match.end() + 1].islower() or re.match(r"[A-ZĄĆĘŁŃÓŚŹŻ]{2,}\b", text[match.end():]) is not None
                 or re.match(r"[XVI]+ (?:wieku|stulecia)\b", text[match.end():]) is not None)
            or decade is not None and is_continued_decade_clause(text, decade.start("value")))
