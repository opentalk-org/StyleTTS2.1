from __future__ import annotations

import calendar
import re

from runner.nodes.text_processing.polish_inflections import (
    cardinal,
    feminine_accusative_ordinal,
    feminine_genitive_ordinal,
    feminine_nominative_ordinal,
    genitive_cardinal,
    genitive_ordinal,
    locative_ordinal,
    ordinal,
)
from runner.nodes.text_processing.polish_syntax import PARSER


CONTEXTUAL_DATE_RE = re.compile(r"\b(?P<prefix>(?P<about>(?:mów\w*|rozmawi\w*|wspomina\w*|myśl\w*|dyskut\w*) o)|(?P<today>(?:jest )?(?:dzisiaj|dziś)(?: jest)?,?)) (?P<day>[1-9]|[12][0-9]|3[01]) (?P<month>stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\b", re.IGNORECASE)
COMPLETION_MINUTES_RE = re.compile(r"\b(?P<verb>(?:s|za|u)kończ\w*) po (?P<value>\d+) min(?P<dot>[.]?)", re.IGNORECASE)
NEGATED_CONTRAST_DURATION_RE = re.compile(r"(?P<prefix>\bnie\b[^.!?]{0,80}\b(?:minut|godzin|dni|tygodni|miesięcy|lat)\b,?\s+tylko\s+)(?P<value>\d+)(?=\s+(?:minut|godzin|dni|tygodni|miesięcy|lat)\b)", re.IGNORECASE)


DATE_VALIDATION_RE = re.compile(r"\b(?P<day>\d{1,2}) (?P<month>stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)(?: (?P<year>\d{3,4}))?\b", re.IGNORECASE)
MONTH_NUMBERS = {name: number for number, name in enumerate("stycznia lutego marca kwietnia maja czerwca lipca sierpnia września października listopada grudnia".split(), 1)}


def has_invalid_date(text: str) -> bool:
    return any(
        int(match["day"]) < 1
        or int(match["day"]) > calendar.monthrange(int(match["year"] or 2000), MONTH_NUMBERS[match["month"].lower()])[1]
        for match in DATE_VALIDATION_RE.finditer(text)
    )


def expand_completion_minutes(text: str) -> str:
    text = NEGATED_CONTRAST_DURATION_RE.sub(lambda match: f"{match['prefix']}{cardinal(int(match['value']))}", text)
    return COMPLETION_MINUTES_RE.sub(
        lambda match: f"{match['verb']} po {genitive_cardinal(int(match['value']))} min{match['dot']}", text
    )


def date_words(match: re.Match[str]) -> str:
    day_inflector = ordinal if re.search(r"\bbył\s+$", match.string[:match.start()], re.IGNORECASE) else genitive_ordinal
    day = day_inflector(int(match["day"]))
    year = match["year"]
    if year is None:
        return f"{day} {match['month']}"
    return f"{day} {match['month']} {genitive_ordinal(int(year))}{match['roku'] or ''}"


def date_pair_words(match: re.Match[str]) -> str:
    first = genitive_ordinal(int(match["first"]))
    second = ordinal(int(match["second"])) if match["link"].lower() == "na" else genitive_ordinal(int(match["second"]))
    return f"{first} {match['link']} {second} {match['month']}"


def date_range_words(match: re.Match[str]) -> str:
    first = genitive_ordinal(int(match["first"]))
    second = genitive_ordinal(int(match["second"]))
    return f"od {first} do {second} {match['month']}"


def year_words(match: re.Match[str]) -> str:
    return f"{genitive_ordinal(int(match['year']))} roku"


def year_list_words(match: re.Match[str]) -> str:
    first = locative_ordinal(int(match["first"]))
    second = locative_ordinal(int(match["second"]))
    return f"w {first}, {second} roku"


def year_pair_words(match: re.Match[str]) -> str:
    converter = locative_ordinal if match["prefix"] else genitive_ordinal
    first = converter(int(match["first"]))
    second = converter(int(match["second"]))
    return f"{match['prefix'] or ''}{first} {match['link']} {second} roku"


def year_after_in_words(match: re.Match[str]) -> str:
    converter = genitive_ordinal if "ciągu" in match["prefix"].lower() else locative_ordinal
    return f"{match['prefix']} roku {converter(int(match['year']))}"


def short_year_after_words(match: re.Match[str]) -> str:
    following = next((token for token in PARSER(match.string) if token.idx >= match.end()), None)
    if following is not None and following.pos_ in {"NOUN", "SYM", "X"}:
        return match.group()
    return f"{match['prefix']} {ordinal(int(match['year']))}"


def clock_words(match: re.Match[str]) -> str:
    minute = match["minute"]
    minute_words = _minute_words(minute)
    prefix = match["prefix"] or ""
    converter = feminine_genitive_ordinal if prefix.lower() == "około " else feminine_nominative_ordinal
    suffix = f" {minute_words}" if minute_words else ""
    return f"{prefix}{converter(int(match['hour']))}{suffix}"


def governed_clock_words(match: re.Match[str]) -> str:
    accusative = match["prefix"].lower() == "na" or match["prefix"].lower().startswith("przed")
    converter = feminine_accusative_ordinal if accusative else feminine_genitive_ordinal
    hour = converter(int(match["hour"]))
    minute = match["minute"]
    minute_words = _minute_words(minute) if minute is not None else ""
    suffix = f" {minute_words}" if minute_words else ""
    return f"{match['prefix']} {hour}{suffix}"


def o_clock_words(match: re.Match[str]) -> str:
    minute = match["minute"]
    minute_words = _minute_words(minute)
    suffix = f" {minute_words}" if minute_words else ""
    return f"{match['prefix']} {feminine_genitive_ordinal(int(match['hour']))}{suffix}"


def context_comma_clock_words(match: re.Match[str]) -> str:
    prefix = match["prefix"] or ""
    if prefix.endswith(" to") and prefix.split()[0].lower() not in {feminine_nominative_ordinal(value) for value in range(1, 25)}:
        return match.group()
    value = int(match["hour"] or match["bare_hour"] or match["clause_hour"])
    event_clock = re.fullmatch(r"(?:zacz|rozpocz|skończ|kończ)\w*(?: się)?", prefix, re.IGNORECASE)
    hour = feminine_genitive_ordinal(value) if prefix.lower() == "w minucie" or event_clock else feminine_accusative_ordinal(value) if prefix.split()[:1] and prefix.split()[0].endswith("ę") else feminine_nominative_ordinal(value)
    raw_minute = match["minute"] or match["bare_minute"] or match["clause_minute"]
    minute = _minute_words(raw_minute)
    suffix = f" {minute}" if minute else ""
    return f"{prefix}{' o ' if event_clock else ' ' if prefix else ''}{hour}{suffix}"


def expand_context_comma_clocks(text: str, pattern: re.Pattern[str]) -> str:
    adverbial_offsets = {token.idx for token in PARSER(text) if token.dep_ == "obl" and token.head.pos_ == "VERB"}
    return pattern.sub(lambda match: context_comma_clock_words(match)
                       if match["clause_hour"] is None or match.start("clause_hour") == 0 or match.start("clause_hour") in adverbial_offsets else match.group(), text)


def decade_words(match: re.Match[str]) -> str:
    words = ordinal(int(match["decade"]))
    assert words.endswith("y"), f"unsupported Polish decade ordinal: {words}"
    return f"{match['prefix'] or ''}{match['stem']} {words[:-1]}ych{_sentence_dot(match)}"


def lat_decade_words(match: re.Match[str]) -> str:
    words = ordinal(int(match["decade"]))
    assert words.endswith("y"), f"unsupported Polish decade ordinal: {words}"
    return f"{match['prefix']} {words[:-1]}ych{_sentence_dot(match)}"


def lata_decade_words(match: re.Match[str]) -> str:
    words = ordinal(int(match["decade"]))
    assert words.endswith("y"), f"unsupported Polish decade ordinal: {words}"
    return f"{match['prefix']} {words[:-1]}e{_sentence_dot(match)}"


def lat_decade_pair_words(match: re.Match[str]) -> str:
    first = ordinal(int(match["first"]))[:-1]
    second = ordinal(int(match["second"]))[:-1]
    return f"lat {first}ych {match['link']} {second}ych{_sentence_dot(match)}"


def lata_decade_list_words(match: re.Match[str]) -> str:
    first = ordinal(int(match["first"]))[:-1]
    second = ordinal(int(match["second"]))[:-1]
    return f"{match['prefix']} {first}e, {second}e"


def time_range_words(match: re.Match[str]) -> str:
    first = feminine_genitive_ordinal(int(match["first"]))
    second = feminine_genitive_ordinal(int(match["second"]))
    return f"o tej {first} do {second}"


def minute_alternative_words(match: re.Match[str]) -> str:
    first = feminine_accusative_ordinal(int(match["first"]))
    second = feminine_accusative_ordinal(int(match["second"]))
    link = re.search(r"\b(czy|lub|albo)\b", match.group(), re.IGNORECASE)
    assert link is not None, "minute alternative link missing"
    return f"tę {first} {link.group()} {second} minutę"


def _sentence_dot(match: re.Match[str]) -> str:
    remainder = match.string[match.end():].lstrip()
    if remainder.startswith("."):
        return "." if match["dot"] == "." else ""
    sentence_end = not remainder or remainder[:1].isupper() and re.match(r"[A-ZĄĆĘŁŃÓŚŹŻ]{2,}\b", remainder) is None
    roman_century = re.match(r"^[XVI]+ (?:wieku|stulecia)", remainder) is not None
    continued_adverbial = is_continued_decade_clause(match.string, match.start("decade"))
    return "." if match["dot"] == "." and sentence_end and not (roman_century or continued_adverbial) else ""


def is_continued_decade_clause(text: str, decade_offset: int) -> bool:
    document = PARSER(text)
    decade = next((token for token in document if token.idx == decade_offset), None)
    if decade is None:
        return False
    clause_start = max((token.i + 1 for token in document[:decade.i] if token.is_punct), default=0)
    prefix_has_predicate = any(token.pos_ in {"AUX", "VERB"} and token.morph.get("VerbForm") == ["Fin"] for token in document[clause_start:decade.i])
    following = next((token for token in document[decade.i + 1:] if not token.is_punct), None)
    return (not prefix_has_predicate and following is not None
            and (following.dep_ == "nsubj" or following.pos_ in {"NOUN", "PROPN", "PRON"} and following.morph.get("Case") == ["Nom"]))


def _minute_words(minute: str) -> str:
    return "" if minute == "00" else f"zero {cardinal(int(minute))}" if minute.startswith("0") else cardinal(int(minute))
