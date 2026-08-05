import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path


LOCALE_ALIASES = {
    "ga": "ga-IE",
    "hy": "hy-AM",
    "ne": "ne-NP",
    "pa": "pa-IN",
    "sv": "sv-SE",
    "zh": "zh-CN",
    "zh-yue": "yue",
}
SPLITS = ("train", "dev", "test", "other", "invalidated")
VALID_SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class SplitCounts:
    train: int
    dev: int
    test: int
    other: int
    invalidated: int

    def count(self, split: str) -> int:
        return int(getattr(self, split))


@dataclass(frozen=True)
class LocaleSpec:
    language: str
    hf_locale: str
    part: int
    target_hours: float
    available_hours: float
    validated_clips: int
    metadata_rows: SplitCounts
    shards: SplitCounts

    @property
    def target_seconds(self) -> float:
        return self.target_hours * 3600.0

    @property
    def average_clip_seconds(self) -> float:
        return self.available_hours * 3600.0 / self.validated_clips


def parse_release_stats(path: Path) -> dict[str, object]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "STATS"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1, "release_stats.py must assign STATS once"
    return ast.literal_eval(assignments[0].value)


def load_specs(
    repository_root: Path,
    release_stats_path: Path,
    shards_path: Path,
) -> list[LocaleSpec]:
    assignments = _link_assignments(
        repository_root / "imports" / "mozilla-common-voice-26-terms-links.md"
    )
    targets = _waterfill_targets(
        repository_root / "imports" / "waterfill-75h-by-language-dataset.md"
    )
    stats = parse_release_stats(release_stats_path)["locales"]
    shard_data = json.loads(shards_path.read_text(encoding="utf-8"))
    specs = []
    for language, part in assignments:
        hf_locale = (
            LOCALE_ALIASES[language]
            if language in LOCALE_ALIASES
            else language
        )
        locale_stats = stats[hf_locale]
        buckets = locale_stats["buckets"]
        available_hours = float(locale_stats["validHrs"])
        target_hours = min(targets[language], available_hours, 50.0)
        assert target_hours > 0.0, f"{language}: target must be positive"
        specs.append(
            LocaleSpec(
                language=language,
                hf_locale=hf_locale,
                part=part,
                target_hours=target_hours,
                available_hours=available_hours,
                validated_clips=int(buckets["validated"]),
                metadata_rows=_split_counts(buckets),
                shards=_split_counts(shard_data[hf_locale]),
            )
        )
    assert len(specs) == 42
    assert len({spec.language for spec in specs}) == 42
    assert {
        part: sum(spec.part == part for spec in specs)
        for part in (1, 2, 3)
    } == {1: 21, 2: 9, 3: 12}
    return specs


def specs_for_part(specs: list[LocaleSpec], part: int) -> list[LocaleSpec]:
    selected = [spec for spec in specs if spec.part == part]
    return sorted(
        selected,
        key=lambda spec: (spec.target_hours, spec.available_hours),
        reverse=True,
    )


def _split_counts(values: dict[str, object]) -> SplitCounts:
    return SplitCounts(
        train=int(values["train"]),
        dev=int(values["dev"]),
        test=int(values["test"]),
        other=int(values["other"]),
        invalidated=int(values["invalidated"]),
    )


def _link_assignments(path: Path) -> list[tuple[str, int]]:
    assignments = []
    part = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.fullmatch(r"## Part ([1-4]).*", line)
        if heading:
            source_part = int(heading.group(1))
            part = 1 if source_part == 4 else source_part
        entry = re.match(r"- `([^`]+)`", line)
        if entry:
            assert part in (1, 2, 3)
            assignments.append((entry.group(1), part))
    return assignments


def _waterfill_targets(path: Path) -> dict[str, float]:
    targets = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [
            field.strip()
            for field in line.strip().strip("|").split("|")
        ]
        if (
            len(fields) >= 7
            and fields[1].startswith("`")
            and fields[2] in ("CV26", "CV26 Scripted")
        ):
            targets[fields[1].strip("`")] = float(fields[6])
    return targets
