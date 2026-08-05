import fcntl
import json
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Iterator

from imports.stage1.common.hf22_catalog import LocaleSpec
from imports.stage1.common.schema import AudioRecord, DatasetManifest


class State(StrEnum):
    ACTIVE = "ACTIVE"
    COMPLETE_LOCAL = "COMPLETE_LOCAL"
    TIME_LIMIT = "TIME_LIMIT"
    DISK_LIMIT = "DISK_LIMIT"
    NOT_POSSIBLE = "NOT_POSSIBLE"
    FAILED = "FAILED"


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ManifestStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.parent / f"{path.name}.hf22.lock"

    def locale_records(self, spec: LocaleSpec) -> list[AudioRecord]:
        with _exclusive_lock(self.lock_path):
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        prefix = f"cv22:{spec.language}:"
        return [
            AudioRecord.model_validate(record)
            for record in payload["audio_files"]
            if record["source_id"].startswith(prefix)
        ]

    def merge(
        self,
        spec: LocaleSpec,
        records: list[AudioRecord],
    ) -> None:
        with _exclusive_lock(self.lock_path):
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            prefix = f"cv22:{spec.language}:"
            retained = [
                record
                for record in payload["audio_files"]
                if not record["source_id"].startswith(prefix)
            ]
            by_source = {record.source_id: record for record in records}
            retained.extend(
                record.model_dump(mode="json")
                for record in by_source.values()
            )
            payload["audio_files"] = retained
            actual_hours = (
                sum(record.duration for record in by_source.values()) / 3600.0
            )
            payload["dataset"]["language_limits_hours"][spec.language] = (
                actual_hours
            )
            temporary = self.path.parent / f"{self.path.name}.tmp"
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.path)

    def validate(self) -> int:
        with _exclusive_lock(self.lock_path):
            manifest = DatasetManifest.model_validate_json(
                self.path.read_text(encoding="utf-8")
            )
        return len(manifest.audio_files)


class StatusStore:
    def __init__(self, part_root: Path, part: int) -> None:
        self.path = part_root / "HF22_STATUS.json"
        self.markdown_path = part_root / "HF22_STATUS.md"
        self.lock_path = part_root / "HF22_STATUS.json.lock"
        self.part = part

    def update(self, spec: LocaleSpec, state: State, **values: object) -> None:
        locale = {
            "language": spec.language,
            "hf_locale": spec.hf_locale,
            "target_hours": spec.target_hours,
            "state": state.value,
            **values,
        }
        with _exclusive_lock(self.lock_path):
            payload = self._load()
            payload["locales"][spec.language] = locale
            temporary = self.path.parent / f"{self.path.name}.tmp"
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.path)
            self._write_markdown(payload)

    def normalize_exhausted(
        self,
        specs: list[LocaleSpec],
        minimum_fraction: float,
    ) -> set[str]:
        by_language = {spec.language: spec for spec in specs}
        with _exclusive_lock(self.lock_path):
            payload = self._load()
            locales = payload["locales"]
            assert isinstance(locales, dict)
            for language, item in locales.items():
                assert isinstance(item, dict)
                if item["state"] != State.NOT_POSSIBLE.value:
                    continue
                spec = by_language[language]
                fraction = float(item["actual_hours"]) / spec.target_hours
                if fraction >= minimum_fraction:
                    item["state"] = State.COMPLETE_LOCAL.value
                    item["error"] = (
                        "accepted after exhausting trusted splits at "
                        f"{fraction:.1%} of target"
                    )
            temporary = self.path.parent / f"{self.path.name}.tmp"
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.path)
            self._write_markdown(payload)
            return {
                language
                for language, item in locales.items()
                if item["state"]
                in (State.COMPLETE_LOCAL.value, State.NOT_POSSIBLE.value)
            }

    def _load(self) -> dict[str, object]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "source": (
                "https://huggingface.co/datasets/fsicoli/common_voice_22_0"
            ),
            "part": self.part,
            "backend_uploaded": False,
            "locales": {},
        }

    def _write_markdown(self, payload: dict[str, object]) -> None:
        locales = payload["locales"]
        assert isinstance(locales, dict)
        lines = [
            "# Common Voice 22 Hugging Face fallback",
            "",
            "Backend uploaded: **no**",
            "",
            "| Language | HF locale | State | Hours | Records |",
            "|---|---|---|---:|---:|",
        ]
        for language in sorted(locales):
            item = locales[language]
            assert isinstance(item, dict)
            lines.append(
                f"| `{language}` | `{item['hf_locale']}` | "
                f"{item['state']} | {float(item['actual_hours']):.4f} | "
                f"{int(item['records'])} |"
            )
        self.markdown_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
