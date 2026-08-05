import argparse
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from imports.stage1.common.schema import DatasetManifest


STAGE_ROOT = Path(__file__).resolve().parent
DURATION_TOLERANCE_SECONDS = 0.001


@dataclass(frozen=True)
class AuditResult:
    slug: str
    audio_count: int
    duration_seconds: float
    languages: tuple[str, ...]


def audit_dataset(root: Path) -> AuditResult:
    manifest_path = root / "data.json"
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    tmp_entries = list((root / "tmp").iterdir())
    if tmp_entries:
        raise ValueError(f"{root.name}: tmp is not empty: {tmp_entries[0].name}")
    referenced = {record.path for record in manifest.audio_files}
    present = {
        path.relative_to(root).as_posix()
        for path in (root / "wavs").glob("*.wav")
    }
    if referenced != present:
        missing = sorted(referenced.difference(present))
        extra = sorted(present.difference(referenced))
        raise ValueError(f"{root.name}: WAV reference mismatch missing={missing[:3]} extra={extra[:3]}")
    measured_duration = 0.0
    declared_duration = 0.0
    for record in manifest.audio_files:
        path = root / record.path
        info = sf.info(path)
        if info.samplerate != 24_000 or info.channels != 1 or info.subtype != "PCM_24":
            raise ValueError(f"{record.source_id}: expected 24 kHz mono PCM-24, got {info}")
        duration = info.frames / info.samplerate
        if abs(duration - record.duration) > DURATION_TOLERANCE_SECONDS:
            raise ValueError(
                f"{record.source_id}: manifest duration {record.duration} differs from WAV {duration}"
            )
        if not record.segments:
            raise ValueError(f"{record.source_id}: no transcript segments")
        measured_duration += duration
        declared_duration += record.duration
    if abs(measured_duration - declared_duration) > DURATION_TOLERANCE_SECONDS * len(manifest.audio_files):
        raise ValueError(f"{root.name}: aggregate manifest and WAV durations differ")
    limits = manifest.dataset.language_limits_hours
    language_duration: dict[str, float] = {language: 0.0 for language in limits}
    for record in manifest.audio_files:
        language_duration[record.language] += record.duration
    return AuditResult(
        slug=root.name,
        audio_count=len(manifest.audio_files),
        duration_seconds=measured_duration,
        languages=tuple(sorted(language_duration)),
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit terminal Stage 1 dataset folders")
    parser.add_argument("slugs", nargs="+")
    return parser.parse_args()


def main() -> None:
    for slug in arguments().slugs:
        result = audit_dataset(STAGE_ROOT / slug)
        print(
            f"AUDITED {result.slug} records={result.audio_count} "
            f"seconds={result.duration_seconds:.6f} languages={','.join(result.languages)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
