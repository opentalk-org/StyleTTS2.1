import concurrent.futures
import csv
import math
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from imports.stage1.common.audio import normalize_audio_bytes
from imports.stage1.common.hf22_catalog import (
    LocaleSpec,
    SPLITS,
    VALID_SPLITS,
)
from imports.stage1.common.hf22_records import build_audio_record
from imports.stage1.common.schema import AudioRecord


@dataclass(frozen=True)
class ClipMetadata:
    split: str
    row: dict[str, str]


@dataclass(frozen=True)
class MetadataIndex:
    clips: dict[str, ClipMetadata]
    row_counts: dict[str, int]
    speaker_count: int


@dataclass(frozen=True)
class ConversionInput:
    spec: LocaleSpec
    split: str
    row: dict[str, str]
    audio_bytes: bytes
    wav_root: Path


@dataclass(frozen=True)
class ConversionOutcome:
    source_id: str
    record: AudioRecord | None
    error: str | None


@dataclass(frozen=True)
class ShardResult:
    records: list[AudioRecord]
    failures: list[str]
    scanned: int
    timed_out: bool


class SpeakerBudget:
    def __init__(
        self,
        spec: LocaleSpec,
        speaker_count: int,
        existing: list[AudioRecord],
    ) -> None:
        desired_speakers = min(max(speaker_count, 1), 2_000)
        seconds_per_speaker = spec.target_seconds / desired_speakers
        self.max_clips = max(
            1,
            math.ceil(
                seconds_per_speaker
                / spec.average_clip_seconds
                * 1.5
            ),
        )
        self.counts: dict[str, int] = {}
        for record in existing:
            if record.speaker_id is not None:
                self.counts[record.speaker_id] = (
                    self.counts[record.speaker_id] + 1
                    if record.speaker_id in self.counts
                    else 1
                )

    def reserve(self, speaker_id: str) -> bool:
        current = self.counts[speaker_id] if speaker_id in self.counts else 0
        if current >= self.max_clips:
            return False
        self.counts[speaker_id] = current + 1
        return True


def read_metadata(
    spec: LocaleSpec,
    paths: dict[str, Path],
) -> MetadataIndex:
    clips = {}
    row_counts = {}
    speakers = set()
    for split in SPLITS:
        count = 0
        with paths[split].open(encoding="utf-8", newline="") as source:
            rows = csv.DictReader(
                source,
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
            )
            for row in rows:
                assert all(value is not None for value in row.values()), (
                    f"{spec.language}:{split}: malformed TSV row"
                )
                count += 1
                if split not in VALID_SPLITS:
                    continue
                filename = row["path"]
                assert filename not in clips, (
                    f"{spec.language}:{filename}: duplicate validated path"
                )
                clips[filename] = ClipMetadata(split=split, row=row)
                speakers.add(row["client_id"])
        row_counts[split] = count
        assert count == spec.metadata_rows.count(split), (
            f"{spec.language}:{split}: expected "
            f"{spec.metadata_rows.count(split)} rows, found {count}"
        )
    return MetadataIndex(
        clips=clips,
        row_counts=row_counts,
        speaker_count=len(speakers),
    )


def process_shard(
    spec: LocaleSpec,
    shard_path: Path,
    split: str,
    metadata: MetadataIndex,
    existing_source_ids: set[str],
    existing_duration: float,
    budget: SpeakerBudget,
    wav_root: Path,
    deadline: float,
    workers: int,
) -> ShardResult:
    records = []
    failures = []
    scanned = 0
    batch = []
    batch_size = workers * 2
    timed_out = False
    with tarfile.open(shard_path, "r:") as archive:
        for member in archive:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            if not member.isfile():
                continue
            filename = Path(member.name).name
            if filename not in metadata.clips:
                continue
            clip = metadata.clips[filename]
            if clip.split != split:
                continue
            scanned += 1
            source_id = f"cv22:{spec.language}:{filename}"
            if source_id in existing_source_ids:
                continue
            speaker_id = f"cv22:{clip.row['client_id']}"
            if not budget.reserve(speaker_id):
                continue
            source = archive.extractfile(member)
            assert source is not None
            batch.append(
                ConversionInput(
                    spec=spec,
                    split=split,
                    row=clip.row,
                    audio_bytes=source.read(),
                    wav_root=wav_root,
                )
            )
            if len(batch) < batch_size:
                continue
            outcomes = _convert_batch(batch, workers)
            _collect(outcomes, records, failures, existing_source_ids)
            batch = []
            duration = existing_duration + sum(
                record.duration for record in records
            )
            if duration >= spec.target_seconds:
                break
        if batch and (
            existing_duration
            + sum(record.duration for record in records)
            < spec.target_seconds
        ):
            outcomes = _convert_batch(batch, workers)
            _collect(outcomes, records, failures, existing_source_ids)
    return ShardResult(
        records=records,
        failures=failures,
        scanned=scanned,
        timed_out=timed_out,
    )


def _convert_batch(
    batch: list[ConversionInput],
    workers: int,
) -> list[ConversionOutcome]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_convert, batch))


def _collect(
    outcomes: list[ConversionOutcome],
    records: list[AudioRecord],
    failures: list[str],
    source_ids: set[str],
) -> None:
    for outcome in outcomes:
        if outcome.record is None:
            assert outcome.error is not None
            failures.append(f"{outcome.source_id}: {outcome.error}")
        else:
            records.append(outcome.record)
            source_ids.add(outcome.source_id)


def _convert(item: ConversionInput) -> ConversionOutcome:
    filename = item.row["path"]
    source_id = f"cv22:{item.spec.language}:{filename}"
    destination = (
        item.wav_root
        / f"cv22_{item.spec.language}_{Path(filename).stem}.wav"
    )
    try:
        if destination.exists():
            info = sf.info(destination)
            assert (
                info.samplerate,
                info.channels,
                info.subtype,
            ) == (24_000, 1, "PCM_24")
            duration = info.frames / info.samplerate
        else:
            duration = normalize_audio_bytes(item.audio_bytes, destination)
        record = build_audio_record(
            item.spec,
            item.split,
            item.row,
            destination,
            duration,
        )
        return ConversionOutcome(source_id, record, None)
    except Exception as error:
        destination.unlink(missing_ok=True)
        return ConversionOutcome(source_id, None, repr(error))
