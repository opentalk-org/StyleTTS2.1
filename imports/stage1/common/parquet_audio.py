import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from imports.stage1.common.audio import normalize_audio_bytes
from imports.stage1.common.schema import AudioRecord, DatasetManifest, DatasetRecord, SegmentRecord


@dataclass(frozen=True)
class ParquetSource:
    dataset_name: str
    source_url: str
    language: str
    audio_field: str
    text_field: str
    speaker_field: str | None
    style_field: str | None


@dataclass(frozen=True)
class Candidate:
    shard: Path
    row_index: int
    row: dict


def record_language(candidate: Candidate, source: ParquetSource) -> str:
    return str(candidate.row[source.language[1:]]) if source.language.startswith("@") else source.language


def inventory(repository: Path, source: ParquetSource) -> list[Candidate]:
    candidates = []
    for shard in sorted(repository.rglob("*.parquet")):
        for row_index, row in enumerate(pq.read_table(shard).to_pylist()):
            assert row[source.audio_field]["bytes"], f"empty audio: {shard}:{row_index}"
            assert str(row[source.text_field]).strip(), f"empty transcript: {shard}:{row_index}"
            candidates.append(Candidate(shard, row_index, row))
    assert candidates, f"no Parquet audio records found under {repository}"
    return candidates


def make_record(candidate: Candidate, repository: Path, stage_root: Path,
                source: ParquetSource) -> AudioRecord:
    shard = candidate.shard.relative_to(repository).as_posix()
    source_id = f"{shard}:{candidate.row_index}"
    output_id = hashlib.sha1(source_id.encode()).hexdigest()[:16]
    language = record_language(candidate, source)
    destination = stage_root / "wavs" / f"{language}_{output_id}.wav"
    audio = candidate.row[source.audio_field]
    duration = normalize_audio_bytes(audio["bytes"], destination)
    publisher_row = {key: value for key, value in candidate.row.items() if key != source.audio_field}
    publisher_row[source.audio_field] = {"path": audio["path"], "byte_length": len(audio["bytes"])}
    speaker = None if source.speaker_field is None else str(candidate.row[source.speaker_field])
    style = None if source.style_field is None else str(candidate.row[source.style_field])
    return AudioRecord(
        path=destination.relative_to(stage_root).as_posix(), source_id=source_id,
        duration=duration, language=language, speaker_id=speaker,
        style_prompt=style, voice_prompt=None, score=None, accuracy=None,
        segments=[SegmentRecord(start=0.0, end=duration, text=str(candidate.row[source.text_field]).strip(),
                                source="dataset", score=None, accuracy=None, alignment=[])],
        metadata={"source_dataset": source.dataset_name, "source_url": source.source_url,
                  "parquet_shard": shard, "row_index": candidate.row_index,
                  "publisher_row": publisher_row},
    )


def prepare(repository: Path, stage_root: Path, source: ParquetSource, workers: int) -> None:
    candidates = inventory(repository, source)
    shutil.rmtree(stage_root / "wavs", ignore_errors=True)
    (stage_root / "wavs").mkdir(parents=True)
    worker = partial(make_record, repository=repository, stage_root=stage_root, source=source)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(tqdm(executor.map(worker, candidates), total=len(candidates), desc=source.dataset_name))
    language_limits = {}
    for record in records:
        language_limits[record.language] = language_limits.get(record.language, 0.0) + record.duration / 3600.0
    manifest = DatasetManifest(
        dataset=DatasetRecord(name=source.dataset_name,
                              language_limits_hours=language_limits, source_url=source.source_url),
        audio_files=records,
    )
    temporary = stage_root / "data.json.tmp"
    temporary.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")
    temporary.replace(stage_root / "data.json")
    shutil.rmtree(stage_root / "tmp")
    (stage_root / "tmp").mkdir()


def optional_field(value: str) -> str | None:
    return None if value == "-" else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a publisher Parquet audio dataset")
    parser.add_argument("repository", type=Path)
    parser.add_argument("stage_root", type=Path)
    parser.add_argument("dataset_name")
    parser.add_argument("source_url")
    parser.add_argument("language")
    parser.add_argument("audio_field")
    parser.add_argument("text_field")
    parser.add_argument("speaker_field")
    parser.add_argument("style_field")
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()
    source = ParquetSource(
        dataset_name=arguments.dataset_name, source_url=arguments.source_url,
        language=arguments.language, audio_field=arguments.audio_field,
        text_field=arguments.text_field, speaker_field=optional_field(arguments.speaker_field),
        style_field=optional_field(arguments.style_field),
    )
    prepare(arguments.repository, arguments.stage_root, source, arguments.workers)


if __name__ == "__main__":
    main()
