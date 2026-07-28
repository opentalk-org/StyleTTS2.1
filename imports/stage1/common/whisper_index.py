import argparse
import json
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Any

from imports.stage1.common.schema import AudioRecord, DatasetManifest, DatasetRecord, SegmentRecord


MODEL_KIND = "whisper"
MODEL_ID = "turbo"


def checkpoint_path() -> Path:
    from shared.db import database_session
    from shared.db.assets import crud as asset_crud

    with database_session() as session:
        matches = [
            checkpoint
            for checkpoint in asset_crud.list_checkpoints(session)
            if checkpoint.type_ == MODEL_KIND
            and checkpoint.metadata_["model_kind"] == MODEL_KIND
            and checkpoint.metadata_["model_id"] == MODEL_ID
        ]
        assert len(matches) == 1, f"expected one Whisper Turbo checkpoint, found {len(matches)}"
        return asset_crud.get_checkpoint_path(session, matches[0].id)


def load_journal(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source_id = str(row["source_id"])
        assert source_id not in rows, f"duplicate transcript journal source ID: {source_id}"
        rows[source_id] = list(row["segments"])
    return rows


def transcribe_part(arguments: tuple[int, list[dict[str, Any]], str, str, str]) -> tuple[int, int]:
    part, records, stage_root_raw, checkpoint_raw, language = arguments
    from runner.nodes.asr.whisper import load_whisper_model, transcribe_wav_to_segments

    stage_root = Path(stage_root_raw)
    journal = stage_root / "tmp" / "transcripts" / f"part-{part:02d}.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    completed = load_journal(journal)
    model = load_whisper_model(Path(checkpoint_raw))
    written = 0
    with journal.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            source_id = str(record["source_id"])
            if source_id in completed:
                continue
            spans = transcribe_wav_to_segments(
                model,
                stage_root / str(record["path"]),
                float(record["duration"]),
                language,
            )
            segments = [
                {"start": start, "end": end, "text": text, "score": score}
                for start, end, text, score in spans
            ]
            handle.write(json.dumps({"source_id": source_id, "segments": segments}, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            if written % 25 == 0:
                os.fsync(handle.fileno())
            if (index + 1) % 100 == 0:
                print(f"TRANSCRIBE part={part} records={index + 1}/{len(records)}", flush=True)
        os.fsync(handle.fileno())
    return part, written


def all_transcripts(stage_root: Path, records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    transcripts: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((stage_root / "tmp" / "transcripts").glob("part-*.jsonl")):
        for source_id, segments in load_journal(path).items():
            assert source_id not in transcripts, f"duplicate transcript across journals: {source_id}"
            transcripts[source_id] = segments
    requested = {str(record["source_id"]) for record in records}
    assert set(transcripts) == requested, f"transcribed {len(transcripts)} of {len(requested)} records"
    empty = [source_id for source_id, segments in transcripts.items() if not segments]
    assert not empty, f"Whisper returned no transcript for {len(empty)} records, first={empty[0]}"
    return transcripts


def make_manifest(
    stage_root: Path,
    dataset_name: str,
    language: str,
    source_url: str,
    records: list[dict[str, Any]],
    transcripts: dict[str, list[dict[str, Any]]],
) -> None:
    audio_files = []
    for record in records:
        source_id = str(record["source_id"])
        duration = float(record["duration"])
        segments = [
            SegmentRecord(
                start=max(0.0, float(segment["start"])),
                end=min(duration, float(segment["end"])),
                text=str(segment["text"]).strip(),
                source=f"whisper:{MODEL_ID}",
                score=segment["score"],
                accuracy=None,
                alignment=[],
            )
            for segment in transcripts[source_id]
        ]
        publisher_row = {key: value for key, value in record.items() if key not in {"path", "duration"}}
        audio_files.append(AudioRecord(
            path=str(record["path"]), source_id=source_id, duration=duration,
            language=language, speaker_id=str(record["speaker_id"]),
            style_prompt=str(record["emotion"]), voice_prompt=None, score=None, accuracy=None,
            segments=segments,
            metadata={"source_dataset": dataset_name, "source_url": source_url, "publisher_row": publisher_row},
        ))
    manifest = DatasetManifest(
        dataset=DatasetRecord(
            name=dataset_name,
            language_limits_hours={language: sum(record.duration for record in audio_files) / 3600.0},
            source_url=source_url,
        ),
        audio_files=audio_files,
    )
    temporary = stage_root / "data.json.tmp"
    temporary.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")
    temporary.replace(stage_root / "data.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a prepared Stage 1 source index with Whisper Turbo")
    parser.add_argument("stage_root", type=Path)
    parser.add_argument("dataset_name")
    parser.add_argument("language")
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    index = json.loads((arguments.stage_root / "source-index.json").read_text(encoding="utf-8"))
    records = list(index["records"])
    checkpoint = checkpoint_path()
    work = [
        (part, records[part::arguments.workers], str(arguments.stage_root), str(checkpoint), arguments.language)
        for part in range(arguments.workers)
    ]
    context = multiprocessing.get_context("spawn")
    with context.Pool(arguments.workers) as pool:
        for part, written in pool.imap_unordered(transcribe_part, work):
            print(f"TRANSCRIBE_COMPLETE part={part} written={written}", flush=True)
    transcripts = all_transcripts(arguments.stage_root, records)
    make_manifest(
        arguments.stage_root,
        arguments.dataset_name,
        arguments.language,
        str(index["source_url"]),
        records,
        transcripts,
    )
    shutil.rmtree(arguments.stage_root / "tmp")
    (arguments.stage_root / "tmp").mkdir()
    print(f"TRANSCRIBED records={len(records)} workers={arguments.workers}", flush=True)


if __name__ == "__main__":
    main()
