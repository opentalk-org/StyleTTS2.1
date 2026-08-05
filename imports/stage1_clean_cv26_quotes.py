import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from shared.audio_annotations import AudioAnnotations
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.schemas import AudioUpdate
from shared.db.datasets import crud as dataset_crud


EXPECTED_CORRECTIONS = 924
DATASET_NAME = "Mozilla Common Voice 26"
STAGE_ROOT = Path(__file__).resolve().parent / "stage1"


@dataclass(frozen=True)
class TranscriptCorrection:
    source_id: str
    part: int
    old_text: str
    corrected_text: str


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove qualifying enclosing quotes from Common Voice transcripts"
    )
    parser.add_argument("--audit-backend", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("/tmp/cv26-quote-cleanup-plan.json"),
    )
    return parser.parse_args()


def remove_majority_quotes(text: str) -> str:
    leading_length = len(text) - len(text.lstrip())
    trailing_start = len(text.rstrip())
    prefix = text[:leading_length]
    core = text[leading_length:trailing_start]
    suffix = text[trailing_start:]
    if len(core) >= 2 and core[0] == '"' and core[-1] == '"':
        core = core[1:-1]
    positions = [index for index, character in enumerate(core) if character == '"']
    content_length = len(core.replace('"', ""))
    qualifying_pairs = [
        (opening, closing)
        for opening, closing in zip(positions[::2], positions[1::2])
        if content_length and (closing - opening - 1) / content_length > 0.5
    ]
    assert len(qualifying_pairs) <= 1, "disjoint quote spans cannot both exceed 50%"
    removals = {
        position
        for pair in qualifying_pairs
        for position in pair
    }
    cleaned = "".join(
        character
        for index, character in enumerate(core)
        if index not in removals
    )
    return prefix + cleaned + suffix


def load_manifests() -> dict[int, tuple[Path, dict[str, object]]]:
    manifests = {}
    for part in (1, 2, 3):
        path = STAGE_ROOT / f"common_voice_part{part}" / "data.json"
        manifests[part] = (
            path,
            json.loads(path.read_text(encoding="utf-8")),
        )
    return manifests


def find_corrections(
    manifests: dict[int, tuple[Path, dict[str, object]]],
) -> list[TranscriptCorrection]:
    corrections = []
    for part, (_, payload) in manifests.items():
        for record in payload["audio_files"]:
            old_text = record["segments"][0]["text"]
            corrected_text = remove_majority_quotes(old_text)
            if old_text == corrected_text:
                continue
            metadata = record["metadata"]
            if "publisher_row" in metadata:
                assert metadata["publisher_row"]["sentence"] == old_text, (
                    f"{record['source_id']}: publisher transcript differs"
                )
            corrections.append(
                TranscriptCorrection(
                    source_id=record["source_id"],
                    part=part,
                    old_text=old_text,
                    corrected_text=corrected_text,
                )
            )
    return corrections


def backend_items(
    corrections: list[TranscriptCorrection],
) -> dict[str, object]:
    expected_sources = {correction.source_id for correction in corrections}
    with database_session() as session:
        dataset = dataset_crud.get_dataset_by_name(session, DATASET_NAME)
        assert dataset is not None, f"{DATASET_NAME}: backend dataset not found"
        items = {
            item.metadata_["stage1_source_id"]: item
            for item in dataset.audio_files
            if item.metadata_["stage1_source_id"] in expected_sources
        }
        assert set(items) == expected_sources, "backend correction sources differ"
        for correction in corrections:
            item = items[correction.source_id]
            assert len(item.segments) == 1, (
                f"{correction.source_id}: expected one backend segment"
            )
            assert item.segments[0]["text"] == correction.old_text, (
                f"{correction.source_id}: unexpected backend transcript"
            )
            if "publisher_row" in item.metadata_:
                assert (
                    item.metadata_["publisher_row"]["sentence"]
                    == correction.old_text
                ), f"{correction.source_id}: unexpected publisher transcript"
        return items


def apply_backend(corrections: list[TranscriptCorrection]) -> None:
    items = backend_items(corrections)
    updates = {}
    for correction in corrections:
        item = items[correction.source_id]
        metadata = copy.deepcopy(item.metadata_)
        if "publisher_row" in metadata:
            metadata["publisher_row"]["sentence"] = correction.corrected_text
        segments = copy.deepcopy(item.segments)
        segments[0]["text"] = correction.corrected_text
        updates[item.id] = AudioUpdate(
            name=item.name,
            wav_bytes=None,
            duration=item.duration,
            annotations=AudioAnnotations(
                speaker_id=item.speaker_id,
                score=item.score,
                accuracy=item.accuracy,
                metadata=metadata,
            ),
            language=item.language,
            style_prompt=item.style_prompt,
            voice_prompt=item.voice_prompt,
            segments=segments,
            virtual=item.virtual,
        )
    with database_session() as session:
        updated = audio_crud.bulk_update_audio_files(session, updates)
    assert set(updated) == set(updates), "backend did not update every transcript"
    print(f"BACKEND_UPDATED records={len(updated)}", flush=True)


def write_manifests(
    manifests: dict[int, tuple[Path, dict[str, object]]],
    corrections: list[TranscriptCorrection],
) -> None:
    by_source = {correction.source_id: correction for correction in corrections}
    for part, (path, payload) in manifests.items():
        changed = 0
        for record in payload["audio_files"]:
            source_id = record["source_id"]
            if source_id not in by_source:
                continue
            correction = by_source[source_id]
            assert correction.part == part
            record["segments"][0]["text"] = correction.corrected_text
            metadata = record["metadata"]
            if "publisher_row" in metadata:
                metadata["publisher_row"]["sentence"] = correction.corrected_text
            changed += 1
        temporary = path.parent / f"{path.name}.tmp"
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        print(f"MANIFEST_UPDATED part={part} records={changed}", flush=True)


def write_plan(path: Path, corrections: list[TranscriptCorrection]) -> None:
    payload = [asdict(correction) for correction in corrections]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"PLAN_WRITTEN path={path} records={len(payload)}", flush=True)


def main() -> None:
    args = arguments()
    manifests = load_manifests()
    corrections = find_corrections(manifests)
    print(
        f"AUDIT corrections={len(corrections)} "
        f"unique_texts={len({item.old_text for item in corrections})}",
        flush=True,
    )
    if args.audit_backend:
        backend_items(corrections)
        print(f"BACKEND_AUDITED records={len(corrections)}", flush=True)
    if args.apply:
        assert len(corrections) == EXPECTED_CORRECTIONS, (
            f"expected {EXPECTED_CORRECTIONS} corrections, found {len(corrections)}"
        )
        write_plan(args.plan, corrections)
        apply_backend(corrections)
        write_manifests(manifests, corrections)


if __name__ == "__main__":
    main()
