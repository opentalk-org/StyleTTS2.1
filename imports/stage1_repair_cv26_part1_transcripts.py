import argparse
import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from shared.audio_annotations import AudioAnnotations
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.schemas import AudioUpdate
from shared.db.datasets import crud as dataset_crud


EXPECTED_CORRECTIONS = 362
LANGUAGES = ("nl", "el", "ky", "ltg", "or", "tr", "vi")
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "stage1"
    / "common_voice_part1"
    / "data.json"
)


@dataclass(frozen=True)
class TranscriptCorrection:
    source_id: str
    old_text: str
    corrected_text: str


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair proven Common Voice 26 Part 1 transcript corruption"
    )
    parser.add_argument("--cv22-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit-backend", action="store_true")
    parser.add_argument("--apply-backend", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def load_reference(root: Path) -> dict[str, dict[str, str]]:
    reference: dict[str, dict[str, str]] = {}
    for language in LANGUAGES:
        sentences: dict[str, str] = {}
        paths = sorted((root / language).glob("*.tsv"))
        assert len(paths) == 5, f"{language}: expected five CV22 split TSVs"
        for path in paths:
            with path.open(encoding="utf-8", newline="") as source:
                rows = csv.DictReader(
                    source,
                    delimiter="\t",
                    quoting=csv.QUOTE_NONE,
                )
                for row in rows:
                    sentence_id = row["sentence_id"]
                    sentence = row["sentence"]
                    if sentence_id in sentences:
                        assert sentences[sentence_id] == sentence, (
                            f"{language}:{sentence_id}: conflicting CV22 sentences"
                        )
                    sentences[sentence_id] = sentence
        reference[language] = sentences
    return reference


def find_corrections(
    payload: dict[str, object],
    reference: dict[str, dict[str, str]],
) -> list[TranscriptCorrection]:
    corrections = []
    for record in payload["audio_files"]:
        language = record["language"]
        if language not in reference or not record["source_id"].startswith("cv26:"):
            continue
        publisher_row = record["metadata"]["publisher_row"]
        sentence_id = publisher_row["sentence_id"]
        if sentence_id not in reference[language]:
            continue
        old_text = record["segments"][0]["text"]
        assert publisher_row["sentence"] == old_text, (
            f"{record['source_id']}: segment and publisher transcript differ"
        )
        corrected_text = reference[language][sentence_id]
        if old_text != corrected_text:
            corrections.append(
                TranscriptCorrection(
                    source_id=record["source_id"],
                    old_text=old_text,
                    corrected_text=corrected_text,
                )
            )
    return corrections


def backend_items(
    payload: dict[str, object],
    corrections: list[TranscriptCorrection],
) -> dict[str, object]:
    expected_sources = {correction.source_id for correction in corrections}
    with database_session() as session:
        dataset = dataset_crud.get_dataset_by_name(
            session,
            payload["dataset"]["name"],
        )
        assert dataset is not None, "Part 1 backend dataset not found"
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
                f"{correction.source_id}: unexpected backend segment text"
            )
            assert (
                item.metadata_["publisher_row"]["sentence"]
                == correction.old_text
            ), f"{correction.source_id}: unexpected backend publisher transcript"
        return items


def apply_backend(
    payload: dict[str, object],
    corrections: list[TranscriptCorrection],
) -> None:
    items = backend_items(payload, corrections)
    updates = {}
    for correction in corrections:
        item = items[correction.source_id]
        metadata = copy.deepcopy(item.metadata_)
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


def write_manifest(
    path: Path,
    payload: dict[str, object],
    corrections: list[TranscriptCorrection],
) -> None:
    by_source = {correction.source_id: correction for correction in corrections}
    for record in payload["audio_files"]:
        source_id = record["source_id"]
        if source_id not in by_source:
            continue
        corrected_text = by_source[source_id].corrected_text
        record["segments"][0]["text"] = corrected_text
        record["metadata"]["publisher_row"]["sentence"] = corrected_text
    temporary = path.parent / f"{path.name}.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(f"MANIFEST_UPDATED records={len(corrections)}", flush=True)


def main() -> None:
    args = arguments()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    reference = load_reference(args.cv22_root)
    corrections = find_corrections(payload, reference)
    print(
        f"AUDIT proven_corrections={len(corrections)} "
        f"unique_sentence_ids={len({item.corrected_text for item in corrections})}",
        flush=True,
    )
    if args.audit_backend:
        backend_items(payload, corrections)
        print(f"BACKEND_AUDITED records={len(corrections)}", flush=True)
    if args.apply_backend or args.write_manifest:
        assert len(corrections) == EXPECTED_CORRECTIONS, (
            f"expected {EXPECTED_CORRECTIONS} corrections, found {len(corrections)}"
        )
    if args.apply_backend:
        apply_backend(payload, corrections)
    if args.write_manifest:
        write_manifest(args.manifest, payload, corrections)


if __name__ == "__main__":
    main()
