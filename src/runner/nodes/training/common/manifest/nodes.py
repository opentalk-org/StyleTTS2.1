from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode, Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import ASSET_BUNDLE, CHECKPOINT_REF, JSON, TRAINING_MANIFEST
from runner.nodes.models import AssetBundleRef, AudioRecordRef, AudioSegment, CheckpointRef, TrainingManifest, stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud

DEFAULT_MANIFEST_OUTPUT_DIR = Path("data/training/manifests")


class BuildTrainingManifestSettings(StrictSettings):
    validation_samples: int = Field(default=32, title="Validation samples", ge=1, le=512)
    output_dir: Path = Field(default=DEFAULT_MANIFEST_OUTPUT_DIR, title="Output directory")
    root_path: str = Field(default="", title="Manifest root path")


@dataclass(frozen=True)
class ManifestLine:
    audio_id: UUID
    value: str
    phon: str
    speaker: str


class BuildTrainingManifestNode(Node):
    NODE_TYPE = "BuildTrainingManifest"
    CATEGORY = "Training / Preparation"
    SETTINGS = BuildTrainingManifestSettings
    INPUTS = {
        "audio_file_ids": Port("audio_file_ids", JSON),
        "base_checkpoint": Port("base_checkpoint", CHECKPOINT_REF, join_mode=JoinMode.BROADCAST),
        "assets": Port("assets", ASSET_BUNDLE, join_mode=JoinMode.BROADCAST, optional=True, default=None),
        "phoneme_alphabet": Port("phoneme_alphabet", JSON, join_mode=JoinMode.BROADCAST, optional=True, default={}),
    }
    OUTPUTS = {"training_manifest": Port("training_manifest", TRAINING_MANIFEST)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            selection = audio_file_selection(inputs["audio_file_ids"])
            settings = manifest_settings_for_run(self.settings, context.run_id)
            outputs.append({
                "training_manifest": build_training_manifest(
                    dataset_id=selection.dataset_id,
                    segments=segments_from_audio_file_ids(selection.audio_file_ids),
                    base_checkpoint=_typed_checkpoint(inputs["base_checkpoint"]),
                    assets=_typed_assets(inputs["assets"]),
                    phoneme_alphabet=phoneme_alphabet_symbols(inputs["phoneme_alphabet"]),
                    settings=settings,
                )
            })
        return outputs


@dataclass(frozen=True)
class AudioFileSelection:
    dataset_id: UUID
    audio_file_ids: list[UUID]


def build_training_manifest(
    dataset_id: UUID,
    segments: list[AudioSegment],
    base_checkpoint: CheckpointRef,
    assets: AssetBundleRef | None,
    phoneme_alphabet: list[str],
    settings: BuildTrainingManifestSettings,
) -> TrainingManifest:
    audio_dir = _manifest_audio_dir(settings)
    groups = _usable_groups(segments, audio_dir)
    if len(groups) < 2:
        raise ValueError("training manifest requires at least 2 usable source audio groups")

    ordered_audio_ids = list(groups)
    validation_count = min(settings.validation_samples, len(ordered_audio_ids) - 1)
    if validation_count < 1:
        raise ValueError("training manifest requires at least 1 validation item")

    validation_ids = set(ordered_audio_ids[-validation_count:])
    train_lines = _manifest_lines(groups, ordered_audio_ids[:-validation_count], settings.root_path, audio_dir)
    validation_lines = _manifest_lines(groups, ordered_audio_ids[-validation_count:], settings.root_path, audio_dir)
    if not train_lines:
        raise ValueError("training manifest requires at least 1 training item")
    if not validation_lines:
        raise ValueError("training manifest requires at least 1 validation item")

    lists_dir = Path(settings.output_dir) / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    train_path = lists_dir / "train.txt"
    validation_path = lists_dir / "validation.txt"
    _write_manifest(train_path, train_lines)
    _write_manifest(validation_path, validation_lines)

    sidecar_path = lists_dir / "segments.jsonl"
    all_lines = train_lines + validation_lines
    _write_text_sidecar(sidecar_path, groups, ordered_audio_ids)

    manifest_id = stable_id("training_manifest", dataset_id, base_checkpoint.id, len(all_lines))
    return TrainingManifest(
        dataset_id=dataset_id,
        audio_file_ids=ordered_audio_ids,
        base_checkpoint=base_checkpoint,
        phoneme_alphabet=phoneme_alphabet,
        id=manifest_id,
        lineage_id=manifest_id,
        assets=assets,
        metadata={
            "train_manifest_path": str(train_path),
            "validation_manifest_path": str(validation_path),
            "segment_text_path": str(sidecar_path),
            "train_count": len(train_lines),
            "validation_count": len(validation_lines),
            "segment_count": sum(len(items) for items in groups.values()),
            "audio_count": len(all_lines),
            "root_path": settings.root_path,
            "audio_dir": str(audio_dir),
            "ordered_audio_ids": [str(audio_id) for audio_id in ordered_audio_ids],
            "validation_audio_ids": [str(audio_id) for audio_id in ordered_audio_ids if audio_id in validation_ids],
        },
    )


def phoneme_alphabet_symbols(value: dict) -> list[str]:
    symbols = value["symbols"] if "symbols" in value else ""
    if isinstance(symbols, str):
        return [part for part in symbols.split(" ") if part]
    if isinstance(symbols, list):
        return [str(part) for part in symbols]
    raise TypeError("phoneme alphabet symbols must be a string or list")


def manifest_settings_for_run(settings: BuildTrainingManifestSettings, run_id: object) -> BuildTrainingManifestSettings:
    if settings.output_dir != DEFAULT_MANIFEST_OUTPUT_DIR:
        return settings
    return settings.model_copy(update={"output_dir": DEFAULT_MANIFEST_OUTPUT_DIR / str(run_id)})


def segments_from_audio_file_ids(audio_file_ids: list[UUID]) -> list[AudioSegment]:
    segments: list[AudioSegment] = []
    with database_session() as session:
        for audio_file_id in audio_file_ids:
            item = audio_crud.get_audio_file(session, audio_file_id)
            ref = AudioRecordRef(item.id, item.name, item.duration, item.byte_length, item.virtual, item.metadata_)
            segments.extend(
                _audio_segment_from_dict(ref, segment)
                for segment in audio_crud.list_audio_segments(session, item.id)
            )
    return segments


def audio_file_selection(value: dict[str, Any]) -> AudioFileSelection:
    dataset_id = UUID(str(value["dataset_id"]))
    raw_ids = value["ids"]
    if not isinstance(raw_ids, list):
        raise TypeError("audio_file_ids.ids must be a list")
    return AudioFileSelection(dataset_id, [UUID(str(audio_file_id)) for audio_file_id in raw_ids])


def _audio_segment_from_dict(ref: AudioRecordRef, segment: dict[str, Any]) -> AudioSegment:
    segment_id = str(segment["id"])
    metadata = dict(segment["metadata"]) if isinstance(segment.get("metadata"), dict) else {}
    metadata.setdefault("type_", _segment_type(segment))
    return AudioSegment(
        source_audio_id=ref.audio_file_id,
        name=ref.name,
        start=float(segment["start"]),
        end=float(segment["end"]),
        sample_rate=int(ref.metadata["sample_rate"]),
        channels=int(ref.metadata["channels"]) if "channels" in ref.metadata else 1,
        text=str(segment["text"]),
        phon=str(segment["phon"]),
        id=stable_id("segment", ref.audio_file_id, segment_id),
        lineage_id=stable_id("segment_lineage", ref.audio_file_id, segment_id),
        segment_id=segment_id,
        speaker=str(segment["speaker"]) if "speaker" in segment else None,
        voice_id=_optional_uuid(segment["voice_id"]) if "voice_id" in segment else None,
        metadata=metadata,
    )


def _segment_type(segment: dict[str, Any]) -> str:
    if segment.get("type_"):
        return str(segment["type_"])
    metadata = segment.get("metadata")
    if isinstance(metadata, dict):
        if metadata.get("type_"):
            return str(metadata["type_"])
        if metadata.get("model"):
            return str(metadata["model"])
    return "manual"


def _usable_groups(segments: list[AudioSegment], audio_dir: Path) -> dict[UUID, list[AudioSegment]]:
    groups: dict[UUID, list[AudioSegment]] = {}
    for segment in sorted(segments, key=_segment_sort_key):
        if not segment.phon.strip() or segment.start >= segment.end:
            continue
        groups.setdefault(segment.source_audio_id, []).append(segment)
    usable = {audio_id: items for audio_id, items in groups.items() if items}
    _materialize_audio_files(usable, audio_dir)
    return usable


def _segment_sort_key(segment: AudioSegment) -> tuple[str, float, float, str]:
    segment_key = segment.segment_id if segment.segment_id is not None else segment.id
    return (str(segment.source_audio_id), segment.start, segment.end, segment_key)


def _manifest_lines(
    groups: dict[UUID, list[AudioSegment]],
    audio_ids: list[UUID],
    root_path: str,
    audio_dir: Path,
) -> list[ManifestLine]:
    lines: list[ManifestLine] = []
    for audio_id in audio_ids:
        segments = groups[audio_id]
        lines.append(
            ManifestLine(
                audio_id=audio_id,
                value=_manifest_audio_value(audio_id, root_path, audio_dir),
                phon=" ".join(segment.phon.strip() for segment in segments),
                speaker=_speaker_key(segments[0]),
            )
        )
    return lines


def _manifest_audio_value(audio_id: UUID, root_path: str, audio_dir: Path) -> str:
    filename = f"{audio_id}.wav"
    if root_path:
        return str(Path(root_path) / filename)
    return str((audio_dir / filename).resolve())


def _manifest_audio_dir(settings: BuildTrainingManifestSettings) -> Path:
    if settings.root_path:
        return Path(settings.root_path)
    return Path(settings.output_dir) / "audio"


def _speaker_key(segment: AudioSegment) -> str:
    if segment.voice_id is not None:
        return str(segment.voice_id)
    if segment.speaker is not None and segment.speaker.strip():
        return segment.speaker.strip()
    return "0"


def _materialize_audio_files(groups: dict[UUID, list[AudioSegment]], audio_dir: Path) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    with database_session() as session:
        for audio_id in groups:
            target = audio_dir / f"{audio_id}.wav"
            target.write_bytes(audio_crud.read_audio_file(session, audio_id))


def _write_manifest(path: Path, lines: list[ManifestLine]) -> None:
    content = "".join(f"{line.value}|{line.phon}|{line.speaker}\n" for line in lines)
    path.write_text(content, encoding="utf-8")


def _write_text_sidecar(path: Path, groups: dict[UUID, list[AudioSegment]], audio_ids: list[UUID]) -> None:
    rows = [_segment_sidecar_row(segment) for audio_id in audio_ids for segment in groups[audio_id]]
    content = "".join(json.dumps(row, default=str, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def _segment_sidecar_row(segment: AudioSegment) -> dict[str, object]:
    return {
        "audio_id": str(segment.source_audio_id),
        "segment_id": segment.segment_id,
        "runtime_segment_id": segment.id,
        "lineage_id": segment.lineage_id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "phon": segment.phon,
        "speaker": segment.speaker,
        "voice_id": str(segment.voice_id) if segment.voice_id is not None else None,
        "metadata": segment.metadata,
    }


def _typed_checkpoint(value: CheckpointRef | dict) -> CheckpointRef:
    if isinstance(value, CheckpointRef):
        return value
    raise TypeError("BuildTrainingManifest requires a resolved CheckpointRef for base_checkpoint")


def _typed_assets(value: AssetBundleRef | dict | None) -> AssetBundleRef | None:
    if value is None or isinstance(value, AssetBundleRef):
        return value
    raise TypeError("BuildTrainingManifest requires resolved AssetBundleRef values for assets")


def _optional_uuid(value: object) -> UUID | None:
    if value is None or value == "":
        return None
    return UUID(str(value))
